"""Dense and sparse retrieval models, compared offline on the golden sets.

Retrieval is the stage nothing downstream can make up for: a gold chunk that is
not among the result_limit candidates is lost to the reranker and the reader
alike. This scores candidate encoders the way the index uses them - exact dense
cosine, sparse dot product (with Qdrant's IDF where the model relies on it), and
the RRF of the two branches at depth LIMIT - on every aligned question of a
golden set, over the whole library and over the question's own paper.

Arms, each read against the hybrid shipped before this comparison (bge-base-en-v1.5 + BM25;
EmbeddingGemma + BM25 replaced it):

- ``dense:<model>`` and ``sparse:<model>`` alone;
- ``hybrid:<dense>+<sparse>`` for every pair.

Encodings are cached under ``<golden-dir>/retrieval_cache/``, so the report can
be rebuilt, and models added, without re-encoding the others. Encoding time is
recorded per chunk and per query: it is what a user pays to index a library and
to ask a question.

Usage:
  python -m benchmark.retrieval_models --golden-dir output_qasa
"""

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# appended, never prepended: the repo root must keep priority over this directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from benchmark.paraphrase_retrieval import (
    LIMIT,
    bm25,
    idf_table,
    inverted_index,
    retrieve,
)
from benchmark.reports import load_jsonl, markdown_table
from benchmark.retrieval_metrics import bootstrap_ci, recall_at_k, reciprocal_rank
from benchmark.retrieval_score_distribution import (
    gold_ids,
    load_corpus,
    unit_vectors,
)

KS = (1, 3, 10, 30)
SCOPES = ("library", "paper")
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# name -> (loader, model id, how queries are encoded)
DENSE = {
    "bge-base": ("fastembed", "BAAI/bge-base-en-v1.5", "plain"),
    "bge-base+instruction": ("fastembed", "BAAI/bge-base-en-v1.5", "bge"),
    "gte-modernbert": ("st", "Alibaba-NLP/gte-modernbert-base", "prompt"),
    "granite-r2": ("st", "ibm-granite/granite-embedding-english-r2", "prompt"),
    "qwen3-0.6b": ("st", "Qwen/Qwen3-Embedding-0.6B", "prompt"),
    "embeddinggemma": ("st", "google/embeddinggemma-300m", "prompt"),
}
# name -> (loader, model id, how queries are encoded, whether Qdrant applies IDF)
SPARSE = {
    "bm25": ("fastembed", "Qdrant/bm25", "embed", True),
    "bm25+query_embed": ("fastembed", "Qdrant/bm25", "query_embed", True),
    "minicoil": ("fastembed", "Qdrant/minicoil-v1", "query_embed", True),
    "opensearch-v3-gte": ("st", "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
                          "query", False),
}
REFERENCE = "hybrid:bge-base+bm25"  # the pipeline as shipped when this ran
# bge-base and the reranker read 512 tokens, so every model gets the same window.
# Models that accept 8k-32k tokens would otherwise pad a batch to its longest
# chunk, which is what ran a 16GB machine out of memory on Qwen3.
MAX_TOKENS = 512
# fastembed's default batch of 256 is what ran out of memory loading miniCOIL
FASTEMBED_BATCH = 32


def sparse_rows(tensor) -> list[dict]:
    """A batch of torch sparse rows as one {indices, values} dict per row."""
    tensor = tensor.coalesce().cpu()
    rows, cols = tensor.indices().numpy()
    values = tensor.values().float().numpy()
    out = [{"indices": [], "values": []} for _ in range(tensor.shape[0])]
    for row, col, value in zip(rows, cols, values, strict=True):
        out[row]["indices"].append(int(col))
        out[row]["values"].append(float(value))
    return out


def arm_rankings(dense: dict[str, np.ndarray], sparse: dict[str, np.ndarray],
                 ids: list, limit: int = LIMIT) -> dict[str, list]:
    """Every arm's top-``limit`` ids for one question, over the chunks in ``ids``."""
    zeros = np.zeros(len(ids))
    out = {}
    for name, scores in dense.items():
        out[f"dense:{name}"] = [ids[i] for i, _ in retrieve("dense", scores, zeros, limit)]
    for name, scores in sparse.items():
        out[f"sparse:{name}"] = [ids[i] for i, _ in retrieve("sparse", zeros, scores, limit)]
    for d, d_scores in dense.items():
        for s, s_scores in sparse.items():
            out[f"hybrid:{d}+{s}"] = [ids[i] for i, _ in retrieve("hybrid", d_scores, s_scores, limit)]
    return out


def ranking_scores(ranked: list, gold: set) -> dict[str, float]:
    row = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in KS}
    row["mrr"] = reciprocal_rank(ranked, gold)
    return row


def delta(rows: list[dict], base: list[dict], metric: str) -> str:
    deltas = [a[metric] - b[metric] for a, b in zip(rows, base, strict=True)]
    low, high = bootstrap_ci(deltas)
    return f"{np.mean(deltas):+.3f} [{low:+.3f}, {high:+.3f}]"


def report(name: str, rows: dict, n_questions: int, costs: list[list]) -> str:
    """One table per scope, arms by recall@10, paired deltas against the shipped hybrid."""
    header = ["arm", *(f"recall@{k}" for k in KS), "mrr",
              "Δ recall@10", "Δ recall@30", "Δ mrr"]
    out = [f"# Retrieval models, offline: {name}",
           (f"{n_questions} questions, exact scoring, RRF at depth {LIMIT}. "
            f"Δ is the paired difference against `{REFERENCE}` (the shipped pipeline), "
            "with its 95% bootstrap interval.")]
    for scope in SCOPES:
        base = rows[(scope, REFERENCE)]
        arms = sorted((a for s, a in rows if s == scope),
                      key=lambda a: -np.mean([r["recall@10"] for r in rows[(scope, a)]]))
        lines = []
        for arm in arms:
            arm_rows = rows[(scope, arm)]
            cells = [f"{np.mean([r[m] for r in arm_rows]):.3f}"
                     for m in (*(f"recall@{k}" for k in KS), "mrr")]
            cells += ["", "", ""] if arm == REFERENCE else [
                delta(arm_rows, base, m) for m in ("recall@10", "recall@30", "mrr")]
            lines.append([arm, *cells])
        out += [f"## scope: {scope}", markdown_table(header, lines)]
    out += ["## Encoding cost on this machine",
            markdown_table(["kind", "model", "chunks", "seconds", "chunks/s", "ms per query"], costs)]
    return "\n\n".join(out) + "\n"


def free_accelerator() -> None:
    """Drop the last model's memory before the next loads: on a 16GB Mac, MPS
    keeps freed tensors cached, and six models in a row exhaust it."""
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def encode(kind: str, name: str, texts: list[str], queries: list[str], device: str) -> dict:
    """Documents and queries through one model, with the time each side took."""
    spec = (DENSE if kind == "dense" else SPARSE)[name]
    loader, model_id, form = spec[:3]
    if kind == "dense" and loader == "fastembed":
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=model_id)
        docs_fn = lambda xs: np.asarray(list(model.embed(xs, batch_size=FASTEMBED_BATCH)))
        query_fn = lambda xs: docs_fn([BGE_INSTRUCTION + x for x in xs]
                                      if form == "bge" else xs)
    elif kind == "dense":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
        model.max_seq_length = MAX_TOKENS
        docs_fn = lambda xs: model.encode_document(xs, batch_size=16)
        query_fn = lambda xs: model.encode_query(xs, batch_size=16)
    elif loader == "fastembed":
        from fastembed import SparseTextEmbedding
        model = SparseTextEmbedding(model_name=model_id)
        as_rows = lambda embs: [{"indices": e.indices.tolist(), "values": e.values.tolist()}
                                for e in embs]
        docs_fn = lambda xs: as_rows(model.embed(xs, batch_size=FASTEMBED_BATCH))
        query_fn = lambda xs: as_rows(getattr(model, form)(xs))
    else:
        from sentence_transformers import SparseEncoder
        model = SparseEncoder(model_id, device=device, trust_remote_code=True)
        model.max_seq_length = MAX_TOKENS
        docs_fn = lambda xs: sparse_rows(model.encode_document(xs, batch_size=16))
        query_fn = lambda xs: sparse_rows(model.encode_query(xs, batch_size=16))
    started = time.perf_counter()
    docs = docs_fn(texts)
    doc_seconds = time.perf_counter() - started
    started = time.perf_counter()
    encoded_queries = query_fn(queries)
    query_seconds = time.perf_counter() - started
    return {"docs": docs, "queries": encoded_queries,
            "doc_seconds": doc_seconds, "query_seconds": query_seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="output_qasa",
                        help="holds golden_set_aligned.jsonl and chunks/")
    parser.add_argument("--dense", nargs="+", default=list(DENSE))
    parser.add_argument("--sparse", nargs="+", default=list(SPARSE))
    args = parser.parse_args()

    from device import resolve_device

    golden_dir = Path(args.golden_dir)
    ids, texts = load_corpus(golden_dir / "chunks")
    known = set(ids)
    paper_of = np.array([paper for paper, _ in ids])
    records = [r for r in load_jsonl(golden_dir / "golden_set_aligned.jsonl")
               if any(g in known for g in gold_ids(r))]
    queries = [r["question"] for r in records]
    cache_dir = golden_dir / "retrieval_cache"
    cache_dir.mkdir(exist_ok=True)
    device = resolve_device(None)

    encoded, costs = {"dense": {}, "sparse": {}}, []
    for kind, names in (("dense", args.dense), ("sparse", args.sparse)):
        for name in names:
            path = cache_dir / f"{kind}-{name}.pkl"
            if path.exists():
                result = pickle.loads(path.read_bytes())
            else:
                try:
                    result = encode(kind, name, texts, queries, device)
                except Exception as exc:  # noqa: BLE001 - a model that will not load is reported, not fatal
                    print(f"skipped {kind}:{name} - {type(exc).__name__}: {exc}")
                    continue
                path.write_bytes(pickle.dumps(result))
                free_accelerator()
            encoded[kind][name] = result
            costs.append([kind, name, len(texts), f"{result['doc_seconds']:.0f}",
                          f"{len(texts) / result['doc_seconds']:.0f}",
                          f"{1000 * result['query_seconds'] / len(queries):.1f}"])
            print(f"{kind}:{name} ready")

    vectors = {n: (unit_vectors(e["docs"]), unit_vectors(e["queries"]))
               for n, e in encoded["dense"].items()}
    indexes = {}
    for name, e in encoded["sparse"].items():
        index = inverted_index(e["docs"])
        idf = idf_table(e["docs"]) if SPARSE[name][3] else dict.fromkeys(index, 1.0)
        indexes[name] = (index, idf, e["queries"])

    rows = defaultdict(list)
    for q, record in enumerate(records):
        gold = {g for g in gold_ids(record) if g in known}
        dense = {n: docs @ qv[q] for n, (docs, qv) in vectors.items()}
        sparse = {n: bm25(qs[q], index, idf, len(ids)) for n, (index, idf, qs) in indexes.items()}
        for scope in SCOPES:
            keep = (np.flatnonzero(paper_of == record["paper_id"]) if scope == "paper"
                    else np.arange(len(ids)))
            rankings = arm_rankings({n: s[keep] for n, s in dense.items()},
                                    {n: s[keep] for n, s in sparse.items()},
                                    [ids[i] for i in keep])
            for arm, ranked in rankings.items():
                rows[(scope, arm)].append(ranking_scores(ranked, gold))

    out_file = golden_dir / "retrieval_models.md"
    out_file.write_text(report(golden_dir.name, rows, len(records), costs))
    print(f"{len(records)} questions -> {out_file}")


if __name__ == "__main__":
    main()
