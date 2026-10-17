"""Two-stage retrieval: pick the papers first, then the chunks inside them.

Retrieval over the whole library is where the answers are lost: scoped to the
question's own paper (the ablation's ``oracle_paper``), QASPER's recall@10 goes
from 0.33 to 0.85. This asks how much of that gap a first stage that ranks
papers can recover. Each strategy scores every paper, keeps the best ``depth``,
and runs the shipped hybrid search (dense + BM25, RRF) over their chunks only.

Paper scores:

- ``max``: the paper's best chunk cosine;
- ``top3``: the mean of its three best chunk cosines;
- ``centroid``: cosine with the mean of its chunk vectors;
- ``bm25``: BM25 over the paper as one document (its chunks' token weights summed,
  IDF counted over papers);
- ``hybrid``: RRF of ``max`` and ``bm25``.

Offline and exact, on the encodings the scalability study cached
(benchmark/studies/scalability.py), so nothing is re-encoded but the questions.
Reported against the flat search with paired bootstrap intervals.

Usage:
  python -m benchmark.studies.hierarchical_retrieval --golden-dir data/dev/qasper
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.metrics.reports import load_jsonl, markdown_table
from benchmark.metrics.retrieval_metrics import format_delta, recall_at_k
from benchmark.studies.offline_retrieval import (
    bm25,
    gold_ids,
    idf_table,
    inverted_index,
    retrieve,
    rrf,
    unit_vectors,
)

STRATEGIES = ("max", "top3", "centroid", "bm25", "hybrid")
DEPTHS = (1, 3, 5, 10)
K = 10


def per_paper(scores: np.ndarray, paper_idx: np.ndarray, n_papers: int, how: str) -> np.ndarray:
    """Chunk scores reduced to one per paper: the best, or the mean of the best three."""
    out = np.full(n_papers, -np.inf)
    if how == "max":
        np.maximum.at(out, paper_idx, scores)
        return out
    for p in range(n_papers):
        out[p] = np.sort(scores[paper_idx == p])[-3:].mean()
    return out


def paper_documents(sparse: list[dict], paper_idx: np.ndarray, n_papers: int) -> list[dict]:
    """Each paper's chunks as one BM25 document: token weights summed."""
    merged = [defaultdict(float) for _ in range(n_papers)]
    for doc, p in zip(sparse, paper_idx, strict=True):
        for token, weight in zip(doc["indices"], doc["values"], strict=True):
            merged[p][token] += weight
    return [{"indices": list(m), "values": list(m.values())} for m in merged]


def paper_ranking(strategy: str, dense: np.ndarray, query_vec: np.ndarray, query_sparse: dict,
                  library: dict) -> list[int]:
    """Paper indices, best first, under ``strategy``."""
    paper_idx, n = library["paper_idx"], len(library["papers"])
    if strategy in ("max", "top3"):
        scores = per_paper(dense, paper_idx, n, strategy)
    elif strategy == "centroid":
        scores = library["centroids"] @ query_vec
    else:
        lexical = bm25(query_sparse, library["paper_index"], library["paper_idf"], n)
        if strategy == "bm25":
            scores = lexical
        else:
            by_max = np.argsort(-per_paper(dense, paper_idx, n, "max"), kind="stable")
            by_bm25 = np.argsort(-lexical, kind="stable")
            return [int(p) for p, _ in rrf([list(by_max), list(by_bm25)])]
    return [int(p) for p in np.argsort(-scores, kind="stable")]


def two_stage(dense: np.ndarray, sparse: np.ndarray, paper_idx: np.ndarray,
              papers: list[int], ids: list) -> list:
    """The hybrid top chunks, searched only inside ``papers``."""
    keep = np.flatnonzero(np.isin(paper_idx, papers))
    return [ids[keep[i]] for i, _ in retrieve("hybrid", dense[keep], sparse[keep])]


def build_library(encoded: dict, papers: list[str]) -> dict:
    """Chunk ids, vectors and BM25 index of ``papers``, plus their paper-level views."""
    ids, vectors, sparse, paper_idx = [], [], [], []
    for p, paper in enumerate(papers):
        dense, rows = encoded["papers"][paper]
        ids += [(paper, i) for i in range(len(rows))]
        vectors.append(dense)
        sparse += rows
        paper_idx += [p] * len(rows)
    vectors = unit_vectors(np.vstack(vectors))
    paper_idx = np.array(paper_idx)
    docs = paper_documents(sparse, paper_idx, len(papers))
    return {"papers": papers, "ids": ids, "vectors": vectors, "paper_idx": paper_idx,
            "index": inverted_index(sparse), "idf": idf_table(sparse),
            "paper_index": inverted_index(docs), "paper_idf": idf_table(docs),
            "centroids": unit_vectors(np.vstack([vectors[paper_idx == p].mean(axis=0)
                                                 for p in range(len(papers))]))}


def score_question(record: dict, query: dict, library: dict) -> dict:
    """recall@K of the flat search and of every (strategy, depth), and each paper recall."""
    ids, paper_idx = library["ids"], library["paper_idx"]
    query_vec = unit_vectors(query["dense"])
    dense = library["vectors"] @ query_vec
    sparse = bm25(query["sparse"], library["index"], library["idf"], len(ids))
    gold = gold_ids(record)
    gold_paper = library["papers"].index(record["paper_id"])
    row = {"flat": recall_at_k([ids[i] for i, _ in retrieve("hybrid", dense, sparse)], gold, K),
           "oracle": recall_at_k(two_stage(dense, sparse, paper_idx, [gold_paper], ids), gold, K)}
    for strategy in STRATEGIES:
        ranking = paper_ranking(strategy, dense, query_vec, query["sparse"], library)
        for depth in DEPTHS:
            row[(strategy, depth)] = recall_at_k(
                two_stage(dense, sparse, paper_idx, ranking[:depth], ids), gold, K)
            row[(strategy, depth, "paper")] = float(gold_paper in ranking[:depth])
    return row


def report(name: str, rows: list[dict], n_papers: int) -> str:
    """One line per (strategy, depth): paper recall, chunk recall@K, delta against flat."""
    mean = lambda key: np.mean([r[key] for r in rows])
    lines = [["flat (shipped)", "", "", f"{mean('flat'):.3f}", ""],
             ["oracle paper", "1", "1.000", f"{mean('oracle'):.3f}",
              format_delta([r["oracle"] - r["flat"] for r in rows])]]
    for strategy in STRATEGIES:
        for depth in DEPTHS:
            key = (strategy, depth)
            lines.append([strategy, str(depth), f"{mean((*key, 'paper')):.3f}",
                          f"{mean(key):.3f}",
                          format_delta([r[key] - r["flat"] for r in rows])])
    return "\n\n".join([
        f"# Two-stage retrieval: {name}",
        (f"{len(rows)} questions over a library of {n_papers} papers. The first stage keeps "
         f"the best `depth` papers, the second runs the shipped hybrid search inside them. "
         f"Δ is the paired difference in recall@{K} against the flat search, with its 95% "
         "bootstrap interval."),
        markdown_table(["strategy", "depth", "paper recall", f"recall@{K}", f"Δ recall@{K}"],
                       lines)]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", default="data/dev/qasper")
    parser.add_argument("--library", nargs="+", default=["data/dev/qasper", "data/dev/qasa"],
                        help="golden dirs whose chunks/ make up the library, as in the ablation")
    parser.add_argument("--encodings", default="runs/dev/scalability/encodings.pkl")
    parser.add_argument("--out-dir", default="runs/dev/hierarchical_retrieval")
    args = parser.parse_args()

    from zotero_rag.retrieval.embedding_manager import EmbeddingManager  # heavy import

    encoded = pickle.loads(Path(args.encodings).read_bytes())
    papers = sorted(path.stem for d in args.library for path in Path(d, "chunks").glob("*.json"))
    library = build_library(encoded, papers)
    known = set(library["ids"])
    records = [r for r in load_jsonl(Path(args.golden_dir) / "golden_set_aligned.jsonl")
               if gold_ids(r) and gold_ids(r) <= known]
    embedder = EmbeddingManager()
    rows = [score_question(r, embedder.encode_query(r["question"]), library) for r in records]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(args.golden_dir).name}.md"
    out_file.write_text(report(Path(args.golden_dir).name, rows, len(papers)))
    print(f"{len(records)} questions -> {out_file}")


if __name__ == "__main__":
    main()
