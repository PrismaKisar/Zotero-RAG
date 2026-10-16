"""How the cost of answering a question grows with the size of the library.

The library grows by a factor of ten at a time, ``base * 10**i`` papers, and at
every size the same questions are answered end to end. Each stage is timed on
its own: only retrieval reads the whole library, while the reranker and the
reader see ``result_limit`` candidates and their neighbours whatever its size,
so the expectation to check is a constant cost for those two and a growing one
for the search. The search is also timed alone, from a query already encoded,
which is the figure the growth exponent (slope of log time against log chunks)
is fitted on.

Two dense searches are compared at every size: the exact scan the product ships
(linear in the chunks) and Qdrant's HNSW. Below Qdrant's indexing threshold
there is no graph and HNSW scans as well, so the two only part past it.

The real papers are the 685 the golden sets have parsed, ~50 chunks each. Past
them the library is synthetic: copies of the real papers under a new pdf_hash,
same text and BM25 vector, dense vector moved by a little Gaussian noise so no
two points coincide. The cost of a search does not depend on what the chunks
say, so a copy costs what a new paper would; what the answers are worth past
685 papers is not measured here, and no number from this study says it.
Questions come from the dev golden sets, and their papers are the first to
enter the library, so the small sizes hold papers the questions are about.

Requires a Qdrant server (``docker compose up -d qdrant``); ``--qdrant-url ''``
measures the embedded storage the app ships instead, which has no HNSW. Each
run starts from an empty collection.

Usage:
  python -m benchmark.studies.scalability --max-docs 100000
"""

import argparse
import csv
import json
import pickle
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmark.metrics.reports import load_jsonl, markdown_table
from zotero_rag.models import Chunk
from zotero_rag.settings import SETTINGS

CHUNK_DIRS = ("data/*/*/chunks", "runs/archive/*/chunks")
GOLDEN_SETS = ("data/dev/qasper/golden_set_aligned.jsonl",
               "data/dev/qasa/golden_set_aligned.jsonl")
MODES = ("exact", "hnsw")
COLLECTION_SUFFIX = "_scale"
# Per-dimension noise on a unit 768-d vector: norm ~0.14, cosine to the original ~0.99.
NOISE = 0.005
UPLOAD_BATCH = 256


def library_sizes(base: int, max_docs: int) -> list[int]:
    """``base * 10**i`` for every i that stays within ``max_docs``."""
    sizes, size = [], base
    while size <= max_docs:
        sizes.append(size)
        size *= 10
    return sizes


def corpus_order(papers: list[str], first: list[str], seed: int) -> list[str]:
    """``first`` (the questions' papers) in their order, then the rest shuffled."""
    wanted = list(dict.fromkeys(p for p in first if p in set(papers)))
    rest = sorted(set(papers) - set(wanted))
    random.Random(seed).shuffle(rest)
    return wanted + rest


def placement(position: int, order: list[str]) -> tuple[str, str, int]:
    """The real paper behind library position ``position``, its pdf_hash there, its copy number.

    The first ``len(order)`` positions are the real papers; copy r of a paper is
    filed under ``<paper>~r``.
    """
    replica, index = divmod(position, len(order))
    paper = order[index]
    return paper, paper if replica == 0 else f"{paper}~{replica}", replica


def replica_vectors(dense: np.ndarray, replica: int, seed: int) -> np.ndarray:
    """A copy's dense vectors: the originals for copy 0, unit vectors nearby for the rest."""
    if replica == 0:
        return dense
    noisy = dense + np.random.default_rng((seed, replica)).normal(0, NOISE, dense.shape)
    return noisy / np.linalg.norm(noisy, axis=1, keepdims=True)


def growth_exponent(chunks: list[int], seconds: list[float]) -> float:
    """Slope of log time against log size: 1 linear, ~0 constant or logarithmic."""
    return float(np.polyfit(np.log10(chunks), np.log10(seconds), 1)[0])


def summary(rows: list[dict], index_rows: list[dict]) -> str:
    """Medians per size and mode, the index build, and the fitted exponents."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["docs"], row["chunks"], row["mode"]), []).append(row)
    lines, fits = [], {}
    for (docs, chunks, mode), group in sorted(groups.items()):
        med = {k: float(np.median([r[k] for r in group]))
               for k in ("search_s", "retrieval_s", "rerank_s", "read_s", "total_s")}
        p95 = float(np.percentile([r["search_s"] for r in group], 95))
        reached = np.mean([r["reached_reader"] for r in group])
        lines.append([docs, chunks, "yes" if group[0]["synthetic"] else "no", mode,
                      f"{1000 * med['search_s']:.1f}", f"{1000 * p95:.1f}",
                      f"{med['retrieval_s']:.3f}", f"{med['rerank_s']:.3f}",
                      f"{med['read_s']:.3f}", f"{med['total_s']:.3f}", f"{reached:.0%}"])
        fits.setdefault(mode, []).append((chunks, med["search_s"]))
    out = ["# Scalability of inference",
           (f"{len({r['question_id'] for r in rows})} questions per size and mode. Times are "
            "medians; search is the Qdrant query alone, retrieval adds the query encoding "
            "and the neighbours. Rerank and read run only for questions that reach them."),
           markdown_table(["papers", "chunks", "synthetic", "mode", "search ms", "search p95 ms",
                           "retrieval s", "rerank s", "read s", "total s", "reached reader"],
                          lines),
           "## Growth exponent of the search (log-log slope against chunks)",
           markdown_table(["mode", "exponent", "sizes"],
                          [[mode, f"{growth_exponent(*zip(*pts)):.2f}", len(pts)]
                           for mode, pts in fits.items() if len(pts) > 1]),
           "## Index build",
           markdown_table(["papers", "chunks", "upload s", "until indexed s"],
                          [[r["docs"], r["chunks"], f"{r['upload_s']:.1f}",
                            f"{r['optimize_s']:.1f}"] for r in index_rows])]
    return "\n\n".join(out) + "\n"


def load_papers(root: Path) -> dict[str, list[str]]:
    """Every parsed paper once, as {paper_id: chunk texts}; the first copy found wins."""
    papers = {}
    for pattern in CHUNK_DIRS:
        for path in sorted(root.glob(f"{pattern}/*.json")):
            papers.setdefault(path.stem, json.loads(path.read_text()))
    return papers


def encode_library(rag, papers: dict[str, list[str]], cache: Path) -> dict:
    """{paper: (dense array, sparse rows)} and the encoding time, cached in ``cache``."""
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    ids = list(papers)
    texts = [t for p in ids for t in papers[p]]
    started = time.perf_counter()
    encoded = rag.embedding_manager.encode_chunks(texts)
    seconds = time.perf_counter() - started
    dense = np.asarray(encoded["dense"], dtype=np.float32)
    out, start = {"seconds": seconds, "chunks": len(texts), "papers": {}}, 0
    for p in ids:
        end = start + len(papers[p])
        out["papers"][p] = (dense[start:end], encoded["sparse"][start:end])
        start = end
    cache.write_bytes(pickle.dumps(out))
    return out


def library_points(order, papers, encoded, start: int, stop: int, seed: int):
    """The Qdrant points of library positions [start, stop), built one paper at a time."""
    from qdrant_client.http import models as qmodels

    from zotero_rag.retrieval.qdrant_manager import QdrantManager

    for position in range(start, stop):
        paper, pdf_hash, replica = placement(position, order)
        dense, sparse = encoded["papers"][paper]
        for index, (text, vector, bm25) in enumerate(
                zip(papers[paper], replica_vectors(dense, replica, seed), sparse, strict=True)):
            chunk = Chunk(text=text, page_number=1, chunk_index=index, title=pdf_hash,
                          pdf_hash=pdf_hash)
            yield qmodels.PointStruct(id=QdrantManager._generate_point_id(*chunk.key),
                                      vector={"": vector.tolist(), "text-sparse": bm25},
                                      payload=asdict(chunk))


def wait_until_indexed(manager) -> float:
    """Seconds until Qdrant has finished building the collection's index."""
    from qdrant_client.http import models as qmodels

    started = time.perf_counter()
    while manager.client.get_collection(manager.chunk_collection).status != \
            qmodels.CollectionStatus.GREEN:
        time.sleep(1)
    return time.perf_counter() - started


def time_question(rag, record: dict, query: dict, mode: str) -> dict:
    """One question end to end, plus its search alone, under ``mode``."""
    exact = mode == "exact"
    started = time.perf_counter()
    rag.qdrant_manager.search(query, result_limit=SETTINGS["result_limit"], exact=exact)
    search = time.perf_counter() - started
    started = time.perf_counter()
    rag.answer_question(record["question"], overrides={"retrieval_exact": exact})
    total = time.perf_counter() - started
    stages = rag.last_trace.stage_times
    return {"question_id": record["question_id"], "mode": mode, "search_s": search,
            "retrieval_s": stages.get("retrieval", 0.0), "rerank_s": stages.get("rerank", 0.0),
            "read_s": stages.get("read", 0.0), "total_s": total,
            "reached_reader": "read" in stages}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=int, default=1, help="papers at the smallest size")
    parser.add_argument("--max-docs", type=int, default=100_000,
                        help="largest library, in papers; ~50 chunks and ~1.5KB of dense "
                             "vector each, so 1e5 papers are ~7.5GB of Qdrant memory")
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qdrant-url", default="http://localhost:6333",
                        help="'' for the app's embedded storage (exact search only)")
    parser.add_argument("--out-dir", default="runs/dev/scalability")
    args = parser.parse_args()

    from zotero_rag.pipeline import ZoteroRAG  # heavy import (torch/transformers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    papers = load_papers(Path("."))
    records = [r for path in GOLDEN_SETS for r in load_jsonl(Path(path))
               if r["paper_id"] in papers]
    records = random.Random(args.seed).sample(records, min(args.questions, len(records)))
    order = corpus_order(list(papers), [r["paper_id"] for r in records], args.seed)

    qdrant_url = args.qdrant_url or None
    rag = ZoteroRAG(qdrant_url=qdrant_url, output_base_dir=str(out_dir / "work"),
                    qdrant_collection_suffix=COLLECTION_SUFFIX)
    encoded = encode_library(rag, papers, out_dir / "encodings.pkl")
    print(f"{encoded['chunks']} chunks encoded in {encoded['seconds']:.0f}s")
    queries = [rag.embedding_manager.encode_query(r["question"]) for r in records]
    modes = MODES if qdrant_url else ("exact",)

    manager = rag.qdrant_manager
    manager.client.delete_collection(manager.chunk_collection)
    manager._collections_ready = False

    rows, index_rows, docs, chunks = [], [], 0, 0
    for size in library_sizes(args.base, args.max_docs):
        started = time.perf_counter()
        manager.client.upload_points(
            manager.chunk_collection,
            library_points(order, papers, encoded, docs, size, args.seed),
            batch_size=UPLOAD_BATCH, wait=True)
        upload = time.perf_counter() - started
        chunks += sum(len(papers[placement(p, order)[0]]) for p in range(docs, size))
        docs = size
        index_rows.append({"docs": docs, "chunks": chunks, "upload_s": upload,
                           "optimize_s": wait_until_indexed(manager)})
        for mode in modes:
            time_question(rag, records[0], queries[0], mode)  # warm-up, not recorded
            for record, query in zip(records, queries, strict=True):
                rows.append({"docs": docs, "chunks": chunks, "synthetic": docs > len(order),
                             **time_question(rag, record, query, mode)})
        print(f"{docs} papers, {chunks} chunks done")
        write_csv(out_dir / "scalability.csv", rows)
        write_csv(out_dir / "index_build.csv", index_rows)
        (out_dir / "scalability.md").write_text(summary(rows, index_rows))
    print(f"-> {out_dir / 'scalability.md'}")


if __name__ == "__main__":
    main()
