"""Index the benchmark PDFs into Qdrant, bypassing Zotero.

ZoteroRAG.upsert_pdfs() takes a generic List[CachedPDF]; it does not require a
Zotero library. This ingests the PDFs already downloaded for a golden set
(data/<split>/<corpus>/pdfs/) as a folder source instead, and records which
pdf_hash Qdrant ended up assigning to each paper_id, needed to score retrieval
rankings against that set's golden_set_aligned.jsonl (see benchmark/README.md).

Requires GROBID and Qdrant running (docker-compose up).

Each split gets its own Qdrant collection via --qdrant-collection-suffix, and the
two corpora of a split share it: retrieval is library-wide, and a library holds
papers from both, so dev and held-out are searched the same way. A split must
never share a collection with another, or held-out papers would sit among the
distractors of the development questions.

Usage:
  python -m benchmark.datasets.index_pdfs --pdf-dir data/dev/qasper/pdfs \
      --out-file data/dev/qasper/pdf_hash_map.json --work-dir data/dev/grobid \
      --qdrant-collection-suffix _dev
"""

import argparse
import json
import time
from pathlib import Path


def index_pdfs(pdf_dir: Path, work_dir: Path, grobid_url: str, qdrant_url: str,
               qdrant_collection_suffix: str = "") -> dict[str, str]:
    """Ingest and index every PDF in ``pdf_dir``, return {paper_id: pdf_hash}."""
    from zotero_rag.pipeline import (
        ZoteroRAG,  # heavy import (torch/transformers), kept out of the pure path
    )

    rag = ZoteroRAG(grobid_url=grobid_url, qdrant_url=qdrant_url, output_base_dir=str(work_dir),
                    qdrant_collection_suffix=qdrant_collection_suffix)
    ingest_result = rag.ingest_pdfs_from_folder(str(pdf_dir))
    for failure in ingest_result.failed_pdfs:
        print(f"  ingest failed: {failure['title']}: {failure['error']}")

    # One PDF per upsert: a failure aborts its upsert, so batching every PDF
    # would lose all of them to one error. Indexed PDFs are skipped, so a rerun
    # resumes.
    chunks, started = 0, time.perf_counter()
    for cached in ingest_result.ingested_pdfs:
        chunks += rag.upsert_pdfs([cached]).indexed_chunks
    elapsed = time.perf_counter() - started
    if chunks:
        print(f"indexed {chunks} chunks in {elapsed:.0f}s ({elapsed / chunks:.2f}s per chunk)")
    return {cached.title: cached.pdf_hash for cached in ingest_result.ingested_pdfs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--out-file", default="data/dev/qasper/pdf_hash_map.json")
    parser.add_argument("--work-dir", default="data/dev/grobid",
                        help="output_base_dir for ZoteroRAG (pdf cache, index)")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-collection-suffix", default="",
                        help="isolate this corpus in its own Qdrant collection, e.g. _dev")
    args = parser.parse_args()

    hash_map = index_pdfs(Path(args.pdf_dir), Path(args.work_dir), args.grobid_url, args.qdrant_url,
                          qdrant_collection_suffix=args.qdrant_collection_suffix)
    Path(args.out_file).write_text(json.dumps(hash_map, indent=2))
    print(f"indexed {len(hash_map)} papers -> {args.out_file}")


if __name__ == "__main__":
    main()
