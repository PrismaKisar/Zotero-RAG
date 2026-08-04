"""Index the benchmark PDFs into Qdrant, bypassing Zotero.

ZoteroRAG.upsert_pdfs() takes a generic List[CachedPDF]; it does not require a
Zotero library. This ingests the PDFs already downloaded for the golden set
(benchmark_out/pdfs/) as a folder source instead, and records which pdf_hash
Qdrant ended up assigning to each paper_id, needed to score retrieval rankings
against benchmark_out/golden_set_aligned.jsonl (see benchmark/README.md).

Requires GROBID and Qdrant running (docker-compose up).

Usage:
  python -m benchmark.index_benchmark_pdfs --pdf-dir benchmark_out/pdfs \
      --out benchmark_out/pdf_hash_map.json --work-dir benchmark_out/grobid
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from models import PathSource, PDFIngestItem  # noqa: E402


def build_ingest_items(pdf_dir: Path) -> list[PDFIngestItem]:
    """One PDFIngestItem per PDF in ``pdf_dir``, titled by its paper_id (file stem)."""
    return [PDFIngestItem(title=pdf.stem, source=PathSource(str(pdf)))
            for pdf in sorted(pdf_dir.glob("*.pdf"))]


def index_pdfs(pdf_dir: Path, work_dir: Path, grobid_url: str, qdrant_url: str) -> dict[str, str]:
    """Ingest and index every PDF in ``pdf_dir``, return {paper_id: pdf_hash}."""
    from zotero_rag import ZoteroRAG  # heavy import (torch/transformers), kept out of the pure path

    rag = ZoteroRAG(grobid_url=grobid_url, qdrant_url=qdrant_url, output_base_dir=str(work_dir))
    items = build_ingest_items(pdf_dir)
    ingest_result = rag._ingest_pdfs(items)
    for failure in ingest_result.failed_uploads:
        print(f"  ingest failed: {failure['title']}: {failure['error']}")

    rag.upsert_pdfs(ingest_result.ingested_pdfs)
    return {cached.title: cached.pdf_hash for cached in ingest_result.ingested_pdfs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--out", default="benchmark_out/pdf_hash_map.json")
    parser.add_argument("--work-dir", default="benchmark_out/grobid",
                        help="output_base_dir for ZoteroRAG (pdf cache, index)")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    hash_map = index_pdfs(Path(args.pdf_dir), Path(args.work_dir), args.grobid_url, args.qdrant_url)
    Path(args.out).write_text(json.dumps(hash_map, indent=2))
    print(f"indexed {len(hash_map)} papers -> {args.out}")


if __name__ == "__main__":
    main()
