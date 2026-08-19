"""Index the benchmark PDFs into Qdrant, bypassing Zotero.

ZoteroRAG.upsert_pdfs() takes a generic List[CachedPDF]; it does not require a
Zotero library. This ingests the PDFs already downloaded for the golden set
(benchmark_out_qasper/pdfs/) as a folder source instead, and records which pdf_hash
Qdrant ended up assigning to each paper_id, needed to score retrieval rankings
against benchmark_out_qasper/golden_set_aligned.jsonl (see benchmark/README.md).

Requires GROBID and Qdrant running (docker-compose up).

Each corpus gets its own Qdrant collection via --qdrant-collection-suffix: retrieval is
corpus-wide (the system searches a whole library, not one paper), so two datasets
sharing a collection would contaminate each other's ranking.

Chunk contextualization is off unless --contextualize is passed. It is an
unmeasured Phase 2 intervention, and its failure path is a silent fallback to raw
chunks, so leaving it on by default would make the baseline index unreproducible.

Usage:
  python -m benchmark.index_pdfs --pdf-dir benchmark_out_qasper/pdfs \
      --out-file benchmark_out_qasper/pdf_hash_map.json --work-dir benchmark_out_qasper/grobid \
      --qdrant-collection-suffix _qasper
"""

import argparse
import json
import sys
from pathlib import Path

# appended (not inserted): prepending lets zotero_rag.py shadow the zotero_rag package
sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from models import PathSource, PDFIngestItem


def build_ingest_items(pdf_dir: Path) -> list[PDFIngestItem]:
    """One PDFIngestItem per PDF in ``pdf_dir``, titled by its paper_id (file stem)."""
    return [PDFIngestItem(title=pdf.stem, source=PathSource(str(pdf)))
            for pdf in sorted(pdf_dir.glob("*.pdf"))]


def index_pdfs(pdf_dir: Path, work_dir: Path, grobid_url: str, qdrant_url: str,
               qdrant_collection_suffix: str = "", contextualize: bool = False) -> dict[str, str]:
    """Ingest and index every PDF in ``pdf_dir``, return {paper_id: pdf_hash}."""
    from zotero_rag.zotero_rag import (
        ZoteroRAG,  # heavy import (torch/transformers), kept out of the pure path
    )

    rag = ZoteroRAG(grobid_url=grobid_url, qdrant_url=qdrant_url, output_base_dir=str(work_dir),
                    qdrant_collection_suffix=qdrant_collection_suffix,
                    use_chunk_contextualization=contextualize)
    items = build_ingest_items(pdf_dir)
    ingest_result = rag._ingest_pdfs(items)
    for failure in ingest_result.failed_pdfs:
        print(f"  ingest failed: {failure['title']}: {failure['error']}")

    rag.upsert_pdfs(ingest_result.ingested_pdfs)
    return {cached.title: cached.pdf_hash for cached in ingest_result.ingested_pdfs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--out-file", default="benchmark_out_qasper/pdf_hash_map.json")
    parser.add_argument("--work-dir", default="benchmark_out_qasper/grobid",
                        help="output_base_dir for ZoteroRAG (pdf cache, index)")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-collection-suffix", default="",
                        help="isolate this corpus in its own Qdrant collection, e.g. _qasper")
    parser.add_argument("--contextualize", action="store_true",
                        help="prefix each chunk with an LLM-generated context before embedding "
                             "(needs Ollama); off by default so the baseline index is reproducible")
    args = parser.parse_args()

    hash_map = index_pdfs(Path(args.pdf_dir), Path(args.work_dir), args.grobid_url, args.qdrant_url,
                          qdrant_collection_suffix=args.qdrant_collection_suffix,
                          contextualize=args.contextualize)
    Path(args.out_file).write_text(json.dumps(hash_map, indent=2))
    print(f"indexed {len(hash_map)} papers -> {args.out_file}")


if __name__ == "__main__":
    main()
