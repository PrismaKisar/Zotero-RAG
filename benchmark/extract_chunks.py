"""Extract the chunks the system produces from each benchmark PDF, so that the
alignment step can run without holding the whole pipeline in memory.

Writes one <paper_id>.json per paper, containing the list of chunk texts in the
order GROBID returns them. Requires the GROBID service to be running.

Usage:
  python -m benchmark.extract_chunks --pdf-dir benchmark_out/pdfs --out chunks/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from pdf_processor import PDFProcessor  # noqa: E402


def extract(pdf_dir: Path, out_dir: Path, work_dir: Path, grobid_url: str, timeout: int):
    processor = PDFProcessor(grobid_url=grobid_url, grobid_timeout=timeout,
                             output_base_dir=str(work_dir))
    cache_dir = Path(processor.pdf_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        target = out_dir / f"{pdf.stem}.json"
        if target.exists():
            continue
        # PDFProcessor reads from its own cache, keyed by file name.
        shutil.copyfile(pdf, cache_dir / f"{pdf.stem}.pdf")
        try:
            paragraphs, _ = processor.extract_text_chunks(pdf.stem)
        except (ValueError, OSError) as error:
            failed.append((pdf.stem, str(error)))
            continue
        target.write_text(json.dumps([p.text for p in paragraphs]))
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--out", default="chunks")
    parser.add_argument("--work-dir", default="benchmark_out/grobid",
                        help="where the PDF and TEI caches are kept")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    failed = extract(Path(args.pdf_dir), Path(args.out), Path(args.work_dir),
                     args.grobid_url, args.timeout)
    print(f"failed: {len(failed)}")
    for paper_id, error in failed:
        print(f"  {paper_id}: {error}")


if __name__ == "__main__":
    main()
