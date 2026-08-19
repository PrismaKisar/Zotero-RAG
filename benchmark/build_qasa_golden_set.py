"""Build the golden set from QASA, the retrieval-evidence counterpart to
build_qasper_golden_set.py (methodology chapter, second passage-level dataset).

Input: a QASA release file, e.g. testset_answerable_1554_v1.1.json from
https://github.com/lgresearch/QASA (data/), keyed by an arbitrary index and
holding one {paper_id, title, question_id, question, evidential_info: [...]}
per record. question_id only resets per paper, so the golden set key is
"<paper_id>#<question_id>".

Unlike QASPER, every QASA question already carries its evidence paragraphs
("context" in evidential_info): there is no answer-type filtering to apply,
only deduplication of repeated evidence across annotators.

QASA papers are not keyed by arXiv id, so PDFs are fetched by resolving each
paper's title against the arXiv search API first.

Output:
  golden_set.jsonl   one record per (question, paper) kept
  golden_stats.json  counts for the thesis

Usage:
  python -m benchmark.build_qasa_golden_set testset_answerable_1554_v1.1.json \
      --out-dir out/ --papers 40 --seed 42 --pdf-dir out/pdfs
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ARXIV_API = "https://export.arxiv.org/api/query"

TITLE_RE = re.compile(r"[a-z0-9]+")


def normalize_title(title: str) -> str:
    return " ".join(TITLE_RE.findall(title.lower()))


def build(qasa: dict, papers: int | None, seed: int):
    """Group QASA records by paper, dedup evidence, return (records, stats)."""
    stats = Counter()
    by_paper: dict[str, list] = {}
    titles: dict[str, str] = {}

    for entry in qasa.values():
        stats["questions_total"] += 1
        paper_id = entry["paper_id"]
        titles[paper_id] = entry["title"]

        evidence = [e["context"] for e in entry.get("evidential_info", []) if e.get("context")]
        evidence = list(dict.fromkeys(evidence))
        if not evidence:
            stats["excluded_no_evidence"] += 1
            continue

        stats["included"] += 1
        multi = len(evidence) > 1
        stats["included_multi_evidence" if multi else "included_single_evidence"] += 1
        by_paper.setdefault(paper_id, []).append({
            "paper_id": paper_id,
            "title": entry["title"],
            "question_id": f"{paper_id}#{entry['question_id']}",
            "question": entry["question"],
            "evidence": evidence,
            "multi_evidence": multi,
        })

    stats["papers_with_included_questions"] = len(by_paper)

    selected = sorted(by_paper)
    if papers is not None and papers < len(selected):
        selected = sorted(random.Random(seed).sample(selected, papers))
    stats["papers_sampled"] = len(selected)
    stats["seed"] = seed

    records = [r for pid in selected for r in by_paper[pid]]
    stats["questions_in_golden_set"] = len(records)
    stats["golden_multi_evidence"] = sum(r["multi_evidence"] for r in records)
    return records, dict(stats)


def resolve_arxiv_id(title: str, fetch=None) -> str | None:
    """Best-effort arXiv id for a paper title, or None if unresolved.

    A hit is only accepted when the returned entry's title matches ours
    exactly after normalization, so an unrelated top result never leaks in
    as a false PDF.
    """
    import urllib.parse
    import xml.etree.ElementTree as ET

    if fetch is None:
        import requests

        def fetch(url):
            return requests.get(url, timeout=30).content

    query = urllib.parse.urlencode({"search_query": f'ti:"{title}"', "max_results": 1})
    try:
        raw = fetch(f"{ARXIV_API}?{query}")
    except OSError:
        return None

    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    entry = root.find("a:entry", ns)
    if entry is None:
        return None

    found_title = entry.findtext("a:title", default="", namespaces=ns)
    if normalize_title(found_title) != normalize_title(title):
        return None

    entry_id = entry.findtext("a:id", default="", namespaces=ns)
    return entry_id.rstrip("/").rsplit("/", 1)[-1] if entry_id else None


def download_pdfs(records: list, pdf_dir: Path):
    """Resolve each paper to an arXiv id and download its PDF."""
    import requests

    pdf_dir.mkdir(parents=True, exist_ok=True)
    titles = {r["paper_id"]: r["title"] for r in records}
    unresolved, failed = [], []
    for paper_id, title in sorted(titles.items()):
        target = pdf_dir / f"{paper_id}.pdf"
        if target.exists():
            continue
        arxiv_id = resolve_arxiv_id(title)
        if arxiv_id is None:
            unresolved.append(paper_id)
            continue
        response = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=60)
        if response.ok and response.content[:4] == b"%PDF":
            target.write_bytes(response.content)
        else:
            failed.append(paper_id)
    return unresolved, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qasa", help="path to a QASA JSON release")
    parser.add_argument("--out-dir", default="benchmark_out_qasa", help="output directory")
    parser.add_argument("--papers", type=int, default=None,
                        help="sample size in papers, fixed before the campaign")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pdf-dir", default=None, help="also resolve and download the arXiv PDFs here")
    args = parser.parse_args()

    qasa = json.loads(Path(args.qasa).read_text())
    records, stats = build(qasa, args.papers, args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "golden_set.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    (out / "golden_stats.json").write_text(json.dumps(stats, indent=2))

    if args.pdf_dir:
        unresolved, failed = download_pdfs(records, Path(args.pdf_dir))
        if unresolved:
            print(f"arXiv id not resolved for {len(unresolved)} papers: {unresolved}")
        if failed:
            print(f"PDF download failed for {len(failed)} papers: {failed}")

    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
