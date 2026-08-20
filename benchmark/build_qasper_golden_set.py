"""Build the golden set from QASPER, following the inclusion/exclusion criteria
of the methodology chapter (Table: Composition of the evaluation set).

Input: the official QASPER JSON release, e.g. qasper-dev-v0.3.json from
https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz

Output:
  golden_set.jsonl  one record per (question, paper) kept
  golden_stats.json counts for every exclusion reason (they go in the thesis)
  golden_gold.json  the same questions in QASPER format, so that the official
                    evaluator can be run on the golden set without modification

Usage:
  python -m benchmark.build_qasper_golden_set qasper-dev-v0.3.json --out-dir out/ \
      --papers 25 --seed 42 --pdf-dir out/pdfs
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# QASPER marks evidence coming from a table or a figure with this prefix.
FLOAT_PREFIX = "FLOAT SELECTED"


def annotation_type(answer: dict) -> str:
    """Classify a single QASPER annotation into one of the four answer types."""
    if answer.get("unanswerable"):
        return "unanswerable"
    if answer.get("extractive_spans"):
        return "extractive"
    if answer.get("yes_no") is not None:
        return "yes_no"
    if answer.get("free_form_answer"):
        return "abstractive"
    return "unanswerable"


def question_type(question: dict) -> str:
    """Answer type of a question: the most frequent one across its annotations.

    Ties are broken in favour of the rarer types, so a question is only called
    extractive when extractive annotations are strictly the majority.
    """
    counts = Counter(annotation_type(a["answer"]) for a in question["answers"])
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return top[1][0]
    return top[0][0]


def extractive_annotations(question: dict) -> list:
    return [a["answer"] for a in question["answers"]
            if annotation_type(a["answer"]) == "extractive"]


def build(qasper: dict, papers: int | None, seed: int):
    """Apply the criteria and return (records, stats)."""
    stats = Counter()
    by_paper: dict[str, list] = {}

    for paper_id, paper in qasper.items():
        for qa in paper["qas"]:
            stats["questions_total"] += 1
            qtype = question_type(qa)
            if qtype != "extractive":
                stats[f"excluded_{qtype}"] += 1
                continue

            anns = extractive_annotations(qa)
            evidence = [p for a in anns for p in a.get("evidence", [])]
            if any(p.startswith(FLOAT_PREFIX) for p in evidence):
                stats["excluded_table_or_figure"] += 1
                continue

            # De-duplicate evidence paragraphs across annotators, keeping order.
            evidence = list(dict.fromkeys(evidence))
            if not evidence:
                stats["excluded_no_evidence"] += 1
                continue

            stats["included"] += 1
            multi = len(evidence) > 1
            stats["included_multi_evidence" if multi else "included_single_evidence"] += 1
            by_paper.setdefault(paper_id, []).append({
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "question_id": qa["question_id"],
                "question": qa["question"],
                "gold_spans": list(dict.fromkeys(s for a in anns for s in a["extractive_spans"])),
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


def gold_subset(qasper: dict, records: list) -> dict:
    """QASPER-shaped dict restricted to the golden set questions.

    The official evaluator scores every question present in the gold file, so
    feeding it the full release would count the excluded questions as missing
    predictions.
    """
    kept = {r["question_id"] for r in records}
    subset = {}
    for paper_id, paper in qasper.items():
        qas = [qa for qa in paper["qas"] if qa["question_id"] in kept]
        if qas:
            subset[paper_id] = dict(paper, qas=qas)
    return subset


def download_pdfs(paper_ids, pdf_dir: Path):
    """QASPER paper ids are arXiv ids, so the PDF is one URL away."""
    import requests

    pdf_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for pid in paper_ids:
        target = pdf_dir / f"{pid}.pdf"
        if target.exists():
            continue
        response = requests.get(f"https://arxiv.org/pdf/{pid}", timeout=60)
        if response.ok and response.content[:4] == b"%PDF":
            target.write_bytes(response.content)
        else:
            failed.append(pid)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qasper", help="path to the QASPER JSON release")
    parser.add_argument("--out-dir", default="output_qasper", help="output directory")
    parser.add_argument("--papers", type=int, default=None,
                        help="sample size in papers, fixed before the campaign")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pdf-dir", default=None, help="also download the arXiv PDFs here")
    args = parser.parse_args()

    qasper = json.loads(Path(args.qasper).read_text())
    records, stats = build(qasper, args.papers, args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "golden_set.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    (out / "golden_stats.json").write_text(json.dumps(stats, indent=2))
    (out / "golden_gold.json").write_text(json.dumps(gold_subset(qasper, records)))

    if args.pdf_dir:
        failed = download_pdfs(sorted({r["paper_id"] for r in records}), Path(args.pdf_dir))
        if failed:
            print(f"PDF download failed for {len(failed)} papers: {failed}")

    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
