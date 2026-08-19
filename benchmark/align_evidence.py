"""Align the paragraphs QASPER annotates as evidence with the chunks the system
extracts from the PDF (methodology chapter, Aligning the annotated evidence).

QASPER derives its text from the LaTeX sources, we derive ours from the PDF, so
the two segmentations never coincide exactly. A paragraph counts as aligned
only when a *single* chunk covers it: GROBID chunks by paragraph/list-item
boundary, so evidence genuinely spread over several chunks (e.g. a QASPER
annotation that concatenates a few bullet points into one evidence string) is
left unaligned on purpose, since retrieving one of those chunks would not on
its own give access to the annotated evidence.

The matching criterion is token overlap normalised by the *longer* of the two
texts (paragraph, chunk) — equivalent to requiring both recall (does the
chunk contain most of the paragraph?) and precision (is the chunk mostly the
paragraph, not padded with unrelated content?) to individually clear the
threshold. Manual validation (see alignment_manual_check.md) found two ways
the earlier shorter-text normalisation failed silently: a chunk that is a
strict subset of a longer paragraph scored near 1.0 despite missing most of
it (recall problem, normalised by the chunk's own short length), and a
paragraph fully contained in an oversized chunk merging unrelated content
also scored near 1.0 (precision problem, normalised by the paragraph's own
short length). Normalising by the longer text catches both.

A paragraph with no chunk above the threshold is left unaligned and counted,
never paired with its closest chunk.

Usage:
  python -m benchmark.align_evidence golden_set.jsonl --chunks-dir chunks/ --out-dir out/
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# Overlap above which a chunk is accepted as covering an annotated paragraph.
# Hand-validated on a sample, see the report produced by --sample.
THRESHOLD = 0.8

# Below this length the overlap ratio is not meaningful: a handful of common
# words is enough to reach any threshold by chance.
MIN_TOKENS = 10

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return TOKEN_RE.findall(text.lower())


def overlap(a_tokens: list, b_tokens: list) -> float:
    """Shared tokens over the length of the longer text, counting repetitions.

    Equivalent to requiring both shared/len(a) and shared/len(b) to clear the
    threshold, since the smaller of the two ratios is always the one with the
    longer text as denominator.
    """
    if len(a_tokens) < MIN_TOKENS or len(b_tokens) < MIN_TOKENS:
        return 0.0
    shared = sum((Counter(a_tokens) & Counter(b_tokens)).values())
    return shared / max(len(a_tokens), len(b_tokens))


def align_paragraph(paragraph: str, chunks: list, threshold: float = THRESHOLD) -> list:
    """Indices of the chunks covering the paragraph, best overlap first."""
    paragraph_tokens = tokenize(paragraph)
    scored = [(i, overlap(paragraph_tokens, tokenize(c))) for i, c in enumerate(chunks)]
    matched = [(i, score) for i, score in scored if score >= threshold]
    return sorted(matched, key=lambda x: (-x[1], x[0]))


def align_records(records: list, chunks_by_paper: dict, threshold: float = THRESHOLD):
    """Attach the aligned chunks to every golden set record.

    Records whose evidence cannot be aligned at all are dropped, as the
    methodology requires, and counted in the returned statistics.
    """
    aligned, dropped, stats = [], [], Counter()
    for record in records:
        chunks = chunks_by_paper.get(record["paper_id"])
        if chunks is None:
            stats["questions_without_chunks"] += 1
            dropped.append(dict(record, drop_reason="no chunks for this paper"))
            continue

        matches = {}
        for paragraph in record["evidence"]:
            stats["evidence_paragraphs_total"] += 1
            if len(tokenize(paragraph)) < MIN_TOKENS:
                # One-line fragments, often the text around a formula. They are
                # counted apart, since no overlap criterion can place them
                # reliably and their failure says nothing about the procedure.
                stats["evidence_paragraphs_too_short"] += 1
                stats["evidence_paragraphs_unaligned"] += 1
                continue
            hits = align_paragraph(paragraph, chunks, threshold)
            if hits:
                stats["evidence_paragraphs_aligned"] += 1
                matches[paragraph] = [{"chunk_index": i, "overlap": round(s, 3)} for i, s in hits]
            else:
                stats["evidence_paragraphs_unaligned"] += 1

        if len(matches) < len(record["evidence"]):
            # Partially aligned evidence would understate the evidence F1 of a
            # correct answer, so the question is dropped rather than scored.
            stats["questions_dropped_partial_alignment" if matches
                  else "questions_dropped_no_alignment"] += 1
            dropped.append(dict(record, drop_reason="evidence not fully aligned"))
            continue

        stats["questions_aligned"] += 1
        aligned.append(dict(record, aligned_chunks=matches))

    stats["questions_total"] = len(records)
    total = stats["evidence_paragraphs_total"]
    stats["paragraph_alignment_rate"] = round(stats["evidence_paragraphs_aligned"] / total, 4) if total else 0.0
    stats["question_alignment_rate"] = round(stats["questions_aligned"] / len(records), 4) if records else 0.0
    stats["threshold"] = threshold
    return aligned, dropped, dict(stats)


def manual_check_report(aligned: list, chunks_by_paper: dict, size: int, seed: int) -> str:
    """Markdown report of a random sample of matches, to be checked by hand.

    The error rate observed on this sample is what the methodology asks to
    report alongside the alignment rate. Each matched chunk already covers
    the paragraph on its own (see overlap), but a paragraph can still match
    more than one chunk (e.g. near-duplicate text across papers); every
    matched chunk is shown, not just the best-scoring one.
    """
    pairs = [(r, paragraph, hits) for r in aligned
             for paragraph, hits in r["aligned_chunks"].items()]
    sample = random.Random(seed).sample(pairs, min(size, len(pairs)))

    lines = [f"# Manual validation sample (n={len(sample)}, seed={seed})", "",
             "For every pair, mark whether the matched chunk(s) really cover the",
             "annotated paragraph. The share of wrong pairs is the alignment error rate.", ""]
    for n, (record, paragraph, hits) in enumerate(sample, 1):
        best_overlap = hits[0]["overlap"]
        lines += [f"## {n}. paper {record['paper_id']}, overlap {best_overlap}",
                  "", "correct? [ ] yes [ ] no", "",
                  "**QASPER paragraph**", "", f"> {paragraph}", ""]
        for hit in hits:
            chunk = chunks_by_paper[record["paper_id"]][hit["chunk_index"]]
            label = "Our chunk" if len(hits) == 1 else f"Our chunk {hit['chunk_index']} (overlap {hit['overlap']})"
            lines += [f"**{label}**", "", f"> {chunk}", ""]
    return "\n".join(lines)


def load_chunks(chunks_dir: Path) -> dict:
    return {path.stem: json.loads(path.read_text())
            for path in sorted(chunks_dir.glob("*.json"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden_set", help="golden_set.jsonl from a golden-set builder")
    parser.add_argument("--chunks-dir", required=True,
                        help="directory of <paper_id>.json files, each a list of chunk texts")
    parser.add_argument("--out-dir", default="benchmark_out_qasper", help="output directory")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--sample", type=int, default=30,
                        help="size of the sample to validate by hand")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-sample", action="store_true",
                        help="overwrite alignment_manual_check.md even if it already exists "
                             "(it holds hand-validated judgements once reviewed, easy to lose otherwise)")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.golden_set).read_text().splitlines() if line]
    chunks_by_paper = load_chunks(Path(args.chunks_dir))
    aligned, dropped, stats = align_records(records, chunks_by_paper, args.threshold)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "golden_set_aligned.jsonl").open("w") as f:
        for record in aligned:
            f.write(json.dumps(record) + "\n")
    with (out / "alignment_dropped.jsonl").open("w") as f:
        for record in dropped:
            f.write(json.dumps(record) + "\n")
    (out / "alignment_stats.json").write_text(json.dumps(stats, indent=2))
    manual_check_path = out / "alignment_manual_check.md"
    if aligned and args.sample:
        if manual_check_path.exists() and not args.force_sample:
            print(f"skipped {manual_check_path}: already exists, pass --force-sample to regenerate "
                  "(this would discard any hand validation already recorded in it)")
        else:
            manual_check_path.write_text(
                manual_check_report(aligned, chunks_by_paper, args.sample, args.seed))

    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
