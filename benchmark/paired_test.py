"""Paired bootstrap of every ablation config against the baseline.

ablation_results.csv reports each config's *marginal* CI, which is the wrong
test for this design: every config answers the same questions, so two marginal
intervals can overlap while the configs differ on nearly every question. The
paired difference removes the between-question variance - and questions here
differ enormously in difficulty, with many scoring 0.0 under every config - so
it is both the correct test and a far more powerful one.

Reads the per-question JSONL that ablation.py already writes, so it never
re-runs the pipeline and can be applied to any finished sweep.

Usage:
  python -m benchmark.paired_test --per-question output_qasper/per_question
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from benchmark.retrieval_metrics import bootstrap_ci

# recall@1 is here because the protocol names it the primary retrieval metric,
# and because it is where a reranker earns its keep: at k=10 a config that drags
# gold evidence from rank 9 to rank 2 is indistinguishable from one that does
# nothing. The deeper k's are left out - they move together with these two, and
# every extra metric widens the uncorrected multiple-comparison problem.
DEFAULT_METRICS = ("answer_f1", "answer_em", "recall@1", "recall@10",
                   "evidence_precision", "evidence_recall", "evidence_f1",
                   "highlighted_chars", "highlight_precision")


def load_per_question(path: Path) -> dict[str, dict]:
    """Map question_id -> metric row, for one config's JSONL dump."""
    rows = [json.loads(line) for line in path.open()]
    return {row["record"]["question_id"]: row for row in rows}


def paired_deltas(baseline: dict[str, dict], other: dict[str, dict],
                  metric: str) -> list[float]:
    """Per-question ``other - baseline`` over the questions both scored.

    Configs can score different subsets (a run that produced no answer for a
    question writes no row), so the intersection is taken rather than assumed.
    """
    shared = sorted(set(baseline) & set(other))
    return [other[q][metric] - baseline[q][metric]
            for q in shared
            if metric in baseline[q] and metric in other[q]]


def compare(baseline: dict[str, dict], other: dict[str, dict],
            metric: str) -> dict | None:
    """Mean paired delta and its bootstrap CI, or None if the metric is absent.

    ``significant`` is the usual percentile-bootstrap read: an interval that
    excludes zero. With ~15 configs compared at once this is uncorrected and so
    is a screening criterion, not a confirmatory one.
    """
    deltas = paired_deltas(baseline, other, metric)
    if not deltas:
        return None
    low, high = bootstrap_ci(deltas, resamples=10000)
    return {"metric": metric, "n": len(deltas), "delta": sum(deltas) / len(deltas),
            "ci_low": low, "ci_high": high, "significant": low > 0 or high < 0}


def compare_all(per_question_dir: Path,
                metrics: tuple[str, ...] = DEFAULT_METRICS) -> list[dict]:
    """Every config in the directory against baseline.jsonl, one row per metric."""
    baseline_path = per_question_dir / "baseline.jsonl"
    if not baseline_path.exists():
        raise SystemExit(f"no baseline.jsonl in {per_question_dir} - "
                         "point --per-question at a finished ablation run")
    baseline = load_per_question(baseline_path)

    rows = []
    for path in sorted(per_question_dir.rglob("*.jsonl")):
        if path == baseline_path:
            continue
        other = load_per_question(path)
        for metric in metrics:
            result = compare(baseline, other, metric)
            if result is not None:
                rows.append({"config": path.stem, **result})
    return rows


def to_markdown(rows: list[dict]) -> str:
    """Rows as a Markdown table, significant effects first."""
    out = ["| config | metric | n | delta | CI 95% | significant |",
           "|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: (not r["significant"], -abs(r["delta"]))):
        out.append(f"| {row['config']} | {row['metric']} | {row['n']} | "
                   f"{row['delta']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                   f"{'yes' if row['significant'] else ''} |")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-question", default="output_qasper/per_question",
                        help="directory of per-question JSONL written by ablation.py")
    parser.add_argument("--out-file", default="output_qasper/paired_test.md")
    args = parser.parse_args()

    rows = compare_all(Path(args.per_question))
    text = to_markdown(rows)
    Path(args.out_file).write_text(text)
    print(text)
    print(f"wrote: {args.out_file}")


if __name__ == "__main__":
    main()
