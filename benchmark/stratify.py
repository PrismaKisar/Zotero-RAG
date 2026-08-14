"""Break benchmark scores down by question stratum, with bootstrap CIs.

The two golden sets differ systematically (see benchmark/compare_datasets.py:
QASA questions are longer, more often causal, over denser papers), so a single
pooled mean cannot say *why* one dataset scores lower than the other. Every
metric is therefore reported per stratum:

- ``question_form``  what/why/how/yes-no, reusing compare_datasets.q_type
- ``evidence_count`` single- vs multi-evidence questions
- ``evidence_spread`` whether the gold chunks are adjacent or scattered

A stratum with fewer than MIN_STRATUM_N questions is reported but flagged: on
a 120-question golden set the tail strata are too small to compare.
"""

from collections.abc import Callable, Sequence

from benchmark.compare_datasets import q_type
from benchmark.retrieval_metrics import summarize

MIN_STRATUM_N = 20


def evidence_count_stratum(record: dict) -> str:
    return "multi-evidence" if len(record.get("evidence", [])) > 1 else "single-evidence"


def evidence_spread_stratum(record: dict) -> str:
    """Adjacent gold chunks vs scattered ones - a proxy for retrieval difficulty."""
    idx = sorted({hit["chunk_index"]
                  for hits in record.get("aligned_chunks", {}).values()
                  for hit in hits})
    if len(idx) < 2:
        return "single-chunk"
    return "adjacent" if idx[-1] - idx[0] <= len(idx) else "scattered"


STRATA: dict[str, Callable[[dict], str]] = {
    "question_form": lambda r: q_type(r["question"]),
    "evidence_count": evidence_count_stratum,
    "evidence_spread": evidence_spread_stratum,
}


def stratify(rows: Sequence[dict], strata: dict | None = None) -> dict:
    """Summarize per-question ``rows`` overall and within each stratum.

    Each row must carry a ``record`` key (the golden-set record it came from)
    alongside its numeric metrics.

    Returns {"overall": {...}, "<stratum_name>": {"<value>": {...}}}.
    """
    if not rows:
        raise ValueError("rows cannot be empty")
    strata = STRATA if strata is None else strata

    def metrics_of(row):
        return {k: v for k, v in row.items() if k != "record"}

    out = {"overall": summarize([metrics_of(r) for r in rows], with_ci=True)}
    for name, key_fn in strata.items():
        buckets: dict[str, list] = {}
        for row in rows:
            buckets.setdefault(key_fn(row["record"]), []).append(metrics_of(row))
        out[name] = {}
        for value, bucket in sorted(buckets.items()):
            summary = summarize(bucket, with_ci=True)
            summary["underpowered"] = len(bucket) < MIN_STRATUM_N
            out[name][value] = summary
    return out


def to_markdown(by_dataset: dict, metrics: Sequence[str]) -> str:
    """Render ``{dataset_name: stratify(...)}`` as thesis-ready tables.

    Cells are ``mean [ci_low, ci_high]``; a stratum too small to interpret is
    marked with a dagger rather than silently dropped.
    """
    names = list(by_dataset)
    lines = ["# Benchmark scores by stratum", "",
             ("Cells are `mean [95% bootstrap CI]`. "
              f"† = fewer than {MIN_STRATUM_N} questions, not interpretable."), ""]

    def cell(summary, metric):
        if summary is None or metric not in summary:
            return "n/a"
        mark = " †" if summary.get("underpowered") else ""
        return (f"{summary[metric]:.3f} "
                f"[{summary[f'{metric}_ci_low']:.3f}, {summary[f'{metric}_ci_high']:.3f}]{mark}")

    for metric in metrics:
        lines += [f"## {metric}", "",
                  "| Stratum | " + " | ".join(f"{n} (n)" for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        row = []
        for n in names:
            s = by_dataset[n]["overall"]
            row.append(f"{cell(s, metric)} ({s['n_questions']})")
        lines.append("| **overall** | " + " | ".join(row) + " |")

        for stratum in STRATA:
            values = sorted({v for n in names for v in by_dataset[n].get(stratum, {})})
            for value in values:
                row = []
                for n in names:
                    s = by_dataset[n].get(stratum, {}).get(value)
                    row.append(f"{cell(s, metric)} ({s['n_questions']})" if s else "n/a")
                lines.append(f"| {stratum}: {value} | " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)
