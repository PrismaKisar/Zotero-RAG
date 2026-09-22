"""Where an answerable question loses its evidence: retrieval, the gate, the margin, or the reader.

Written against the absolute cuts that shipped before, it found them throwing
away evidence they had: rerank 0.90 dropped gold the reranker itself had ranked
first, on 23% of the answerable questions of both sets, and that was the largest
single cause of empty answers (61% of them on QASPER, 48% on QASA). That is what
moved the pipeline to the gate and margin of benchmark/answer_gate.py, and this
now reads that rule, so it keeps answering the same question about the policy
actually shipped.

Per question, the first stage its gold chunks do not survive:

- ``not retrieved``: no gold chunk is in the hybrid top-LIMIT at all, so nothing
  downstream could have kept it. Only retrieval can fix this one.
- ``question not answered``: a gold chunk was a candidate, but no candidate at
  all reached ``rerank_gate``, so the question is answered with silence.
- ``cut by the margin``: the question is answered, and no gold chunk scored
  within ``rerank_margin`` of the best candidate.
- ``cut by the reader``: a gold chunk survived the margin, and none of those
  cleared ``qa_score_threshold``.
- ``kept``: the pipeline highlights gold evidence for the question.

Both tables read the same scores the threshold searches read
(benchmark/rerank_score_distribution.py and benchmark/qa_score_distribution.py),
so this runs offline, without loading a model. The second table restricts to the
questions that came back empty, which is the share the campaign reports: a
question can be counted as cut while still highlighting a non-gold chunk.

Usage:
  python -m benchmark.failure_attribution --golden-dirs output_qasper output_qasa
"""

import argparse
from collections import Counter
from pathlib import Path

from benchmark.ablation import BASELINE
from benchmark.answer_gate import kept
from benchmark.retrieval_score_distribution import _table
from benchmark.threshold_search import load_questions

STAGES = ("not retrieved", "question not answered", "cut by the margin",
          "cut by the reader", "kept")
GATE = BASELINE["rerank_gate"]
MARGIN = BASELINE["rerank_margin"]
QA = BASELINE["qa_score_threshold"]


def stage(question: dict, gate: float = GATE, margin: float = MARGIN, qa: float = QA) -> str:
    """The first cut the question's gold evidence does not survive."""
    candidates = question["candidates"]
    gold = [(r, s) for c, r, s in candidates if c in question["gold"]]
    if not gold:
        return "not retrieved"
    best = max(r for _, r, _ in candidates)
    if best < gate:
        return "question not answered"
    if not any(r >= best - margin for r, _ in gold):
        return "cut by the margin"
    if not any(r >= best - margin and s >= qa for r, s in gold):
        return "cut by the reader"
    return "kept"


def attribute(questions: list[dict], gate: float = GATE, margin: float = MARGIN,
              qa: float = QA) -> list[dict]:
    """Per question, its stage and whether the pipeline highlighted anything at all."""
    return [{"stage": stage(q, gate, margin, qa),
             "empty": not kept(q, gate, margin, qa, relative=True)}
            for q in questions]


def shares(rows: list[dict]) -> list[list]:
    """One row per stage: how many questions end there, and their share."""
    counts = Counter(row["stage"] for row in rows)
    return [[name, counts[name], f"{counts[name] / len(rows):.1%}" if rows else "-"]
            for name in STAGES]


def report(sets: dict[str, list[dict]], gate: float = GATE, margin: float = MARGIN,
           qa: float = QA) -> str:
    out = ["# Where answerable questions lose their evidence",
           (f"Gate {gate}, margin {margin}, qa {qa}. Per question, the first cut its gold "
            "chunks do not survive. The second table keeps only the questions the pipeline "
            "left with nothing highlighted at all; a question counted as cut can still "
            "highlight a chunk that is not gold.")]
    for name, questions in sets.items():
        rows = attribute(questions, gate, margin, qa)
        empty = [row for row in rows if row["empty"]]
        out += [f"## {name}: all {len(rows)} answerable questions",
                _table(["stage", "questions", "share"], shares(rows)),
                f"## {name}: the {len(empty)} left empty ({len(empty) / len(rows):.1%})",
                _table(["stage", "questions", "share"], shares(empty))]
    return "\n\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dirs", nargs="+", default=["output_qasper", "output_qasa"],
                        help="each holds golden_set_aligned.jsonl, chunks/ and the score files")
    parser.add_argument("--rerank-gate", type=float, default=GATE)
    parser.add_argument("--rerank-margin", type=float, default=MARGIN)
    parser.add_argument("--qa-threshold", type=float, default=QA)
    args = parser.parse_args()

    dirs = [Path(d) for d in args.golden_dirs]
    sets = {d.name: load_questions(d) for d in dirs}
    out_file = dirs[-1] / "failure_attribution.md"
    out_file.write_text(report(sets, args.rerank_gate, args.rerank_margin, args.qa_threshold))
    print(f"-> {out_file}")


if __name__ == "__main__":
    main()
