"""Deciding whether to answer once per question, then what to highlight.

benchmark/failure_attribution.py found the shipped cuts throwing away evidence
they had: at rerank 0.90 the reranker drops gold it had ranked first, for 23% of
the answerable questions on both sets, and the reader's 0.05 costs another 16%
on QASA. Lowering either one buys the evidence back and loses the silence on
questions the library cannot answer, because one number is doing two jobs: it
decides whether the library answers the question at all, and which chunks are
worth highlighting.

This splits them. Per question, ``gate`` is read off the best candidate only: no
candidate scoring at least ``gate`` means the question goes unanswered, however
many mediocre chunks it has. A question that clears the gate then keeps its
candidates by one of two rules, and ``qa_keep`` vetoes on the reader's span
score either way:

- absolute (``relative=False``): every candidate at or above ``keep``, the
  shipped rule. Measured first, it never beat ``keep == gate``: below 0.90 gte's
  ordering carries no signal, so a lower cut floods the highlights with noise.
- relative (``relative=True``): every candidate within ``keep`` of *this
  question's* best score. gte's probabilities crowd 0.80-0.93 and the crowd sits
  at a different height per question, so an absolute cut lands mid-crowd on some
  questions and above it on others; a margin from the question's own best asks
  instead which candidates the reranker rates alongside its favourite.

The grid is searched the way benchmark/threshold_search.py searches its pair,
against the same two sides and the same objective: best mean evidence F1 among
the triples that leave at least SILENCE_TARGET of the off-topic questions empty
on every set. The report scores both families against the shipped pair, which is
the absolute rule with ``gate == keep``, and gives the paired difference per
question.

Usage:
  python -m benchmark.answer_gate --golden-dirs output_qasper output_qasa
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.reports import load_jsonl, markdown_table
from benchmark.retrieval_metrics import bootstrap_ci, evidence_prf
from benchmark.threshold_search import SILENCE_TARGET, group_negatives, load_questions
from zotero_rag.selection import rerank_floor

# The gate reads the best candidate, so it lives on gte's upper range; keep is
# free to sit far below it, which is the whole point of separating the two.
GATE_GRID = (0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95)
KEEP_GRID = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90)
# Margins below the question's own best. 0.0 keeps only the joint winners.
MARGIN_GRID = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)
QA_GRID = (0.0, 0.005, 0.01, 0.05)
# The absolute pair shipped before this search, kept as a literal: settings.py
# now carries the relative rule this found, so reading it back would compare the
# winner with itself.
SHIPPED = (0.90, 0.90, 0.05)
TOP = 10


def kept(question: dict, gate: float, keep: float, qa: float, relative: bool = False) -> set:
    """The chunks highlighted: nothing at all unless one candidate clears ``gate``.

    ``keep`` is an absolute score, or a margin below the question's best when
    ``relative``.
    """
    candidates = question["candidates"]
    floor_if_answered = rerank_floor((r for _, r, _ in candidates), gate, keep)
    if floor_if_answered is None:
        return set()
    floor = floor_if_answered if relative else keep
    return {c for c, r, s in candidates if r >= floor and s >= qa}


def question_scores(questions: list[dict], gate: float, keep: float, qa: float,
                    relative: bool = False) -> list[dict]:
    scores = []
    for q in questions:
        highlighted = kept(q, gate, keep, qa, relative)
        scores.append({**evidence_prf(highlighted, q["gold"]), "empty": not highlighted})
    return scores


def silence(negatives: list[dict], gate: float, keep: float, qa: float,
            relative: bool = False) -> dict[str, float]:
    """Per source, the share of unanswerable questions left with nothing highlighted."""
    by_source = defaultdict(list)
    for q in negatives:
        by_source[q["source"]].append(not kept(q, gate, keep, qa, relative))
    return {f"silent_{source}": float(np.mean(v)) for source, v in sorted(by_source.items())}


def evaluate(questions: list[dict], gate: float, keep: float, qa: float, negatives=(),
             relative: bool = False) -> dict:
    scores = question_scores(questions, gate, keep, qa, relative)
    mean = {k: float(np.mean([s[k] for s in scores]))
            for k in ("evidence_f1", "evidence_precision", "evidence_recall", "empty")}
    return {"gate": gate, "keep": keep, "qa": qa, **mean,
            **silence(negatives, gate, keep, qa, relative)}


def grid_search(questions: list[dict], negatives=(), relative: bool = False) -> list[dict]:
    """Every triple worth scoring, best evidence F1 first.

    An absolute ``keep`` above the gate would cut candidates the gate let in, so
    those pairs are skipped; a margin has no such ceiling.
    """
    keeps = MARGIN_GRID if relative else KEEP_GRID
    return sorted((evaluate(questions, g, k, q, negatives, relative)
                   for g in GATE_GRID for k in keeps if relative or k <= g
                   for q in QA_GRID),
                  key=lambda row: -row["evidence_f1"])


def triples(grid: list[dict]) -> dict[tuple, dict]:
    return {(r["gate"], r["keep"], r["qa"]): r for r in grid}


def common_best(grids: dict[str, list[dict]], target: float = SILENCE_TARGET) -> tuple:
    """Best mean evidence F1 among the triples silent enough off topic on every set."""
    rows = {name: triples(grid) for name, grid in grids.items()}
    shared = set.intersection(*(set(r) for r in rows.values()))
    silent = {t: min(r[t].get("silent_offtopic", 1.0) for r in rows.values()) for t in shared}
    f1 = {t: float(np.mean([r[t]["evidence_f1"] for r in rows.values()])) for t in shared}
    allowed = [t for t in shared if silent[t] >= target] or list(shared)
    return max(allowed, key=f1.get)


def paired_delta(questions: list[dict], triple: tuple, reference: tuple,
                 relative: bool = False) -> str:
    deltas = [a["evidence_f1"] - b["evidence_f1"] for a, b in
              zip(question_scores(questions, *triple, relative),
                  question_scores(questions, *reference), strict=True)]
    low, high = bootstrap_ci(deltas)
    return f"{np.mean(deltas):+.3f} [{low:+.3f}, {high:+.3f}]"


HEADER = ["gate", "keep", "qa", "evidence F1", "precision", "recall", "left empty"]
FAMILIES = (("absolute", False), ("relative", True))


def _cells(row: dict, silent_keys: list[str]) -> list:
    return [row["gate"], row["keep"], row["qa"], f"{row['evidence_f1']:.3f}",
            f"{row['evidence_precision']:.3f}", f"{row['evidence_recall']:.3f}",
            f"{row['empty']:.1%}", *(f"{row[k]:.1%}" for k in silent_keys)]


def report(sets: dict[str, list[dict]], negatives: dict[str, list[dict]]) -> str:
    out = ["# Answering once per question, then highlighting",
           (f"`gate`: a question is answered only if its best candidate scores this or more. "
            f"`keep`: absolute cut on the candidates of an answered question, or, in the "
            f"relative family, how far below that question's own best a candidate may sit. "
            f"`qa`: the reader's veto. The shipped pair is the absolute rule with "
            f"gate == keep {SHIPPED}. Chosen per family: best mean evidence F1 with "
            f"silent_offtopic >= {SILENCE_TARGET:.0%} on every set.")]
    chosen = {}
    for family, relative in FAMILIES:
        grids = {n: grid_search(q, negatives.get(n, ()), relative) for n, q in sets.items()}
        chosen[family] = (common_best(grids), relative)
        for name, grid in grids.items():
            keys = sorted(k for k in grid[0] if k.startswith("silent_"))
            header = [*HEADER, *keys]
            references = [
                ("shipped", evaluate(sets[name], *SHIPPED, negatives.get(name, ()))),
                ("chosen", triples(grid)[chosen[family][0]])]
            out += [f"## {name}, {family}: best {TOP} by evidence F1",
                    markdown_table(header, [_cells(r, keys) for r in grid[:TOP]]),
                    f"## {name}, {family}: chosen against shipped",
                    markdown_table(["", *header], [[label, *_cells(r, keys)] for label, r in references])]
    out += ["## Each family's pick against the shipped pair",
            "Paired difference in evidence F1 per question, 95% bootstrap interval.",
            markdown_table(["family", "gate", "keep", "qa", *(f"Δ on {n}" for n in sets)],
                           [[family, *triple,
                             *(paired_delta(q, triple, SHIPPED, relative) for q in sets.values())]
                            for family, (triple, relative) in chosen.items()])]
    return "\n\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dirs", nargs="+", default=["output_qasper", "output_qasa"],
                        help="each holds golden_set_aligned.jsonl, chunks/ and the score files")
    args = parser.parse_args()
    dirs = [Path(d) for d in args.golden_dirs]
    negatives = {d.name: group_negatives(load_jsonl(d / "negative_scores.jsonl"))
                 for d in dirs if (d / "negative_scores.jsonl").exists()}
    out_file = dirs[-1] / "answer_gate.md"
    out_file.write_text(report({d.name: load_questions(d) for d in dirs}, negatives))
    print(f"-> {out_file}")


if __name__ == "__main__":
    main()
