"""Choosing rerank_threshold and qa_score_threshold together, and checking the choice.

A chunk is highlighted when the reranker scores it at or above rerank_threshold
and the reader's span score clears qa_score_threshold. The score files written by
benchmark/studies/rerank_score_distribution.py and benchmark/studies/qa_score_distribution.py
hold both scores for every question's hybrid top-LIMIT, so every pair of
thresholds is scored here offline and exactly, without running a model.

Two sides are read. On the golden set's answerable questions the objective is
the ablation's own evidence F1 (benchmark/metrics/retrieval_metrics.py): per question,
over the chunks highlighted, against *every* gold chunk of the question,
including those retrieval never made candidates; a question left empty scores 0.
On questions the library does not answer (benchmark/studies/negative_scores.py) the
right output is nothing, so they are scored by the share left empty. A pair that
highlights something for every question maximises the first and fails the
second, which is why a threshold is needed at all.

The pair to ship is the best mean evidence F1 over the sets among the pairs that
leave at least SILENCE_TARGET of the off-topic questions empty on every set.
Each set's own optimum is also scored on the other set: a pair tuned and read on
the same questions overstates what it does on a library nobody tuned it for.

Usage:
  python -m benchmark.studies.threshold_search --golden-dirs data/dev/qasper data/dev/qasa
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.metrics.reports import markdown_table
from benchmark.metrics.retrieval_metrics import evidence_prf, format_delta
from benchmark.studies.stored_scores import (
    SILENCE_TARGET,
    load_negatives,
    load_questions,
)

# gte's probabilities crowd the top, so the rerank grid is fine above 0.8; 0.0 is "off"
RERANK_GRID = (0.0, *(round(0.80 + 0.01 * i, 2) for i in range(17)))
QA_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
SHIPPED = (0.001, 0.10)  # the general preset's pair before this search
TOP = 10


def highlighted(question: dict, rerank: float, qa: float) -> set:
    return {c for c, r, s in question["candidates"] if r >= rerank and s >= qa}


def question_scores(questions: list[dict], rerank: float, qa: float) -> list[dict]:
    out = []
    for q in questions:
        kept = highlighted(q, rerank, qa)
        out.append({**evidence_prf(kept, q["gold"]), "empty": not kept})
    return out


def silence(negatives: list[dict], rerank: float, qa: float) -> dict[str, float]:
    """Per source, the share of questions left with nothing highlighted, as they should be."""
    by_source = defaultdict(list)
    for q in negatives:
        by_source[q["source"]].append(not highlighted(q, rerank, qa))
    return {f"silent_{source}": float(np.mean(v)) for source, v in sorted(by_source.items())}


def evaluate(questions: list[dict], rerank: float, qa: float, negatives=()) -> dict:
    scores = question_scores(questions, rerank, qa)
    mean = {k: float(np.mean([s[k] for s in scores]))
            for k in ("evidence_f1", "evidence_precision", "evidence_recall", "empty")}
    return {"rerank": rerank, "qa": qa, **mean, **silence(negatives, rerank, qa)}


def grid_search(questions: list[dict], negatives=(), rerank_grid=RERANK_GRID,
                qa_grid=QA_GRID) -> list[dict]:
    """Every pair, best evidence F1 first."""
    rows = [evaluate(questions, r, q, negatives) for r in rerank_grid for q in qa_grid]
    return sorted(rows, key=lambda row: -row["evidence_f1"])


def over_sets(grids: dict[str, list[dict]], key: str, combine) -> dict[tuple, float]:
    """``combine`` of ``key`` over the sets that report it, per pair."""
    values = defaultdict(list)
    for grid in grids.values():
        for row in grid:
            if key in row:
                values[(row["rerank"], row["qa"])].append(row[key])
    return {pair: float(combine(v)) for pair, v in values.items()}


def common_best(grids: dict[str, list[dict]], target: float = SILENCE_TARGET) -> tuple:
    """Best mean evidence F1 among the pairs silent enough off topic on every set.

    With no pair reaching ``target``, the most silent pairs are the candidates.
    """
    f1 = over_sets(grids, "evidence_f1", np.mean)
    silent = over_sets(grids, "silent_offtopic", min)
    if not silent:
        return max(f1, key=f1.get)
    allowed = [p for p in f1 if silent[p] >= target]
    if not allowed:
        most_silent = max(silent.values())
        allowed = [p for p in f1 if silent[p] == most_silent]
    return max(allowed, key=f1.get)


def pareto(f1: dict[tuple, float], silent: dict[tuple, float]) -> list[tuple]:
    """Pairs no other pair beats on both evidence F1 and silence, most silent first."""
    front, best = [], -1.0
    for pair in sorted(f1, key=lambda p: (-silent[p], -f1[p])):
        if f1[pair] > best:
            front.append(pair)
            best = f1[pair]
    return front


def paired_delta(questions: list[dict], pair: tuple, reference: tuple) -> str:
    return format_delta([a["evidence_f1"] - b["evidence_f1"] for a, b in
                         zip(question_scores(questions, *pair),
                             question_scores(questions, *reference), strict=True)])


BASE_HEADER = ["rerank", "qa", "evidence F1", "precision", "recall", "left empty"]


def _cells(row: dict, silent_keys: list[str]) -> list:
    return [row["rerank"], row["qa"], f"{row['evidence_f1']:.3f}",
            f"{row['evidence_precision']:.3f}", f"{row['evidence_recall']:.3f}",
            f"{row['empty']:.1%}", *(f"{row[k]:.1%}" for k in silent_keys)]


def report(sets: dict[str, list[dict]], negatives: dict[str, list[dict]]) -> str:
    grids = {n: grid_search(q, negatives.get(n, ())) for n, q in sets.items()}
    chosen = common_best(grids)
    names = list(sets)
    out = ["# Rerank and QA thresholds, searched together",
           (f"Grid: rerank {RERANK_GRID[1]}-{RERANK_GRID[-1]} by 0.01 and off, qa {list(QA_GRID)}. "
            "Evidence F1 per answerable question against every gold chunk, averaged; "
            "empty = 0. `silent_*`: share of the questions the library does not answer "
            f"that come back empty. Pair chosen: best mean F1 with silent_offtopic >= "
            f"{SILENCE_TARGET:.0%} on every set.")]
    for name, grid in grids.items():
        keys = sorted(k for k in grid[0] if k.startswith("silent_"))
        header = [*BASE_HEADER, *keys]
        references = [
            ("shipped before", evaluate(sets[name], *SHIPPED, negatives.get(name, ()))),
            ("reranker only", max((r for r in grid if r["qa"] == 0.0), key=lambda r: r["evidence_f1"])),
            ("reader only", max((r for r in grid if r["rerank"] == 0.0), key=lambda r: r["evidence_f1"])),
            ("chosen", next(r for r in grid if (r["rerank"], r["qa"]) == chosen))]
        out += [f"## {name}: best {TOP} by evidence F1",
                markdown_table(header, [_cells(r, keys) for r in grid[:TOP]]),
                f"## {name}: references",
                markdown_table(["", *header], [[label, *_cells(r, keys)] for label, r in references])]
    silent = over_sets(grids, "silent_offtopic", min)
    if silent:
        f1 = over_sets(grids, "evidence_f1", np.mean)
        per_set = {n: {(r["rerank"], r["qa"]): r for r in g} for n, g in grids.items()}
        out += ["## Frontier: mean evidence F1 against off-topic silence (worst set)",
                markdown_table(["rerank", "qa", "mean F1", "silent off-topic",
                                *(f"F1 {n}" for n in names)],
                               [[*p, f"{f1[p]:.3f}", f"{silent[p]:.1%}",
                                 *(f"{per_set[n][p]['evidence_f1']:.3f}" for n in names)]
                                for p in pareto(f1, silent)])]
    picks = [(f"best F1 on {n}", (grids[n][0]["rerank"], grids[n][0]["qa"])) for n in names]
    transfer = [[label, *pair, *(f"{evaluate(sets[t], *pair)['evidence_f1']:.3f}" for t in names)]
                for label, pair in [*picks, ("chosen", chosen)]]
    out += ["## Transfer: each pick scored on every set",
            markdown_table(["picked by", "rerank", "qa", *(f"F1 on {n}" for n in names)], transfer),
            f"## Chosen pair {chosen} against the pair shipped before {SHIPPED}",
            "Paired difference in evidence F1 per question, 95% bootstrap interval.",
            markdown_table(["set", "Δ evidence F1"],
                           [[n, paired_delta(sets[n], chosen, SHIPPED)] for n in names])]
    return "\n\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dirs", nargs="+", default=["data/dev/qasper", "data/dev/qasa"],
                        help="each holds golden_set_aligned.jsonl, chunks/ and the score files")
    args = parser.parse_args()
    dirs = [Path(d) for d in args.golden_dirs]
    negatives = {d.name: scored for d in dirs if (scored := load_negatives(d))}
    out_file = dirs[-1] / "threshold_search.md"
    out_file.write_text(report({d.name: load_questions(d) for d in dirs}, negatives))
    print(f"-> {out_file}")


if __name__ == "__main__":
    main()
