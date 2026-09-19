"""How the extractive reader's span score separates gold chunks from the rest.

qa_score_threshold drops every answer whose span score is below it, and the
answer's highlight with it. Its values (0.05/0.10/0.20 in the ablation) were
never read off the scores themselves. This measures them on the chunks the
reader is actually given: every question's hybrid top-LIMIT, which does not
depend on the reranker (a reranker reorders and filters that list, it adds
nothing to it).

Two scores per chunk:

- ``masked``: the score the reader used to ship, renormalised over the context
  tokens only, so the model's "no answer" option ([CLS] in a SQuAD 2.0 model)
  is removed and every chunk yields a span, however unrelated.
- ``with_null``: the same span with [CLS] kept in the normalisation, so the
  mass the model puts on "no answer" lowers the score. ``null`` is that mass.
  The best span cannot change: keeping [CLS] only rescales the context
  probabilities, so ``with_null = masked * (1 - P_start[CLS]) * (1 - P_end[CLS])``.
  This run is what moved ``_best_span`` to it: it is the score shipped now.

The report gives the distribution of both on gold and non-gold chunks, the
pooled AUC (the chance a gold chunk outscores a non-gold one, which is what a
single global threshold relies on), and a sweep of thresholds. Every score is
also written per chunk, so another cut can be studied without re-running.

Usage:
  python -m benchmark.qa_score_distribution --golden-dir output_qasa
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# appended, never prepended: the repo root must keep priority over this directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from qa_engine import (
    CONTEXT_STRIDE,
    MAX_SEQUENCE_TOKENS,
    QAEngine,
    _best_span,
    _softmax,
)

from benchmark.ablation import load_jsonl
from benchmark.paraphrase_retrieval import LIMIT, HybridSearch
from benchmark.retrieval_score_distribution import _table, gold_ids

SCORES = ("masked", "with_null")
THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
PERCENTILES = (10, 25, 50, 75, 90)


def span_scores(start_logits, end_logits, sequence_ids) -> dict | None:
    """One window's best span, scored with (as shipped) and without the "no answer" option."""
    best = _best_span(start_logits, end_logits, sequence_ids)
    if best is None:
        return None
    start_token, end_token, with_null = best
    context = np.array([sid == 1 for sid in sequence_ids])
    with_cls = context.copy()
    with_cls[0] = True  # [CLS]: where a SQuAD 2.0 model puts "no answer"
    start, end = (_softmax(np.where(context, logits, -np.inf))
                  for logits in (start_logits, end_logits))
    start_cls, end_cls = (_softmax(np.where(with_cls, logits, -np.inf))
                          for logits in (start_logits, end_logits))
    return {"masked": float(start[start_token] * end[end_token]),
            "with_null": with_null,
            "null": float(start_cls[0] * end_cls[0]),
            "start": start_token, "end": end_token}


def chunk_scores(windows: list[dict | None]) -> dict | None:
    """A chunk's scores over its windows, the way the reader picks among them.

    The span is the best-scoring window's, as in extract_answers; ``null`` is
    the least "no answer" any window gives, since one window holding the
    answer is enough.
    """
    windows = [w for w in windows if w]
    if not windows:
        return None
    best = max(windows, key=lambda w: w["masked"])
    return {**best,
            "with_null": max(w["with_null"] for w in windows),
            "null": min(w["null"] for w in windows)}


def reader_scores(engine, question: str, contexts: list[str]) -> list[dict | None]:
    """Each context's span scores, tokenised and windowed the way extract_answers does."""
    encoded = engine.tokenizer(
        [question] * len(contexts), contexts, return_tensors="pt", padding=True,
        truncation="only_second", max_length=MAX_SEQUENCE_TOKENS, stride=CONTEXT_STRIDE,
        return_overflowing_tokens=True)
    sample_of = encoded.pop("overflow_to_sample_mapping").tolist()
    sequence_ids = [encoded.sequence_ids(k) for k in range(len(sample_of))]
    starts, ends = engine._logits_in_windows(encoded.to(engine.device))
    windows = defaultdict(list)
    for k, sample in enumerate(sample_of):
        windows[sample].append(span_scores(starts[k], ends[k], sequence_ids[k]))
    return [chunk_scores(windows[i]) for i in range(len(contexts))]


def auc(gold, other) -> float:
    """P(a gold score beats a non-gold one), ties counting half."""
    other = np.sort(np.asarray(other, dtype=float))
    gold = np.asarray(gold, dtype=float)
    below = np.searchsorted(other, gold, side="left")
    not_above = np.searchsorted(other, gold, side="right")
    return float(np.mean((below + not_above) / 2) / len(other))


def cut_stats(rows: list[dict], keep) -> dict:
    """What the cut ``keep(row) -> bool`` keeps, per chunk and per question."""
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)
    with_gold = [q for q in by_question.values() if any(r["gold"] for r in q)]
    kept = [[r for r in q if keep(r)] for q in by_question.values()]
    n_kept = sum(len(k) for k in kept)
    gold = [keep(r) for r in rows if r["gold"]]
    other = [keep(r) for r in rows if not r["gold"]]
    return {
        "gold_kept": float(np.mean(gold)),
        "other_kept": float(np.mean(other)),
        "precision": sum(gold) / n_kept if n_kept else 0.0,
        "kept_per_question": n_kept / len(by_question),
        "questions_gold_kept": float(np.mean(
            [any(r["gold"] and keep(r) for r in q) for q in with_gold])),
        "questions_empty": float(np.mean([not k for k in kept])),
    }


def sweep(rows: list[dict], score: str, thresholds=THRESHOLDS) -> list[dict]:
    """What a global threshold on ``score`` keeps, at each of ``thresholds``."""
    return [{"threshold": t, **cut_stats(rows, lambda r, t=t: r[score] >= t)}
            for t in thresholds]


STATS_HEADER = ["gold kept", "non-gold kept", "precision", "kept / question",
                "questions with gold kept", "questions left empty"]


def stats_cells(stats: dict) -> list[str]:
    return [f"{stats['gold_kept']:.1%}", f"{stats['other_kept']:.1%}",
            f"{stats['precision']:.3f}", f"{stats['kept_per_question']:.1f}",
            f"{stats['questions_gold_kept']:.1%}", f"{stats['questions_empty']:.1%}"]


def distribution_rows(rows: list[dict], scores) -> list[list]:
    """Percentiles of each score, gold and non-gold apart."""
    out = []
    for score in scores:
        for label, gold in (("gold", True), ("non-gold", False)):
            values = np.array([r[score] for r in rows if r["gold"] == gold])
            out.append([score, label, len(values),
                        *(f"{np.percentile(values, p):.3f}" for p in PERCENTILES)])
    return out


DISTRIBUTION_HEADER = ["score", "chunks", "n", *(f"p{p}" for p in PERCENTILES)]


def report(name: str, rows: list[dict], n_questions: int) -> str:
    gold_rows = [r for r in rows if r["gold"]]
    other_rows = [r for r in rows if not r["gold"]]
    separation = [[score, f"{auc([r[score] for r in gold_rows], [r[score] for r in other_rows]):.3f}"]
                  for score in SCORES]
    out = [f"# Span score distribution, extractive reader: {name}",
           (f"{n_questions} questions, hybrid top-{LIMIT} each, {len(gold_rows)} gold and "
            f"{len(other_rows)} non-gold chunks."),
           "## Distribution",
           _table(DISTRIBUTION_HEADER, distribution_rows(rows, (*SCORES, "null"))),
           "## Separation (pooled AUC; 0.5 = none)",
           _table(["score", "AUC gold vs non-gold"], separation)]
    for score in SCORES:
        out += [f"## Sweep: {score}", _table(["threshold", *STATS_HEADER], [
            [r["threshold"], *stats_cells(r)] for r in sweep(rows, score)])]
    return "\n\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="output_qasa",
                        help="holds golden_set_aligned.jsonl and chunks/")
    args = parser.parse_args()

    golden_dir = Path(args.golden_dir)
    search = HybridSearch(golden_dir / "chunks")
    engine = QAEngine()  # the shipped reader, its device and precision

    rows, n_questions = [], 0
    for record in load_jsonl(golden_dir / "golden_set_aligned.jsonl"):
        gold = {g for g in gold_ids(record) if g in search.text_of}
        if not gold:
            continue
        n_questions += 1
        hits = search.top(record["question"])
        scored = reader_scores(engine, record["question"], [search.text_of[c] for c in hits])
        rows += [{"question_id": record["question_id"], "chunk": list(chunk),
                  "gold": chunk in gold, "masked": s["masked"],
                  "with_null": s["with_null"], "null": s["null"]}
                 for chunk, s in zip(hits, scored, strict=True) if s]

    (golden_dir / "qa_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out_file = golden_dir / "qa_score_distribution.md"
    out_file.write_text(report(golden_dir.name, rows, n_questions))
    print(f"{n_questions} questions, {len(rows)} chunks -> {out_file}")


if __name__ == "__main__":
    main()
