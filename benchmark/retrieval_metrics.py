"""Retrieval and attribution metrics, scored against aligned gold chunks.

Not covered by ``qasper_evaluator.py`` (see benchmark/README.md):

- recall@k / precision@k / MRR isolate the retrieval stage, which acts before
  QA extraction and is invisible to answer F1 alone. precision@k (and the
  threshold-aware ``evidence_prf``) is what makes the filtering parameters
  measurable at all: recall@k scores the ranked list *before* the cut, so a
  threshold can never move it.
- ``evidence_prf`` is the attribution metric the thesis needs for the
  highlighting deliverable. It deliberately scores *chunk ids*, not strings:
  QASPER's official Evidence F1 matches evidence by exact string equality
  against LaTeX-derived text that our GROBID/PDF-extracted paragraphs never
  equal verbatim, so it evaluates to 0.0 regardless of configuration. The ids
  come from benchmark/align_evidence.py's fuzzy overlap alignment.

``bootstrap_ci`` exists because the golden sets are small (120 aligned
questions on QASPER): without an interval, a one-point difference between two
ablation rows is indistinguishable from noise.
"""

import random
from collections.abc import Hashable, Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[Hashable], gold_ids: Iterable[Hashable], k: int) -> float:
    """Fraction of ``gold_ids`` present in the top ``k`` of ``ranked_ids``."""
    gold = set(gold_ids)
    if not gold:
        raise ValueError("gold_ids cannot be empty")
    top_k = set(ranked_ids[:k])
    return len(gold & top_k) / len(gold)


def precision_at_k(ranked_ids: Sequence[Hashable], gold_ids: Iterable[Hashable], k: int) -> float:
    """Fraction of the top ``k`` returned ids that are gold.

    Denominator is what was actually returned (``min(k, len(ranked_ids))``), so
    a configuration that returns fewer, better chunks scores higher - this is
    the half of the picture recall@k structurally cannot see.
    """
    gold = set(gold_ids)
    if not gold:
        raise ValueError("gold_ids cannot be empty")
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    return len(gold & set(top_k)) / len(top_k)


def reciprocal_rank(ranked_ids: Sequence[Hashable], gold_ids: Iterable[Hashable]) -> float:
    """1/rank of the first gold id found in ``ranked_ids`` (0.0 if none found)."""
    gold = set(gold_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def evidence_prf(predicted_ids: Iterable[Hashable], gold_ids: Iterable[Hashable]) -> dict:
    """Precision/recall/F1 of the evidence the system actually attributed.

    ``predicted_ids`` is the *unranked* set of chunks the system surfaced as
    supporting evidence (what the highlighter would mark in the PDF), so unlike
    recall@k this is sensitive to every filtering threshold in the pipeline.
    """
    predicted, gold = set(predicted_ids), set(gold_ids)
    if not gold:
        raise ValueError("gold_ids cannot be empty")
    if not predicted:
        return {"evidence_precision": 0.0, "evidence_recall": 0.0, "evidence_f1": 0.0}
    hits = len(predicted & gold)
    precision = hits / len(predicted)
    recall = hits / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"evidence_precision": precision, "evidence_recall": recall, "evidence_f1": f1}


def bootstrap_ci(values: Sequence[float], confidence: float = 0.95,
                 resamples: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of ``values``.

    Non-parametric on purpose: per-question F1 and recall are bounded and
    heavily skewed, so a normal-approximation interval would be wrong at the
    edges where these metrics actually live.
    """
    if not values:
        raise ValueError("values cannot be empty")
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    lo = means[int((1 - confidence) / 2 * resamples)]
    hi = means[min(int((1 + confidence) / 2 * resamples), resamples - 1)]
    return (lo, hi)


def per_question_scores(ranked_ids: Sequence[Hashable], gold_ids: Iterable[Hashable],
                        k: int, predicted_ids: Iterable[Hashable] | None = None) -> dict:
    """Every retrieval/attribution metric for a single question.

    Keeping the per-question values (rather than aggregating immediately) is
    what lets the caller stratify by question type and bootstrap a CI.
    ``predicted_ids`` defaults to the top ``k``, i.e. "the system attributes
    what it returned".
    """
    scores = {
        f"recall@{k}": recall_at_k(ranked_ids, gold_ids, k),
        f"precision@{k}": precision_at_k(ranked_ids, gold_ids, k),
        "mrr": reciprocal_rank(ranked_ids, gold_ids),
    }
    scores.update(evidence_prf(
        ranked_ids[:k] if predicted_ids is None else predicted_ids, gold_ids))
    return scores


def aggregate(per_question: Sequence[tuple[Sequence[Hashable], Iterable[Hashable]]],
              k: int, with_ci: bool = False) -> dict:
    """Mean of every metric over a batch of (ranked_ids, gold_ids) pairs.

    With ``with_ci``, each metric also gets ``<name>_ci_low``/``_ci_high``.
    """
    if not per_question:
        raise ValueError("per_question cannot be empty")
    rows = [per_question_scores(ranked, gold, k) for ranked, gold in per_question]
    return summarize(rows, with_ci=with_ci)


def summarize(rows: Sequence[dict], with_ci: bool = False) -> dict:
    """Mean (and optional bootstrap CI) of every metric across per-question rows."""
    if not rows:
        raise ValueError("rows cannot be empty")
    result = {}
    for metric in rows[0]:
        values = [row[metric] for row in rows]
        result[metric] = sum(values) / len(values)
        if with_ci:
            lo, hi = bootstrap_ci(values)
            result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = lo, hi
    result["n_questions"] = len(rows)
    return result
