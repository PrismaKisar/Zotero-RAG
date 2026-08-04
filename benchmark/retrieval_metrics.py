"""Recall@k and MRR for the retrieval stage, scored against aligned gold chunks.

Not covered by ``qasper_evaluator.py`` (see benchmark/README.md), needed for the
ablation study to isolate the effect of retrieval_threshold/rerank_threshold,
which act before QA extraction and are invisible to answer/evidence F1 alone.
"""

from typing import Hashable, Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[Hashable], gold_ids: Iterable[Hashable], k: int) -> float:
    """Fraction of ``gold_ids`` present in the top ``k`` of ``ranked_ids``."""
    gold = set(gold_ids)
    if not gold:
        raise ValueError("gold_ids cannot be empty")
    top_k = set(ranked_ids[:k])
    return len(gold & top_k) / len(gold)


def reciprocal_rank(ranked_ids: Sequence[Hashable], gold_ids: Iterable[Hashable]) -> float:
    """1/rank of the first gold id found in ``ranked_ids`` (0.0 if none found)."""
    gold = set(gold_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def aggregate(per_question: Sequence[tuple[Sequence[Hashable], Iterable[Hashable]]], k: int) -> dict:
    """Mean recall@k and MRR over a batch of (ranked_ids, gold_ids) pairs."""
    if not per_question:
        raise ValueError("per_question cannot be empty")
    recalls = [recall_at_k(ranked, gold, k) for ranked, gold in per_question]
    mrrs = [reciprocal_rank(ranked, gold) for ranked, gold in per_question]
    return {
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "n_questions": len(per_question),
    }
