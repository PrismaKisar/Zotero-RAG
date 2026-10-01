"""Which reranked candidates reach the reader: the gate and the margin, as one rule.

The reranker applies it to live scores; benchmark/answer_gate.py and
benchmark/failure_attribution.py replay it on the saved ones. Kept free of torch
so the offline studies import the rule itself instead of restating it.
"""


def rerank_floor(scores, gate: float, margin: float,
                 absolute: bool = False) -> float | None:
    """The lowest rerank score kept, or None when the question goes unanswered.

    ``gate`` is read off the best score alone: below it nothing is kept, however
    many middling candidates there are. A question that clears it keeps every
    candidate scoring at least ``margin`` below that best one.

    ``absolute`` is the rule the margin replaced, kept so the benchmarks can
    cross it with the reader: every candidate at or above ``gate``, whatever the
    best one scored, and ``margin`` unread.
    """
    best = max(scores, default=None)
    if best is None or best < gate:
        return None
    return gate if absolute else best - margin
