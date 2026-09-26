"""The rerank cut, as one rule shared by the pipeline and the offline studies.

The gate reads the best candidate alone and decides whether the question is
answered; the margin below that best decides what else is kept. The reranker
applies it online and benchmark/answer_gate.py and benchmark/failure_attribution.py
replay it on saved scores, so a change here moves all three together.
"""

from zotero_rag.selection import rerank_floor


def test_no_candidate_reaching_the_gate_leaves_nothing_to_keep():
    assert rerank_floor([0.86, 0.87], gate=0.88, margin=0.02) is None


def test_the_floor_sits_margin_below_this_question_s_best():
    assert rerank_floor([0.95, 0.94, 0.92], gate=0.88, margin=0.02) == 0.95 - 0.02


def test_a_best_exactly_on_the_gate_is_answered():
    assert rerank_floor([0.88], gate=0.88, margin=0.0) == 0.88


def test_no_candidates_leave_nothing_to_keep():
    assert rerank_floor([], gate=0.0, margin=1.0) is None
