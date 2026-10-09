import pytest

from benchmark.studies.stored_scores import group, group_negatives
from benchmark.studies.threshold_search import (
    common_best,
    grid_search,
    pareto,
    question_scores,
    report,
    silence,
)

ROWS = [
    {"question_id": "a", "chunk": ["p", 1], "rerank": 0.95, "qa": 0.5},
    {"question_id": "a", "chunk": ["p", 2], "rerank": 0.85, "qa": 0.0},
    {"question_id": "b", "chunk": ["q", 1], "rerank": 0.90, "qa": 0.2},
    {"question_id": "b", "chunk": ["q", 2], "rerank": 0.92, "qa": 0.01},
]
# ("q", 9) is gold that retrieval never made a candidate: it still counts against recall
GOLD = {"a": {("p", 1)}, "b": {("q", 1), ("q", 9)}}
QUESTIONS = group(ROWS, GOLD)
NEGATIVES = group_negatives([
    {"question_id": "x", "source": "offtopic", "chunk": ["p", 3], "rerank": 0.70, "qa": 0.3},
    {"question_id": "y", "source": "offtopic", "chunk": ["q", 3], "rerank": 0.93, "qa": 0.0},
    {"question_id": "z", "source": "unanswerable", "chunk": ["p", 4], "rerank": 0.91, "qa": 0.2},
])


def test_recall_counts_gold_retrieval_never_found():
    [_, b] = question_scores(QUESTIONS, 0.0, 0.0)
    assert b["evidence_precision"] == 0.5 and b["evidence_recall"] == 0.5


def test_a_question_left_with_nothing_scores_zero_and_is_flagged():
    scores = question_scores(QUESTIONS, 0.96, 0.0)
    assert all(s["empty"] and s["evidence_f1"] == 0.0 for s in scores)


def test_silence_is_the_share_of_unanswerable_questions_left_empty_per_source():
    # rerank 0.9 silences x; y (0.93) still gets a highlight; z (0.91) too
    assert silence(NEGATIVES, 0.9, 0.0) == {"silent_offtopic": 0.5, "silent_unanswerable": 0.0}
    # the reader's cut then silences y and leaves z, whose span scores 0.2
    assert silence(NEGATIVES, 0.9, 0.1) == {"silent_offtopic": 1.0, "silent_unanswerable": 0.0}


def test_grid_search_puts_the_best_pair_first():
    grid = grid_search(QUESTIONS, rerank_grid=(0.0, 0.9, 0.93), qa_grid=(0.0, 0.1))
    # the qa cut drops a2 and b2: a is exact, b keeps q1 of its two gold chunks
    assert grid[0]["evidence_f1"] == pytest.approx((1 + 2 / 3) / 2)
    assert grid[0]["qa"] == 0.1
    assert grid[-1]["evidence_f1"] < grid[0]["evidence_f1"]


def test_common_best_maximises_the_mean_over_sets_without_negatives():
    grids = {"x": [{"rerank": 0.9, "qa": 0.1, "evidence_f1": 0.8},
                   {"rerank": 0.8, "qa": 0.0, "evidence_f1": 0.7}],
             "y": [{"rerank": 0.9, "qa": 0.1, "evidence_f1": 0.2},
                   {"rerank": 0.8, "qa": 0.0, "evidence_f1": 0.6}]}
    assert common_best(grids) == (0.8, 0.0)


def test_common_best_only_considers_pairs_silent_enough_on_every_set():
    grids = {"x": [{"rerank": 0.8, "qa": 0.0, "evidence_f1": 0.9, "silent_offtopic": 0.5},
                   {"rerank": 0.9, "qa": 0.1, "evidence_f1": 0.4, "silent_offtopic": 0.95}],
             "y": [{"rerank": 0.8, "qa": 0.0, "evidence_f1": 0.9, "silent_offtopic": 1.0},
                   {"rerank": 0.9, "qa": 0.1, "evidence_f1": 0.4, "silent_offtopic": 0.92}]}
    assert common_best(grids, target=0.9) == (0.9, 0.1)
    # nothing reaches 0.99: fall back on the most silent pairs
    assert common_best(grids, target=0.99) == (0.9, 0.1)


def test_pareto_keeps_only_pairs_no_other_beats_on_both():
    f1 = {(1, 0): 0.9, (2, 0): 0.5, (3, 0): 0.7}
    silent = {(1, 0): 0.2, (2, 0): 0.6, (3, 0): 0.8}
    # (2, 0) is dominated by (3, 0): louder and worse
    assert pareto(f1, silent) == [(3, 0), (1, 0)]


def test_report_reads_both_sides_and_scores_each_pick_on_every_set():
    text = report({"s1": QUESTIONS, "s2": QUESTIONS}, {"s1": NEGATIVES})
    for part in ("## s1: best 10", "silent_offtopic", "silent_unanswerable", "## Frontier",
                 "## Transfer", "chosen", "F1 on s2", "Δ evidence F1"):
        assert part in text
