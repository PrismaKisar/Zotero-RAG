import pytest

from benchmark.studies.answer_gate import (
    MARGIN_GRID,
    common_best,
    evaluate,
    grid_search,
    kept,
    question_scores,
    report,
    silence,
)
from benchmark.studies.stored_scores import group, group_negatives

ROWS = [
    # one strong candidate opens the gate; the weak gold chunk rides in behind it
    {"question_id": "a", "chunk": ["p", 1], "rerank": 0.95, "qa": 0.5},
    {"question_id": "a", "chunk": ["p", 2], "rerank": 0.60, "qa": 0.5},
    # nothing strong: the question stays unanswered however many candidates it has
    {"question_id": "b", "chunk": ["q", 1], "rerank": 0.70, "qa": 0.5},
    {"question_id": "b", "chunk": ["q", 2], "rerank": 0.75, "qa": 0.5},
]
GOLD = {"a": {("p", 1), ("p", 2)}, "b": {("q", 1)}}
QUESTIONS = group(ROWS, GOLD)
NEGATIVES = group_negatives([
    {"question_id": "x", "source": "offtopic", "chunk": ["p", 3], "rerank": 0.70, "qa": 0.5},
    {"question_id": "y", "source": "offtopic", "chunk": ["q", 3], "rerank": 0.95, "qa": 0.5},
])
# two questions whose candidates crowd at different heights: the point of a margin
CROWDED = group([
    {"question_id": "high", "chunk": ["p", 1], "rerank": 0.95, "qa": 0.5},
    {"question_id": "high", "chunk": ["p", 2], "rerank": 0.93, "qa": 0.5},
    {"question_id": "low", "chunk": ["q", 1], "rerank": 0.91, "qa": 0.5},
    {"question_id": "low", "chunk": ["q", 2], "rerank": 0.89, "qa": 0.5},
], {"high": {("p", 2)}, "low": {("q", 2)}})


def test_a_weak_chunk_is_kept_when_a_strong_one_opened_the_gate():
    assert kept(QUESTIONS[0], 0.9, 0.5, 0.0) == {("p", 1), ("p", 2)}


def test_many_mediocre_candidates_never_open_the_gate():
    assert kept(QUESTIONS[1], 0.9, 0.5, 0.0) == set()


def test_the_reader_still_vetoes_inside_an_answered_question():
    rows = [*ROWS, {"question_id": "a", "chunk": ["p", 5], "rerank": 0.95, "qa": 0.0}]
    question = group(rows, {**GOLD, "a": {("p", 1)}})[0]
    assert kept(question, 0.9, 0.5, 0.01) == {("p", 1), ("p", 2)}


def test_the_gate_keeps_the_silence_a_single_low_cut_would_lose():
    """keep 0.5 alone would highlight for x; the gate reads only its best candidate."""
    assert silence(NEGATIVES, 0.9, 0.5, 0.0) == {"silent_offtopic": 0.5}
    assert silence(NEGATIVES, 0.99, 0.5, 0.0) == {"silent_offtopic": 1.0}


def test_a_question_left_unanswered_scores_zero_and_is_flagged_empty():
    [_, b] = question_scores(QUESTIONS, 0.9, 0.5, 0.0)
    assert b["empty"] and b["evidence_f1"] == 0.0


def test_evaluate_reports_both_sides_of_the_objective():
    row = evaluate(QUESTIONS, 0.9, 0.5, 0.0, NEGATIVES)
    assert row["evidence_f1"] == pytest.approx(0.5)  # a exact, b empty
    assert row["empty"] == pytest.approx(0.5)
    assert row["silent_offtopic"] == pytest.approx(0.5)


def test_a_margin_reads_each_question_at_its_own_height():
    """One absolute cut cannot keep both runners-up; a 0.02 margin keeps both."""
    high, low = CROWDED
    assert kept(high, 0.9, 0.02, 0.0, relative=True) == {("p", 1), ("p", 2)}
    assert kept(low, 0.9, 0.02, 0.0, relative=True) == {("q", 1), ("q", 2)}
    # the absolute rule that keeps the low question's gold also lets in nothing extra
    # up high, and the one tuned up high drops the low question's gold entirely
    assert kept(low, 0.9, 0.93, 0.0) == set()


def test_a_margin_of_zero_keeps_only_the_question_s_best():
    assert kept(CROWDED[0], 0.9, 0.0, 0.0, relative=True) == {("p", 1)}


def test_a_margin_never_answers_a_question_the_gate_refused():
    assert kept(QUESTIONS[1], 0.9, 0.2, 0.0, relative=True) == set()
    assert silence(NEGATIVES, 0.99, 0.2, 0.0, relative=True) == {"silent_offtopic": 1.0}


def test_grid_search_never_keeps_below_a_gate_it_could_not_open():
    grid = grid_search(QUESTIONS, NEGATIVES)
    assert all(row["keep"] <= row["gate"] for row in grid)
    assert grid[0]["evidence_f1"] >= grid[-1]["evidence_f1"]


def test_the_relative_grid_searches_margins_not_scores():
    grid = grid_search(CROWDED, NEGATIVES, relative=True)
    assert {row["keep"] for row in grid} == set(MARGIN_GRID)


def test_common_best_only_considers_triples_silent_enough_on_every_set():
    grids = {"x": [{"gate": 0.9, "keep": 0.5, "qa": 0.0, "evidence_f1": 0.9,
                    "silent_offtopic": 0.5},
                   {"gate": 0.95, "keep": 0.5, "qa": 0.0, "evidence_f1": 0.4,
                    "silent_offtopic": 1.0}],
             "y": [{"gate": 0.9, "keep": 0.5, "qa": 0.0, "evidence_f1": 0.9,
                    "silent_offtopic": 1.0},
                   {"gate": 0.95, "keep": 0.5, "qa": 0.0, "evidence_f1": 0.4,
                    "silent_offtopic": 1.0}]}
    assert common_best(grids, target=0.9) == (0.95, 0.5, 0.0)


def test_report_compares_both_families_against_the_shipped_pair():
    text = report({"s1": QUESTIONS, "s2": CROWDED}, {"s1": NEGATIVES})
    for part in ("## s1, absolute: best 10", "## s1, relative: best 10",
                 "chosen against shipped", "Δ on s1", "silent_offtopic"):
        assert part in text
