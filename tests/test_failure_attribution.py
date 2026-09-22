from benchmark.failure_attribution import attribute, report, shares, stage
from benchmark.threshold_search import group

# Every question's best candidate is 0.95 unless stated, so the gate is open and
# the margin is what acts; "closed" is the one that never reaches the gate.
ROWS = [
    # gold never retrieved: only a non-gold candidate, and a strong one
    {"question_id": "miss", "chunk": ["p", 1], "rerank": 0.95, "qa": 0.9},
    # nothing reaches the gate: the question is answered with silence
    {"question_id": "closed", "chunk": ["q", 1], "rerank": 0.80, "qa": 0.9},
    # gold retrieved and the gate is open, but gold sits far below the best
    {"question_id": "margin", "chunk": ["r", 1], "rerank": 0.95, "qa": 0.9},
    {"question_id": "margin", "chunk": ["r", 2], "rerank": 0.80, "qa": 0.9},
    # gold is within the margin, but the reader scores its span at zero
    {"question_id": "reader", "chunk": ["s", 1], "rerank": 0.95, "qa": 0.0},
    # gold clears everything
    {"question_id": "kept", "chunk": ["t", 1], "rerank": 0.95, "qa": 0.9},
]
GOLD = {"miss": {("p", 9)}, "closed": {("q", 1)}, "margin": {("r", 2)},
        "reader": {("s", 1)}, "kept": {("t", 1)}}
QUESTIONS = group(ROWS, GOLD)
CUTS = (0.88, 0.02, 0.05)


def test_each_question_is_blamed_on_the_first_cut_its_gold_fails():
    assert [stage(q, *CUTS) for q in QUESTIONS] == [
        "not retrieved", "question not answered", "cut by the margin",
        "cut by the reader", "kept"]


def test_a_second_gold_chunk_that_survives_rescues_the_question():
    """One gold chunk kept is enough: the question is not a failure."""
    rows = [*ROWS, {"question_id": "margin", "chunk": ["r", 3], "rerank": 0.94, "qa": 0.9}]
    gold = {**GOLD, "margin": {("r", 2), ("r", 3)}}
    assert stage(group(rows, gold)[2], *CUTS) == "kept"


def test_gold_below_an_absolute_cut_survives_when_its_question_scores_low():
    """The point of the relative rule: 0.89 is kept because its question tops at 0.90."""
    rows = [{"question_id": "low", "chunk": ["u", 1], "rerank": 0.90, "qa": 0.9},
            {"question_id": "low", "chunk": ["u", 2], "rerank": 0.89, "qa": 0.9}]
    assert stage(group(rows, {"low": {("u", 2)}})[0], *CUTS) == "kept"


def test_a_cut_question_can_still_highlight_something_that_is_not_gold():
    """The empty share and the failure share are different questions."""
    [_, _, margin_cut, _, _] = attribute(QUESTIONS, *CUTS)
    assert margin_cut == {"stage": "cut by the margin", "empty": False}


def test_a_question_that_never_opened_the_gate_is_counted_empty():
    [_, closed, *_] = attribute(QUESTIONS, *CUTS)
    assert closed == {"stage": "question not answered", "empty": True}


def test_shares_cover_every_stage_even_the_ones_nobody_lands_on():
    rows = [{"stage": "kept", "empty": False}, {"stage": "kept", "empty": False}]
    assert shares(rows) == [["not retrieved", 0, "0.0%"], ["question not answered", 0, "0.0%"],
                            ["cut by the margin", 0, "0.0%"], ["cut by the reader", 0, "0.0%"],
                            ["kept", 2, "100.0%"]]


def test_report_gives_both_tables_per_set():
    text = report({"s1": QUESTIONS, "s2": QUESTIONS}, *CUTS)
    for part in ("## s1: all 5 answerable questions", "## s1: the 2 left empty (40.0%)",
                 "## s2: all 5", "cut by the margin", "cut by the reader"):
        assert part in text
