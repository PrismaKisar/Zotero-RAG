import json

from benchmark.studies.stored_scores import (
    group,
    group_negatives,
    join_qa,
    load_negatives,
)

ROWS = [
    {"question_id": "a", "chunk": ["p", 1], "rerank": 0.9},
    {"question_id": "a", "chunk": ["p", 2], "rerank": 0.4},
    {"question_id": "b", "chunk": ["q", 2], "rerank": 0.05},
]
QA = [
    {"question_id": "a", "chunk": ["p", 1], "with_null": 0.6},
    {"question_id": "a", "chunk": ["p", 2], "with_null": 0.0},
]


def test_join_attaches_the_reader_score_and_drops_candidates_without_one():
    joined = join_qa(ROWS, QA)
    assert [r["qa"] for r in joined] == [0.6, 0.0]
    assert ["q", 2] not in [r["chunk"] for r in joined]


def test_group_keeps_only_questions_with_gold():
    joined = [{**r, "qa": 0.1} for r in ROWS]
    [question] = group(joined, {"a": {("p", 1)}})
    assert question["gold"] == {("p", 1)}
    assert question["candidates"] == [(("p", 1), 0.9, 0.1), (("p", 2), 0.4, 0.1)]


def test_negatives_are_grouped_per_question_with_their_source(tmp_path):
    rows = [{"question_id": "x", "source": "foreign", "chunk": ["p", 1], "rerank": 0.9, "qa": 0.1},
            {"question_id": "x", "source": "foreign", "chunk": ["p", 2], "rerank": 0.2, "qa": 0.0}]
    assert group_negatives(rows) == [{"source": "foreign", "candidates": [
        (("p", 1), 0.9, 0.1), (("p", 2), 0.2, 0.0)]}]
    (tmp_path / "negative_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert load_negatives(tmp_path) == group_negatives(rows)


def test_a_set_never_scored_for_negatives_has_none(tmp_path):
    assert load_negatives(tmp_path) == []
