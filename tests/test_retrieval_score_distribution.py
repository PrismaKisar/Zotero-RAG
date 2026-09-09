import json

import numpy as np

from benchmark.retrieval_score_distribution import (
    gold_ids,
    load_corpus,
    question_scores,
    report,
    sweep,
)


def _question(top, top_gold, gold, noise=(0.1,)):
    return {"top": np.array(top), "top_gold": np.array(top_gold),
            "gold": np.array(gold), "noise": np.array(noise)}


def test_load_corpus_ids_every_chunk_by_paper_and_position(tmp_path):
    (tmp_path / "b.json").write_text(json.dumps(["b0"]))
    (tmp_path / "a.json").write_text(json.dumps(["a0", "a1"]))
    ids, texts = load_corpus(tmp_path)
    assert ids == [("a", 0), ("a", 1), ("b", 0)]
    assert texts == ["a0", "a1", "b0"]


def test_gold_ids_collects_every_aligned_chunk():
    record = {"paper_id": "p", "aligned_chunks": {
        "e1": [{"chunk_index": 3}], "e2": [{"chunk_index": 5}, {"chunk_index": 3}]}}
    assert gold_ids(record) == {("p", 3), ("p", 5)}


def test_question_scores_keeps_gold_that_ranks_outside_the_limit():
    scores = np.array([0.5, 0.9, 0.7, 0.2])
    is_gold = np.array([False, False, False, True])
    q = question_scores(scores, is_gold, limit=2)
    assert q["top"].tolist() == [0.9, 0.7]
    assert q["top_gold"].tolist() == [False, False]
    assert q["gold"].tolist() == [0.2]
    assert sorted(q["noise"].tolist()) == [0.5, 0.7, 0.9]


def test_a_threshold_below_every_score_is_a_no_op():
    """The failure being measured: the shipped 0.45 against scores of 0.6+."""
    questions = [_question([0.8, 0.7, 0.6], [False, True, False], [0.7])]
    row = sweep(questions, [0.45])[0]
    assert row == {"threshold": 0.45, "gold_kept": 1.0, "noise_kept": 1.0,
                   "survivors": 3.0, "empty": 0.0, "recall": 1.0}


def test_absolute_threshold_cuts_and_can_empty_a_question():
    questions = [_question([0.8, 0.7, 0.6], [False, True, False], [0.7]),
                 _question([0.65, 0.6], [True, False], [0.65])]
    row = sweep(questions, [0.75])[0]
    assert row["gold_kept"] == 0.0
    assert row["noise_kept"] == 1 / 3
    assert row["survivors"] == 0.5
    assert row["empty"] == 0.5
    assert row["recall"] == 0.0


def test_gold_outside_the_limit_costs_retention_but_never_counts_as_recall():
    questions = [_question([0.9, 0.8], [False, False], [0.85])]
    row = sweep(questions, [0.5])[0]
    assert row["gold_kept"] == 1.0
    assert row["recall"] == 0.0


def test_relative_threshold_at_one_keeps_only_the_best_candidate():
    questions = [_question([0.8, 0.7], [False, True], [0.7]),
                 _question([0.6, 0.59], [True, False], [0.6])]
    row = sweep(questions, [1.0], relative=True)[0]
    assert row["survivors"] == 1.0
    assert row["empty"] == 0.0
    assert row["recall"] == 0.5
    # 0.59 / 0.6 is within 2% of the top, 0.7 / 0.8 is not
    assert sweep(questions, [0.98], relative=True)[0]["survivors"] == 1.5


def test_report_renders_every_section():
    questions = [_question([0.8, 0.7, 0.6], [False, True, False], [0.7], [0.8, 0.6, 0.3]),
                 _question([0.75, 0.62], [True, False], [0.75], [0.62, 0.2])]
    text = report(questions, limit=3)
    for section in ("## Percentiles", "## Histogram", "## Absolute threshold",
                    "## Relative threshold"):
        assert section in text
    assert "2 of 2 gold chunks" in text
