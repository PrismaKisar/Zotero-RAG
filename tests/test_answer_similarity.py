import json

import numpy as np
import pytest

from benchmark.answer_similarity import (
    best_cosines,
    report,
    rescale,
    unrelated_baseline,
)

# Three orthogonal directions: a text maps to its first word's axis, so two texts
# sharing a first word are identical and any other pair is unrelated.
AXES = {"cat": [1.0, 0.0, 0.0], "dog": [0.0, 1.0, 0.0], "fish": [0.0, 0.0, 1.0]}


def encode(texts):
    return np.array([AXES[t.split()[0]] for t in texts])


def test_the_best_reference_counts_and_an_empty_prediction_scores_zero():
    scores = best_cosines(["cat a", ""], [["dog", "cat b"], ["cat"]], encode)
    assert scores.tolist() == [1.0, 0.0]


def test_the_baseline_never_scores_a_prediction_against_its_own_references():
    assert unrelated_baseline(["cat", "dog"], [["cat"], ["dog"]], encode) == 0.0


def test_the_baseline_needs_two_questions():
    with pytest.raises(ValueError):
        unrelated_baseline(["cat"], [["cat"]], encode)


def test_rescaling_maps_the_baseline_to_zero_and_identity_to_one():
    assert rescale(np.array([0.4, 1.0]), 0.4).tolist() == [0.0, 1.0]


def _dump(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(qid, answer, f1):
    return {"answer_f1": f1, "record": {"question_id": qid, "predicted_answer": answer}}


def test_the_report_puts_similarity_beside_f1_with_paired_deltas(tmp_path):
    gold = {"q1": [{"answer": "cat"}], "q2": [{"answer": "dog"}]}
    _dump(tmp_path / "baseline.jsonl", [_row("q1", "fish", 0.0), _row("q2", "fish", 0.0)])
    _dump(tmp_path / "arm.jsonl", [_row("q1", "cat x", 0.5), _row("q2", "dog y", 0.5)])

    text = report(tmp_path, gold, encode)

    assert "| arm | 2 | 0.500 | 1.000 |" in text
    assert "+1.000" in text and "+0.500" in text


def test_a_dump_without_predicted_answers_is_refused(tmp_path):
    _dump(tmp_path / "baseline.jsonl", [{"answer_f1": 0.0, "record": {"question_id": "q1"}}])
    with pytest.raises(SystemExit, match="predicted_answer"):
        report(tmp_path, {"q1": [{"answer": "cat"}]}, encode)
