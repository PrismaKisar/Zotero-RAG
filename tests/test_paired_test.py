"""Unit tests for benchmark/paired_test.py."""

import json

import pytest

from benchmark.ablation import config_label
from benchmark.paired_test import (
    compare,
    compare_all,
    load_per_question,
    paired_deltas,
    to_markdown,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def row(qid, **metrics):
    return {**metrics, "record": {"question_id": qid}}


def test_load_per_question_keys_on_question_id(tmp_path):
    path = tmp_path / "baseline.jsonl"
    write_jsonl(path, [row("q1", answer_f1=0.5), row("q2", answer_f1=0.25)])
    assert load_per_question(path) == {
        "q1": {"answer_f1": 0.5, "record": {"question_id": "q1"}},
        "q2": {"answer_f1": 0.25, "record": {"question_id": "q2"}},
    }


def test_paired_deltas_pairs_by_question_not_by_position():
    baseline = {"q1": row("q1", answer_f1=0.1), "q2": row("q2", answer_f1=0.9)}
    other = {"q2": row("q2", answer_f1=1.0), "q1": row("q1", answer_f1=0.3)}
    assert paired_deltas(baseline, other, "answer_f1") == pytest.approx([0.2, 0.1])


def test_paired_deltas_uses_only_questions_both_configs_scored():
    """A config that produced no row for a question must not shift the mean."""
    baseline = {"q1": row("q1", answer_f1=0.4), "q2": row("q2", answer_f1=1.0)}
    other = {"q1": row("q1", answer_f1=0.6)}
    assert paired_deltas(baseline, other, "answer_f1") == [pytest.approx(0.2)]


def test_compare_flags_a_consistent_shift_as_significant():
    baseline = {f"q{i}": row(f"q{i}", answer_f1=0.2) for i in range(30)}
    other = {f"q{i}": row(f"q{i}", answer_f1=0.5) for i in range(30)}
    result = compare(baseline, other, "answer_f1")
    assert result["delta"] == pytest.approx(0.3)
    assert result["significant"]


def test_compare_does_not_flag_a_zero_centred_difference():
    """Alternating +0.4/-0.4 averages to zero, so the CI must straddle it."""
    baseline = {f"q{i}": row(f"q{i}", answer_f1=0.5) for i in range(40)}
    other = {f"q{i}": row(f"q{i}", answer_f1=0.5 + (0.4 if i % 2 else -0.4))
             for i in range(40)}
    result = compare(baseline, other, "answer_f1")
    assert result["delta"] == pytest.approx(0.0)
    assert not result["significant"]


def test_compare_returns_none_for_a_metric_the_config_lacks():
    """oracle_context writes no retrieval metrics; that must not raise."""
    baseline = {"q1": row("q1", answer_f1=0.5, evidence_f1=0.2)}
    other = {"q1": row("q1", answer_f1=0.7)}
    assert compare(baseline, other, "evidence_f1") is None


def test_compare_all_skips_baseline_and_covers_every_other_config(tmp_path):
    write_jsonl(tmp_path / "baseline.jsonl", [row("q1", answer_f1=0.2)])
    write_jsonl(tmp_path / "variant_a.jsonl", [row("q1", answer_f1=0.4)])
    write_jsonl(tmp_path / "variant_b.jsonl", [row("q1", answer_f1=0.1)])

    rows = compare_all(tmp_path, metrics=("answer_f1",))

    assert {r["config"] for r in rows} == {"variant_a", "variant_b"}
    assert [r["delta"] for r in rows] == [pytest.approx(0.2), pytest.approx(-0.1)]


def test_compare_all_rejects_a_directory_without_a_baseline(tmp_path):
    write_jsonl(tmp_path / "variant_a.jsonl", [row("q1", answer_f1=0.4)])
    with pytest.raises(SystemExit):
        compare_all(tmp_path, metrics=("answer_f1",))


def test_to_markdown_lists_significant_rows_first():
    rows = [
        {"config": "noise", "metric": "answer_f1", "n": 5, "delta": 0.9,
         "ci_low": -0.1, "ci_high": 1.9, "significant": False},
        {"config": "real", "metric": "answer_f1", "n": 5, "delta": 0.1,
         "ci_low": 0.05, "ci_high": 0.15, "significant": True},
    ]
    lines = to_markdown(rows).splitlines()
    assert "real" in lines[2]
    assert "noise" in lines[3]


def test_config_label_keeps_a_slashed_model_id_in_one_file():
    """A raw slash would make the dump a subdirectory instead of a file."""
    assert config_label("qa_model", "deepset/roberta-base-squad2") == \
        "qa_model_deepset_roberta-base-squad2"


def test_config_label_neutralises_parent_directory_traversal():
    assert ".." not in config_label("qa_model", "../../etc/passwd")


def test_config_label_of_a_reference_row_has_no_value_suffix():
    assert config_label("baseline", None) == "baseline"
