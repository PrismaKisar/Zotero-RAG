import pytest

from benchmark.stratify import (
    MIN_STRATUM_N,
    evidence_count_stratum,
    evidence_spread_stratum,
    stratify,
    to_markdown,
)


def record(question="What is the model?", n_evidence=1, chunk_indices=(0,)):
    return {
        "question": question,
        "evidence": ["e"] * n_evidence,
        "aligned_chunks": {"e": [{"chunk_index": i, "overlap": 1.0} for i in chunk_indices]},
    }


def test_evidence_count_stratum():
    assert evidence_count_stratum(record(n_evidence=1)) == "single-evidence"
    assert evidence_count_stratum(record(n_evidence=3)) == "multi-evidence"


def test_evidence_spread_stratum():
    assert evidence_spread_stratum(record(chunk_indices=(4,))) == "single-chunk"
    assert evidence_spread_stratum(record(chunk_indices=(4, 5))) == "adjacent"
    assert evidence_spread_stratum(record(chunk_indices=(2, 60))) == "scattered"


def test_stratify_splits_by_question_form():
    rows = ([{"record": record("Why does it fail?"), "recall@10": 0.0}] * 3
            + [{"record": record("What is the model?"), "recall@10": 1.0}] * 3)
    out = stratify(rows)

    assert out["overall"]["recall@10"] == 0.5
    assert out["overall"]["n_questions"] == 6
    assert out["question_form"]["why (causal)"]["recall@10"] == 0.0
    assert out["question_form"]["what/which (factual)"]["recall@10"] == 1.0


def test_stratify_flags_underpowered_strata():
    rows = [{"record": record(), "recall@10": 1.0}] * 3
    out = stratify(rows)
    assert out["question_form"]["what/which (factual)"]["underpowered"] is True

    big = [{"record": record(), "recall@10": 1.0}] * (MIN_STRATUM_N + 1)
    assert stratify(big)["question_form"]["what/which (factual)"]["underpowered"] is False


def test_stratify_attaches_confidence_intervals():
    rows = [{"record": record(), "recall@10": float(i % 2)} for i in range(40)]
    overall = stratify(rows)["overall"]
    assert overall["recall@10_ci_low"] <= overall["recall@10"] <= overall["recall@10_ci_high"]
    assert "record" not in overall


def test_stratify_rejects_empty_rows():
    with pytest.raises(ValueError):
        stratify([])


def test_to_markdown_renders_overall_and_strata():
    rows = [{"record": record("Why does it fail?"), "recall@10": 0.0},
            {"record": record("What is the model?"), "recall@10": 1.0}]
    md = to_markdown({"QASPER": stratify(rows)}, ["recall@10"])

    assert "## recall@10" in md
    assert "| **overall** |" in md
    assert "question_form: why (causal)" in md
    assert "†" in md  # both strata are tiny here


def test_to_markdown_handles_metric_missing_from_one_dataset():
    a = stratify([{"record": record(), "recall@10": 1.0}])
    b = stratify([{"record": record(), "answer_f1": 0.5}])
    md = to_markdown({"QASPER": a, "QASA": b}, ["recall@10"])
    assert "n/a" in md
