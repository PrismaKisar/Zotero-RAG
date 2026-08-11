import pytest

from benchmark.retrieval_metrics import (
    aggregate,
    bootstrap_ci,
    evidence_prf,
    per_question_scores,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)


def test_recall_at_k_counts_fraction_of_gold_found():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0


def test_recall_at_k_truncates_to_k():
    assert recall_at_k(["a", "b", "c"], {"c"}, k=1) == 0.0


def test_recall_at_k_rejects_empty_gold():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)


def test_precision_at_k_divides_by_what_was_returned():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == 0.5
    assert precision_at_k(["a", "b"], {"a"}, k=1) == 1.0
    # fewer results than k: denominator is the short list, not k
    assert precision_at_k(["a"], {"a"}, k=10) == 1.0
    assert precision_at_k([], {"a"}, k=10) == 0.0


def test_precision_and_recall_disagree_when_returning_extra_chunks():
    """The point of adding precision@k: recall alone cannot see over-returning."""
    ranked, gold = ["a", "x", "y", "z"], {"a"}
    assert recall_at_k(ranked, gold, k=4) == 1.0
    assert precision_at_k(ranked, gold, k=4) == 0.25


def test_reciprocal_rank_of_first_hit():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["a", "x", "b"], {"a"}) == 1.0


def test_reciprocal_rank_zero_when_gold_absent():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_evidence_prf_on_ids():
    perfect = evidence_prf({"a", "b"}, {"a", "b"})
    assert perfect == {"evidence_precision": 1.0, "evidence_recall": 1.0, "evidence_f1": 1.0}

    partial = evidence_prf({"a", "x"}, {"a", "b"})
    assert partial["evidence_precision"] == 0.5
    assert partial["evidence_recall"] == 0.5
    assert partial["evidence_f1"] == 0.5

    assert evidence_prf({"x"}, {"a"})["evidence_f1"] == 0.0
    assert evidence_prf(set(), {"a"})["evidence_f1"] == 0.0


def test_evidence_prf_rewards_precision_unlike_recall_at_k():
    """A run that highlights the whole paper has perfect recall but poor F1."""
    everything = evidence_prf(set("abcdefghij"), {"a"})
    assert everything["evidence_recall"] == 1.0
    assert everything["evidence_precision"] == 0.1
    assert everything["evidence_f1"] < 0.2


def test_evidence_prf_rejects_empty_gold():
    with pytest.raises(ValueError):
        evidence_prf({"a"}, set())


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [0.0, 0.5, 1.0] * 40
    lo, hi = bootstrap_ci(values)
    mean = sum(values) / len(values)
    assert lo < mean < hi
    assert bootstrap_ci(values) == (lo, hi)  # same seed, same interval


def test_bootstrap_ci_narrows_with_more_data():
    small = bootstrap_ci([0.0, 1.0] * 10)
    large = bootstrap_ci([0.0, 1.0] * 500)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_degenerate_cases():
    assert bootstrap_ci([0.7]) == (0.7, 0.7)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_per_question_scores_defaults_predicted_to_top_k():
    scores = per_question_scores(["a", "x"], {"a"}, k=2)
    assert scores["recall@2"] == 1.0
    assert scores["precision@2"] == 0.5
    assert scores["mrr"] == 1.0
    assert scores["evidence_precision"] == 0.5

    # an explicit prediction set decouples attribution from the ranked list
    filtered = per_question_scores(["a", "x"], {"a"}, k=2, predicted_ids={"a"})
    assert filtered["evidence_precision"] == 1.0


def test_aggregate_averages_over_questions():
    per_question = [
        (["a", "b"], {"a"}),   # recall@1 = 1.0, rr = 1.0
        (["b", "a"], {"a"}),   # recall@1 = 0.0, rr = 0.5
    ]
    result = aggregate(per_question, k=1)
    assert result["recall@1"] == 0.5
    assert result["mrr"] == 0.75
    assert result["n_questions"] == 2


def test_aggregate_with_ci_adds_bounds():
    result = aggregate([(["a", "b"], {"a"})] * 20, k=1, with_ci=True)
    assert result["recall@1_ci_low"] <= result["recall@1"] <= result["recall@1_ci_high"]


def test_aggregate_rejects_empty_batch():
    with pytest.raises(ValueError):
        aggregate([], k=1)


def test_summarize_rejects_empty_rows():
    with pytest.raises(ValueError):
        summarize([])
