import pytest

from benchmark.retrieval_metrics import aggregate, recall_at_k, reciprocal_rank


def test_recall_at_k_counts_fraction_of_gold_found():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0


def test_recall_at_k_truncates_to_k():
    assert recall_at_k(["a", "b", "c"], {"c"}, k=1) == 0.0


def test_recall_at_k_rejects_empty_gold():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)


def test_reciprocal_rank_of_first_hit():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["a", "x", "b"], {"a"}) == 1.0


def test_reciprocal_rank_zero_when_gold_absent():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_aggregate_averages_over_questions():
    per_question = [
        (["a", "b"], {"a"}),   # recall@1 = 1.0, rr = 1.0
        (["b", "a"], {"a"}),   # recall@1 = 0.0, rr = 0.5
    ]
    result = aggregate(per_question, k=1)
    assert result == {"recall@1": 0.5, "mrr": 0.75, "n_questions": 2}


def test_aggregate_rejects_empty_batch():
    with pytest.raises(ValueError):
        aggregate([], k=1)
