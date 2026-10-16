import numpy as np
import pytest

from benchmark.studies.scalability import (
    corpus_order,
    growth_exponent,
    library_sizes,
    placement,
    replica_vectors,
    summary,
)


def test_library_grows_by_ten_up_to_the_cap():
    assert library_sizes(1, 1000) == [1, 10, 100, 1000]
    assert library_sizes(5, 4999) == [5, 50, 500]


def test_the_questions_papers_enter_the_library_first():
    order = corpus_order(["a", "b", "c", "d"], ["c", "x", "c", "a"], seed=0)
    assert order[:2] == ["c", "a"]  # x is not in the corpus, c is not repeated
    assert sorted(order) == ["a", "b", "c", "d"]


def test_positions_past_the_real_papers_are_copies_under_a_new_hash():
    order = ["a", "b"]
    assert placement(1, order) == ("b", "b", 0)
    assert placement(2, order) == ("a", "a~1", 1)
    assert placement(5, order) == ("b", "b~2", 2)


def test_copies_are_unit_vectors_near_the_original_and_reproducible():
    dense = np.eye(3, 768)
    assert replica_vectors(dense, 0, seed=1) is dense
    copy = replica_vectors(dense, 1, seed=1)
    assert np.allclose(np.linalg.norm(copy, axis=1), 1.0)
    assert not np.allclose(copy, dense)
    assert np.all(np.sum(copy * dense, axis=1) > 0.95)
    assert np.array_equal(copy, replica_vectors(dense, 1, seed=1))
    assert not np.array_equal(copy, replica_vectors(dense, 2, seed=1))


def test_growth_exponent_reads_linear_and_constant_costs():
    sizes = [10, 100, 1000, 10000]
    assert growth_exponent(sizes, [s * 1e-6 for s in sizes]) == pytest.approx(1.0)
    assert growth_exponent(sizes, [0.5] * 4) == pytest.approx(0.0, abs=1e-9)


def test_summary_has_a_row_per_size_and_mode_and_fits_each_mode():
    rows = [{"docs": d, "chunks": 50 * d, "synthetic": d > 100, "mode": m, "question_id": q,
             "search_s": 1e-5 * d, "retrieval_s": 0.1, "rerank_s": 1.0, "read_s": 1.0,
             "total_s": 2.1, "reached_reader": q == "q1"}
            for d in (1, 10, 1000) for m in ("exact", "hnsw") for q in ("q1", "q2")]
    index_rows = [{"docs": 1, "chunks": 50, "upload_s": 0.2, "optimize_s": 0.0}]
    text = summary(rows, index_rows)
    assert "| 1000 | 50000 | yes | hnsw |" in text
    assert "| exact | 1.00 | 3 |" in text
    assert "50%" in text and "## Index build" in text
