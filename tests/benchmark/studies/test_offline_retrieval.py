"""The offline replica of the index's search."""

import json
import math

import numpy as np

from benchmark.studies.offline_retrieval import (
    bm25,
    gold_ids,
    idf_table,
    inverted_index,
    load_corpus,
    retrieve,
    rrf,
    top,
)


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


def test_top_breaks_ties_by_index():
    assert top(np.array([0.5, 0.9, 0.5, 0.1]), limit=3) == [(1, 0.9), (0, 0.5), (2, 0.5)]


def test_rrf_sums_reciprocal_ranks():
    fused = rrf([[3, 1], [1, 2]])
    # k=2, rank from zero: 1 is 2nd and 1st, 1/3 + 1/2, ahead of 3 (1/2) and 2 (1/3)
    assert [item for item, _ in fused] == [1, 3, 2]
    assert math.isclose(fused[0][1], 1 / 3 + 1 / 2)


def test_rrf_matches_the_index_fusion():
    """First in both branches scores 1.0, as Qdrant and QdrantManager.rrf_fuse do."""
    assert rrf([[7], [7]]) == [(7, 1.0)]


def test_rrf_breaks_ties_by_id():
    """Equal ranks in different lists give equal scores; the order must not depend on input order."""
    assert [item for item, _ in rrf([[3], [2]])] == [2, 3]
    assert [item for item, _ in rrf([[2], [3]])] == [2, 3]


def test_idf_is_qdrants_bm25_idf():
    docs = [{"indices": [7, 8], "values": [1.0, 1.0]}, {"indices": [7], "values": [1.0]}]
    idf = idf_table(docs)
    assert math.isclose(idf[7], math.log((2 - 2 + 0.5) / (2 + 0.5) + 1))
    assert math.isclose(idf[8], math.log((2 - 1 + 0.5) / (1 + 0.5) + 1))


def test_bm25_scores_only_the_documents_that_share_a_token():
    docs = [{"indices": [7], "values": [2.0]}, {"indices": [8], "values": [1.0]},
            {"indices": [9], "values": [1.0]}]
    idf = idf_table(docs)
    scores = bm25({"indices": [7, 8], "values": [1.0, 1.0]}, inverted_index(docs), idf, 3)
    assert scores[0] == idf[7] * 2.0
    assert scores[1] == idf[8]
    assert scores[2] == 0.0


def test_sparse_retrieval_never_returns_a_chunk_without_a_shared_token():
    dense = np.array([0.9, 0.8, 0.7])
    sparse = np.array([0.0, 2.0, 0.0])
    assert retrieve("sparse", dense, sparse) == [(1, 2.0)]
    assert [i for i, _ in retrieve("dense", dense, sparse)] == [0, 1, 2]
    # chunk 1 is in both branches, so the fusion puts it first
    assert retrieve("hybrid", dense, sparse)[0][0] == 1
