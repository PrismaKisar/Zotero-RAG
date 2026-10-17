import numpy as np
import pytest

from benchmark.studies.hierarchical_retrieval import (
    build_library,
    paper_documents,
    paper_ranking,
    per_paper,
    report,
    score_question,
)


def test_per_paper_takes_the_best_chunk_or_the_mean_of_the_best_three():
    scores = np.array([0.1, 0.9, 0.5, 0.4, 0.3, 0.2])
    paper_idx = np.array([0, 0, 1, 1, 1, 1])
    assert per_paper(scores, paper_idx, 2, "max").tolist() == [0.9, 0.5]
    assert per_paper(scores, paper_idx, 2, "top3")[1] == pytest.approx(0.4)


def test_a_paper_document_sums_its_chunks_token_weights():
    docs = paper_documents([{"indices": [1, 2], "values": [1.0, 2.0]},
                            {"indices": [2], "values": [3.0]},
                            {"indices": [7], "values": [1.0]}], np.array([0, 0, 1]), 2)
    assert dict(zip(docs[0]["indices"], docs[0]["values"], strict=True)) == {1: 1.0, 2: 5.0}
    assert docs[1] == {"indices": [7], "values": [1.0]}


def _library():
    """Paper a about token 1 and direction x, paper b about token 2 and direction y."""
    x, y = [1.0, 0.0], [0.0, 1.0]
    encoded = {"papers": {
        "a": (np.array([x, x]), [{"indices": [1], "values": [1.0]}] * 2),
        "b": (np.array([y, y, [0.6, 0.8]]), [{"indices": [2], "values": [1.0]}] * 3)}}
    return build_library(encoded, ["a", "b"])


@pytest.mark.parametrize("strategy", ["max", "top3", "centroid", "bm25", "hybrid"])
def test_every_strategy_ranks_first_the_paper_the_query_is_about(strategy):
    library = _library()
    query_vec = np.array([0.0, 1.0])
    ranking = paper_ranking(strategy, library["vectors"] @ query_vec, query_vec,
                            {"indices": [2], "values": [1.0]}, library)
    assert ranking[0] == 1 and sorted(ranking) == [0, 1]


def test_the_second_stage_searches_only_the_kept_papers():
    library = _library()
    record = {"paper_id": "b", "aligned_chunks": {"p": [{"chunk_index": 1}]}}
    row = score_question(record, {"dense": [0.0, 1.0], "sparse": {"indices": [2], "values": [1.0]}},
                         library)
    assert row["oracle"] == 1.0
    assert row[("max", 1, "paper")] == 1.0 and row[("max", 1)] == 1.0


def test_report_lists_flat_oracle_and_every_strategy_and_depth():
    row = {"flat": 0.5, "oracle": 1.0}
    for s in ("max", "top3", "centroid", "bm25", "hybrid"):
        for d in (1, 3, 5, 10):
            row[(s, d)], row[(s, d, "paper")] = 0.75, 1.0
    text = report("set", [row, row], 2)
    assert "| flat (shipped) |" in text and "| oracle paper | 1 |" in text
    assert "| hybrid | 10 | 1.000 | 0.750 | +0.250 [+0.250, +0.250] |" in text
