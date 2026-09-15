import numpy as np
import torch

from benchmark.retrieval_models import (
    KS,
    REFERENCE,
    arm_rankings,
    ranking_scores,
    report,
    sparse_rows,
)


def test_sparse_rows_split_a_batch_into_one_dict_per_row():
    indices = torch.tensor([[0, 0, 1], [3, 7, 2]])
    batch = torch.sparse_coo_tensor(indices, torch.tensor([0.5, 1.5, 2.0]), (2, 10),
                                    check_invariants=True)
    assert sparse_rows(batch) == [{"indices": [3, 7], "values": [0.5, 1.5]},
                                  {"indices": [2], "values": [2.0]}]


def test_every_pair_gets_a_hybrid_arm_and_each_model_its_own():
    ids = ["a", "b", "c"]
    rankings = arm_rankings({"d": np.array([0.9, 0.1, 0.5])},
                            {"s1": np.array([0.0, 2.0, 0.0]), "s2": np.array([1.0, 0.0, 0.0])},
                            ids)
    assert set(rankings) == {"dense:d", "sparse:s1", "sparse:s2", "hybrid:d+s1", "hybrid:d+s2"}
    assert rankings["dense:d"] == ["a", "c", "b"]
    # a sparse search only returns chunks sharing a token with the query
    assert rankings["sparse:s1"] == ["b"]
    # a is first in both branches of d+s2, so the fusion puts it first
    assert rankings["hybrid:d+s2"][0] == "a"


def test_ranking_scores_are_recall_at_every_k_and_mrr():
    row = ranking_scores(["x", "g"], {"g"})
    assert row["recall@1"] == 0.0 and row["recall@3"] == 1.0 and row["mrr"] == 0.5


def test_report_puts_the_shipped_hybrid_as_reference_in_both_scopes():
    row = {**{f"recall@{k}": 0.5 for k in KS}, "mrr": 0.5}
    better = {**row, "recall@10": 1.0}
    rows = {(scope, arm): [value, value] for scope in ("library", "paper")
            for arm, value in ((REFERENCE, row), ("dense:new", better))}
    text = report("set", rows, 2, [["dense", "new", 10, "5", "2", "1.0"]])
    for part in ("## scope: library", "## scope: paper", "+0.500 [+0.500, +0.500]",
                 "## Encoding cost", "chunks/s"):
        assert part in text
    # sorted by recall@10: the better arm comes before the reference
    library = text.split("## scope: paper")[0]
    assert library.index("| dense:new |") < library.index(f"| {REFERENCE} |")
