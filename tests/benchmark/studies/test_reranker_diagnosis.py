from benchmark.studies.reranker_diagnosis import (
    SCOPES,
    disagreements,
    listwise_scores,
    order_scores,
    ranked_orders,
    render_disagreements,
    report,
)


def test_listwise_results_are_put_back_in_input_order():
    results = [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4},
               {"index": 1, "relevance_score": 0.1}]
    assert listwise_scores(results, 3) == [0.4, 0.1, 0.9]


def test_ranked_orders_share_the_candidates_and_keep_retrieval_order_on_ties():
    hybrid = ["a", "b", "c"]
    orders = ranked_orders(hybrid, {"m": [0.1, 0.9, 0.1]})
    assert orders["hybrid"] == ["a", "b", "c"]
    # b scores highest; a and c tie, so they keep retrieval's order
    assert orders["m"] == ["b", "a", "c"]
    assert sorted(orders["m+rrf"]) == sorted(hybrid)


def test_fusion_puts_first_what_both_orders_rank_high():
    orders = ranked_orders(["a", "b", "c"], {"m": [0.2, 0.9, 0.8]})
    # b is 2nd and 1st, ahead of a (1st and 3rd) and c (3rd and 2nd)
    assert orders["m+rrf"][0] == "b"


def test_order_scores_are_recall_at_every_k_and_mrr():
    row = order_scores(["x", "g", "y"], {"g"})
    assert row["recall@1"] == 0.0 and row["recall@3"] == 1.0
    assert row["mrr"] == 0.5


def _case(qid, hybrid, model):
    return {"question_id": qid, "question": "q?", "evidence": ["ev"], "gold": {"g"},
            "orders": {"hybrid": hybrid, "m": model}, "texts": {c: c for c in "gxy"}}


def test_disagreements_keep_only_questions_where_one_order_puts_gold_first():
    cases = [_case("both", ["g", "x"], ["g", "x"]), _case("neither", ["x", "g"], ["y", "g"]),
             _case("hybrid", ["g", "x"], ["x", "g"]), _case("model", ["x", "g"], ["g", "x"])]
    split = disagreements(cases, "m")
    assert split["counts"] == {"hybrid": 1, "m": 1}
    assert [c["question_id"] for c in split["hybrid"]] == ["hybrid"]
    assert [c["question_id"] for c in split["m"]] == ["model"]
    text = render_disagreements("set", cases, ["m"])
    assert "gold first only for hybrid 1, only for the model 1" in text
    assert "(not gold)" in text


def test_report_renders_a_table_per_scope_with_paired_deltas():
    row = {**{f"recall@{k}": 1.0 for k in (1, 3, 5, 10)}, "mrr": 1.0}
    rows = {(s, o): [row, row] for s in SCOPES for o in ("hybrid", "org/m", "org/m+rrf")}
    judged = {s: {"judged": 2, "total": 3} for s in SCOPES}
    text = report("set", rows, judged, ["org/m"])
    for part in ("## scope: library", "## scope: paper", "| m |", "| m+rrf |",
                 "2 of 3 questions", "+0.000 [+0.000, +0.000]"):
        assert part in text
