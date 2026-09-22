import types

from benchmark.rerank_score_distribution import (
    join_qa,
    joint_sweep,
    report,
    reranker_scores,
    top_k_sweep,
    with_ranks,
)

ROWS = [
    {"question_id": "a", "chunk": ["p", 1], "gold": True, "rerank": 0.9},
    {"question_id": "a", "chunk": ["p", 2], "gold": False, "rerank": 0.4},
    {"question_id": "b", "chunk": ["q", 1], "gold": True, "rerank": 0.2},
    {"question_id": "b", "chunk": ["q", 2], "gold": False, "rerank": 0.05},
]
QA = [
    {"question_id": "a", "chunk": ["p", 1], "with_null": 0.6},
    {"question_id": "a", "chunk": ["p", 2], "with_null": 0.0},
    {"question_id": "b", "chunk": ["q", 1], "with_null": 0.3},
]


def test_join_attaches_the_reader_score_and_drops_candidates_without_one():
    joined = join_qa(ROWS, QA)
    assert [r["qa"] for r in joined] == [0.6, 0.0, 0.3]
    assert ["q", 2] not in [r["chunk"] for r in joined]


def test_joint_sweep_keeps_a_candidate_only_if_it_clears_both_cuts():
    joined = join_qa(ROWS, QA)
    [loose, strict] = joint_sweep(joined, rerank_ts=(0.1,), qa_ts=(0.0, 0.5))
    # rerank 0.1 keeps a1, a2, b1; qa 0.0 keeps all of them
    assert (loose["rerank"], loose["qa"]) == (0.1, 0.0)
    assert loose["kept_per_question"] == 1.5 and loose["gold_kept"] == 1.0
    # qa 0.5 then keeps only a1: b is left with nothing
    assert strict["kept_per_question"] == 0.5
    assert strict["questions_empty"] == 0.5
    assert strict["precision"] == 1.0


def test_reranker_scores_come_back_in_input_order():
    class BestFirst:
        def rerank(self, question, candidates):
            scored = [types.SimpleNamespace(chunk=c, rerank_score=len(c.text) / 10)
                      for c, _ in candidates]
            return sorted(scored, key=lambda s: -s.rerank_score)

    assert reranker_scores(BestFirst(), "q", ["ab", "abcd", "a"]) == [0.2, 0.4, 0.1]


def test_ranks_are_per_question_best_first():
    ranks = {(r["question_id"], r["chunk"][1]): r["rank"] for r in with_ranks(ROWS)}
    assert ranks == {("a", 1): 1, ("a", 2): 2, ("b", 1): 1, ("b", 2): 2}


def test_top_k_keeps_the_best_of_every_question_whatever_its_score():
    [top1] = top_k_sweep(join_qa(ROWS, QA), ks=(1,), qa_ts=(0.0,))
    # b's best scores 0.2, below any absolute cut that would keep a's, and is kept
    assert top1["kept_per_question"] == 1.0
    assert top1["gold_kept"] == 1.0 and top1["precision"] == 1.0


def test_report_renders_the_joint_sweep_only_when_reader_scores_exist():
    alone = report("set", "org/model", ROWS, [])
    assert "## Sweep: rerank threshold alone" in alone
    assert "Joint sweep" not in alone
    assert "`org/model`, 2 questions" in alone
    both = report("set", "org/model", ROWS, join_qa(ROWS, QA))
    assert "## Joint sweep" in both and "3 of 4 candidates" in both
