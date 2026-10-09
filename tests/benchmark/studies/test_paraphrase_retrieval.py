

from benchmark.studies.paraphrase_retrieval import (
    DEEP,
    KS,
    faithful_variants,
    parse_rewrites,
    report,
    shipped_variants,
    union_by_score,
)


def test_union_keeps_the_score_of_the_first_list_to_find_a_chunk():
    """The pipeline merges variants that way: the original question has priority."""
    merged = union_by_score([[(1, 0.5), (2, 0.4)], [(2, 0.9), (3, 0.45)]])
    assert merged == [1, 3, 2]


def test_shipped_variants_keeps_the_least_similar_candidates_that_differ_from_each_other():
    question = "what is the baseline model"
    candidates = [
        "paraphrase: What is the baseline model",        # the question itself
        "what is the baseline",                          # 0.80 from the question
        "which reference system do they compare against",
        "which reference system do they compare with",   # 0.75 from the one above
        "what model serves as the reference",
    ]
    assert shipped_variants(question, candidates) == [
        question, "which reference system do they compare against",
        "what model serves as the reference"]
    assert shipped_variants(question, candidates, n=1) == [
        question, "which reference system do they compare against"]


def test_faithful_variants_keeps_the_closest_distinct_candidates():
    sims = {"What is the baseline model?": 0.95, "What is the starting point of a word?": 0.40,
            "Which baseline do they use?": 0.90}
    chosen = faithful_variants(
        "What is the baseline?",
        ["What is the starting point of a word?", "what is the  baseline?",
         "What is the baseline model?", "Which baseline do they use?",
         "What is the baseline model?"],
        lambda texts: [sims[t] for t in texts])
    # the question itself (up to case and spacing) and repeats are not variants
    assert chosen == ["What is the baseline model?", "Which baseline do they use?"]


def test_faithful_variants_with_nothing_new_returns_nothing():
    assert faithful_variants("Why?", ["why?", " Why? "], lambda texts: [1.0] * len(texts)) == []


def test_parse_rewrites_strips_numbering_quotes_and_preamble():
    raw = ("Here are two rewrites:\n"
           "1. \"What evaluation metric is used?\"\n"
           "2) Which evaluation metric do the authors use?\n")
    assert parse_rewrites(raw, "What evaluation metric do they use?") == [
        "What evaluation metric is used?", "Which evaluation metric do the authors use?"]


def test_parse_rewrites_drops_the_question_repeats_and_extras():
    raw = ("- What evaluation metric do they use?\n- Which metric?\n- which  metric?\n"
           "- What metric is reported?\n- Another one?")
    assert parse_rewrites(raw, "What evaluation metric do they use?", n=2) == [
        "Which metric?", "What metric is reported?"]


def test_report_renders_a_table_per_mode_with_paired_deltas():
    row = {k: 1.0 for k in KS}
    rows = {(mode, arm): [row, row] for mode in ("dense", "sparse", "hybrid")
            for arm in ("original", "shipped+union", "faithful+rrf")}
    text = report(rows, {"shipped": [2, 1], "faithful": [2, 2]})
    for section in ("## dense", "## sparse", "## hybrid", "faithful+rrf", "shipped+union"):
        assert section in text
    assert f"recall@{DEEP}" in text
    assert "+0.000 [+0.000, +0.000]" in text
