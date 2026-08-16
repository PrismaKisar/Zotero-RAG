"""Unit tests for the question-type preset resolver."""

from zotero_rag.question_presets import PRESETS, resolve

LIVE_FIELDS = {
    "retrieval_threshold",
    "rerank_threshold",
    "qa_score_threshold",
    "min_answer_words",
    "section_diversity",
    "retrieval_mode",
    "result_limit",
    "rerank_enabled",
}


def test_presets_carry_only_live_fields():
    expected_types = {
        "factoid", "methodology", "explanation",
        "comparison", "definition", "general", "custom",
    }
    assert set(PRESETS) == expected_types
    for name, preset in PRESETS.items():
        assert set(preset) == LIVE_FIELDS, name


def test_resolve_returns_preset_when_no_overrides():
    assert resolve("factoid") == PRESETS["factoid"]
    assert resolve("methodology", None) == PRESETS["methodology"]


def test_override_precedence():
    resolved = resolve("general", {"qa_score_threshold": 0.99, "min_answer_words": 7})
    assert resolved["qa_score_threshold"] == 0.99
    assert resolved["min_answer_words"] == 7
    # untouched fields keep preset defaults
    assert resolved["retrieval_threshold"] == PRESETS["general"]["retrieval_threshold"]


def test_unknown_type_falls_back_to_general():
    assert resolve("does-not-exist") == PRESETS["general"]
    assert resolve("does-not-exist", {"min_answer_words": 1})["min_answer_words"] == 1


def test_qa_score_threshold_is_literal():
    # no per-type transform: the resolved value equals the preset default...
    for name, preset in PRESETS.items():
        assert resolve(name)["qa_score_threshold"] == preset["qa_score_threshold"]
    # ...and an override passes through untouched (no max()/scaling).
    assert resolve("methodology", {"qa_score_threshold": 0.0})["qa_score_threshold"] == 0.0
    assert resolve("factoid", {"qa_score_threshold": 0.01})["qa_score_threshold"] == 0.01


def test_resolve_does_not_mutate_preset():
    resolve("general", {"min_answer_words": 42})
    assert PRESETS["general"]["min_answer_words"] == 3
