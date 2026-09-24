"""The pipeline's single fixed configuration."""

from zotero_rag.settings import SETTINGS, resolve


def test_settings_carry_the_measured_values():
    assert SETTINGS["rerank_gate"] == 0.88
    assert SETTINGS["rerank_margin"] == 0.02
    assert SETTINGS["qa_score_threshold"] == 0.0
    assert SETTINGS["retrieval_mode"] == "hybrid"
    assert SETTINGS["section_diversity_enabled"] is False


def test_resolve_without_overrides_is_the_settings():
    assert resolve() == SETTINGS
    assert resolve(None) == SETTINGS


def test_overrides_win_and_leave_the_settings_untouched():
    resolved = resolve({"result_limit": 7})
    assert resolved["result_limit"] == 7
    assert resolved["rerank_gate"] == SETTINGS["rerank_gate"]
    assert SETTINGS["result_limit"] == 30
