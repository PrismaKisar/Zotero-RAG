"""Regression guard: the Streamlit UI exposes none of the pipeline's settings.

The thresholds are fixed values measured on the benchmarks (zotero_rag/settings.py),
so a widget for them would let a user undo a calibration they cannot see; the
question types it used to offer carried per-type values nobody had measured. The
checks are static, so they need neither streamlit nor a running app.
"""

import ast
from pathlib import Path

from zotero_rag.settings import SETTINGS

APP = Path(__file__).resolve().parent.parent / "zotero_rag" / "app.py"


def _tree() -> ast.Module:
    return ast.parse(APP.read_text())


def test_app_does_not_import_the_settings():
    assert not any(isinstance(node, ast.ImportFrom)
                   and node.module in ("settings", "question_presets")
                   for node in ast.walk(_tree()))


def test_app_never_names_a_setting():
    source = APP.read_text()
    named = [key for key in SETTINGS if key in source]
    assert not named, f"app.py exposes settings: {named}"


def test_answer_question_gets_neither_overrides_nor_a_question_type():
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "answer_question":
            passed = {keyword.arg for keyword in node.keywords}
            assert not passed & {"overrides", "question_type"}, passed


def test_the_app_chooses_none_of_the_models():
    """The thresholds are calibrated on the pipeline's own models."""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ZoteroRAG":
            passed = {keyword.arg for keyword in node.keywords}
            assert not passed & {"dense_model_name", "qa_model", "reranker_model"}, passed
