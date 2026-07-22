"""Regression guard: the Streamlit UI reads the shared presets.

Locks in the single-source rule for the UI: `app.py` must not carry its own
copy of the question-type hyperparameters, and must prefill its widgets from
`question_presets`. The checks are static so they need neither streamlit nor a
running app.
"""

import ast
from pathlib import Path

from zotero_rag.question_presets import PRESETS

APP = Path(__file__).resolve().parent.parent / "zotero_rag" / "app.py"


def _app_source() -> str:
    return APP.read_text()


def _dict_literal(name: str) -> dict:
    """Return the dict literal assigned to ``name`` anywhere in app.py."""
    tree = ast.parse(_app_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in app.py")


def test_app_does_not_duplicate_the_preset_dict():
    assert "QUESTION_TYPE_PRESETS" not in _app_source(), (
        "app.py still carries a duplicated question-type preset dict"
    )


def test_app_prefills_widgets_from_the_shared_module():
    """app.py imports the shared module and resolves the preset through it."""
    tree = ast.parse(_app_source())
    imports_shared = any(
        isinstance(node, ast.ImportFrom) and node.module == "question_presets"
        for node in ast.walk(tree)
    )
    calls_resolve = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve"
        for node in ast.walk(tree)
    )
    assert imports_shared, "app.py does not import the shared preset module"
    assert calls_resolve, "app.py does not resolve the preset through the shared module"


def test_ui_descriptions_cover_every_shared_preset_type():
    descriptions = _dict_literal("QUESTION_TYPE_DESCRIPTIONS")
    assert set(descriptions) == set(PRESETS), (
        "the type selector and the shared presets disagree on question types"
    )


def test_ui_copy_carries_no_hyperparameters():
    """The UI-only dict holds plain copy, never nested hyperparameters."""
    descriptions = _dict_literal("QUESTION_TYPE_DESCRIPTIONS")
    for question_type, text in descriptions.items():
        assert isinstance(text, str), f"{question_type} carries more than copy"
