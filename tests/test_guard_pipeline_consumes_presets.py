"""Regression guard: every preset key must be read by the QA pipeline.

This locks in the class of bug behind the preset-resolver effort: a preset key
that nothing in the pipeline consumes (as ``max_answer_length`` and
``prefer_entities`` once were). The check is static so it needs neither torch
nor a live index: it reads the pipeline sources and asserts each key in
``PRESETS`` appears as a ``config`` access.
"""

from pathlib import Path

from zotero_rag.question_presets import PRESETS

# Modules that make up the runtime pipeline reading the resolved config.
PIPELINE_SOURCES = ("pipeline.py", "qa_engine.py")


def _pipeline_text() -> str:
    pkg = Path(__file__).resolve().parent.parent / "zotero_rag"
    return "\n".join((pkg / name).read_text() for name in PIPELINE_SOURCES)


def test_every_preset_key_is_read_by_the_pipeline():
    source = _pipeline_text()
    preset_keys = {key for preset in PRESETS.values() for key in preset}
    for key in preset_keys:
        reads = (f"config['{key}']", f'config["{key}"]',
                 f"config.get('{key}'", f'config.get("{key}"')
        assert any(r in source for r in reads), (
            f"preset key '{key}' is never read by the pipeline "
            f"({', '.join(PIPELINE_SOURCES)})"
        )
