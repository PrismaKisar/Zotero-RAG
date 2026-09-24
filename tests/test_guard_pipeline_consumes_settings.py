"""Regression guard: every setting must be read by the QA pipeline.

A setting that nothing consumes (as ``max_answer_length`` and
``prefer_entities`` once were) looks like a decision and does nothing. The
check is static, so it needs neither torch nor a live index: it reads the
pipeline sources and asserts each key in ``SETTINGS`` appears as a ``config``
access.
"""

from pathlib import Path

from zotero_rag.settings import SETTINGS

# Modules that make up the runtime pipeline reading the resolved config.
PIPELINE_SOURCES = ("pipeline.py", "qa_engine.py")


def _pipeline_text() -> str:
    pkg = Path(__file__).resolve().parent.parent / "zotero_rag"
    return "\n".join((pkg / name).read_text() for name in PIPELINE_SOURCES)


def test_every_setting_is_read_by_the_pipeline():
    source = _pipeline_text()
    for key in SETTINGS:
        reads = (f"config['{key}']", f'config["{key}"]',
                 f"config.get('{key}'", f'config.get("{key}"')
        assert any(r in source for r in reads), (
            f"setting '{key}' is never read by the pipeline "
            f"({', '.join(PIPELINE_SOURCES)})"
        )
