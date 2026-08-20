"""Regression guard: example configs only carry knobs the pipeline reads.

`load_config` does no key validation, so a dead knob in an `overrides` block
silently rides into the resolver overrides and looks like a tuning dial that
does nothing. These tests keep the shipped examples honest: every override key
must be a live preset field, and every question type must exist in the shared
presets.
"""

import ast
from pathlib import Path

import pytest
import yaml

from zotero_rag.question_presets import PRESETS

CONFIG_DIR = Path(__file__).resolve().parent.parent / "example_configs"
LIVE_FIELDS = {key for preset in PRESETS.values() for key in preset}

# The keys run_from_config actually reads out of the `defaults` block; anything
# else there is a knob the runner silently ignores.
RUNNER_DEFAULTS = {
    "num_paraphrases",
    "retrieval_threshold",
    "rerank_threshold",
    "highlight_color",
    "question_type",
    "overrides",
}

CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _override_blocks(config: dict):
    """Yield every overrides block in a loaded config."""
    blocks = [config.get("defaults", {}).get("overrides")]
    blocks += [q.get("overrides") for q in config.get("questions", [])]
    return [b for b in blocks if b]


def test_example_configs_exist():
    assert CONFIGS, "no example configs found"


def test_orphan_preset_yaml_is_deleted():
    assert not (CONFIG_DIR / "question_type_presets.yaml").exists()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_filename_names_only_the_variant(path):
    """The directory already says "example config"; the file names the variant."""
    assert not any(word in path.stem for word in ("config", "example")), path.name


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_is_valid_yaml(path):
    assert isinstance(_load(path), dict), f"{path.name} is not a YAML mapping"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_overrides_keys_are_live_preset_fields(path):
    for block in _override_blocks(_load(path)):
        dead = set(block) - LIVE_FIELDS
        assert not dead, f"{path.name} sets keys the pipeline never reads: {sorted(dead)}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_defaults_only_expose_keys_the_runner_reads(path):
    defaults = _load(path).get("defaults", {})
    ignored = set(defaults) - RUNNER_DEFAULTS
    assert not ignored, f"{path.name} sets defaults the runner ignores: {sorted(ignored)}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_question_types_are_known(path):
    config = _load(path)
    types = [config.get("defaults", {}).get("question_type")]
    types += [q.get("question_type") for q in config.get("questions", [])]
    for question_type in filter(None, types):
        assert question_type in PRESETS, f"{path.name}: unknown question type {question_type!r}"


# Keys the runner itself consumes; everything else at the top level is forwarded
# to ZoteroRAG and must therefore carry that constructor's parameter name.
RUNNER_KEYS = {
    "source_type", "zotero_data_dir", "zotero_collection", "folder_path",
    "defaults", "questions", "rebuild_index", "create_highlighted_pdfs",
    "results_file",
}


def _zotero_rag_parameters() -> set[str]:
    """Parameter names of ZoteroRAG.__init__, read without importing the package."""
    source = (Path(__file__).resolve().parent.parent / "zotero_rag" / "pipeline.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return {arg.arg for arg in node.args.args} - {"self"}
    raise AssertionError("ZoteroRAG.__init__ not found")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_top_level_keys_name_the_constructor_parameter_they_feed(path):
    unknown = set(_load(path)) - RUNNER_KEYS - _zotero_rag_parameters()
    assert not unknown, f"{path.name} sets keys nothing reads: {sorted(unknown)}"
