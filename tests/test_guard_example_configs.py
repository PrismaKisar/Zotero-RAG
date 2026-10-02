"""Regression guard: example configs carry only knobs the runner reads, and none
of the pipeline's fixed settings.

`load_config` does no key validation, so a dead key rides along looking like a
tuning dial that does nothing. The thresholds are fixed and calibrated on the
pipeline's own models (zotero_rag/settings.py), so a config may set neither the
thresholds nor any model, nor the question types and overrides that used to
carry them.
"""

import ast
from pathlib import Path

import pytest
import yaml

from zotero_rag.settings import SETTINGS

CONFIG_DIR = Path(__file__).resolve().parent.parent / "example_configs"
FIXED = {*SETTINGS, "question_type", "overrides", "qa_model", "reranker_model",
         "dense_model_name"}
# What run_from_config reads from `defaults` and from each question.
RUNNER_DEFAULTS = {"highlight_color"}
RUNNER_QUESTION_KEYS = {"question", "highlight_color"}

CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _keys(node):
    """Every mapping key anywhere in a loaded config."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _keys(value)


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
def test_no_config_sets_a_fixed_setting(path):
    fixed = FIXED & set(_keys(_load(path)))
    assert not fixed, f"{path.name} sets fixed settings: {sorted(fixed)}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_defaults_and_questions_only_carry_what_the_runner_reads(path):
    config = _load(path)
    ignored = set(config.get("defaults", {})) - RUNNER_DEFAULTS
    for question in config.get("questions", []):
        ignored |= set(question) - RUNNER_QUESTION_KEYS
    assert not ignored, f"{path.name} sets keys the runner ignores: {sorted(ignored)}"


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
