"""Regression guard: example configs only carry knobs the pipeline reads.

`load_config` does no key validation, so a dead knob in a `custom_config`
silently rides into the resolver overrides and looks like a tuning dial that
does nothing. These tests keep the shipped examples honest: every override key
must be a live preset field, and every question type must exist in the shared
presets.
"""

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
    "custom_config",
}

CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _custom_configs(config: dict):
    """Yield every custom_config block in a loaded config."""
    blocks = [config.get("defaults", {}).get("custom_config")]
    blocks += [q.get("custom_config") for q in config.get("questions", [])]
    return [b for b in blocks if b]


def test_example_configs_exist():
    assert CONFIGS, "no example configs found"


def test_orphan_preset_yaml_is_deleted():
    assert not (CONFIG_DIR / "question_type_presets.yaml").exists()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_is_valid_yaml(path):
    assert isinstance(_load(path), dict), f"{path.name} is not a YAML mapping"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_custom_config_keys_are_live_preset_fields(path):
    for block in _custom_configs(_load(path)):
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
