"""Single source of truth for question-type presets and config resolution.

Each preset carries only the "live" hyperparameters actually consumed by the
pipeline (retrieval_threshold, rerank_threshold, qa_score_threshold,
min_answer_words, section_diversity). Per-type tuning lives in each preset's
literal default value; ``resolve`` applies user overrides on top with no hidden
per-type transform, so what you see is what runs.

Nothing is wired to this module yet; later slices migrate the UI and eval onto
it, replacing the duplicated presets in ``app.py`` and
``qa_engine.get_config_for_type``.
"""

from copy import deepcopy

# ponytail: plain dict of scalars — five live fields don't warrant a dataclass.
PRESETS = {
    "factoid": {
        "retrieval_threshold": 0.45,
        "rerank_threshold": 0.45,
        "qa_score_threshold": 0.10,
        "min_answer_words": 2,
        "section_diversity": False,
    },
    "methodology": {
        "retrieval_threshold": 0.35,
        "rerank_threshold": 0.40,
        "qa_score_threshold": 0.05,
        "min_answer_words": 5,
        "section_diversity": True,
    },
    "explanation": {
        "retrieval_threshold": 0.35,
        "rerank_threshold": 0.40,
        "qa_score_threshold": 0.05,
        "min_answer_words": 3,
        "section_diversity": False,
    },
    "comparison": {
        "retrieval_threshold": 0.45,
        "rerank_threshold": 0.45,
        "qa_score_threshold": 0.08,
        "min_answer_words": 3,
        "section_diversity": False,
    },
    "definition": {
        "retrieval_threshold": 0.45,
        "rerank_threshold": 0.45,
        "qa_score_threshold": 0.10,
        "min_answer_words": 3,
        "section_diversity": False,
    },
    "general": {
        "retrieval_threshold": 0.45,
        "rerank_threshold": 0.45,
        "qa_score_threshold": 0.10,
        "min_answer_words": 3,
        "section_diversity": False,
    },
    "custom": {
        "retrieval_threshold": 0.45,
        "rerank_threshold": 0.45,
        "qa_score_threshold": 0.0,
        "min_answer_words": 3,
        "section_diversity": False,
    },
}


def resolve(question_type: str, overrides: dict | None = None) -> dict:
    """Return the final config for ``question_type`` with ``overrides`` applied.

    An unknown ``question_type`` falls back to the ``general`` preset. The
    resolved ``qa_score_threshold`` is literal: no per-type scaling.
    """
    config = deepcopy(PRESETS.get(question_type, PRESETS["general"]))
    if overrides:
        config.update(overrides)
    return config
