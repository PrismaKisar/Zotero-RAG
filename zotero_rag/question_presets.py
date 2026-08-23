"""Single source of truth for question-type presets and config resolution.

Each preset carries only the "live" hyperparameters actually consumed by the
pipeline (retrieval_threshold, rerank_threshold, qa_score_threshold,
min_answer_words, section_diversity_enabled, retrieval_mode, result_limit,
rerank_enabled). Per-type tuning lives in each preset's literal default value;
``resolve`` applies user overrides on top with no hidden per-type transform, so
what you see is what runs.

The QA pipeline (``ZoteroRAG.answer_question``) resolves its config through
this module. Each live field is read by the pipeline: retrieval_mode,
result_limit and retrieval_threshold in stage 1, rerank_enabled and
rerank_threshold in stage 2, qa_score_threshold / min_answer_words /
section_diversity_enabled in QA extraction.

retrieval_mode / result_limit / rerank_enabled exist because the thresholds
alone cannot change the *ranking* the retrieval metrics score - they only
filter an already-fixed order. An ablation restricted to thresholds is
therefore unfalsifiable by construction; these three knobs are the ones that
actually move recall@k and MRR without re-indexing.

``rerank_threshold`` lives on a different scale from the other two: bge-reranker-base
is a conservative model whose probabilities stay low even for real evidence.
Measured on the QASPER golden set, gold evidence has p50 0.007 and p95 0.92 while
random chunks sit at p50 0.000; 0.001 keeps 70% of the evidence and 17% of the
noise, where the old 0.40-0.45 kept only 11-13% of the evidence. It used to look
safe because a second sigmoid squashed every score into [0.5, 0.73], so the
filter passed everything.
"""

from copy import deepcopy

# ponytail: plain dict of scalars — five live fields don't warrant a dataclass.
PRESETS = {
    "factoid": {
        "retrieval_threshold": 0.45,
        "rerank_threshold": 0.001,
        "qa_score_threshold": 0.10,
        "min_answer_words": 2,
        "section_diversity_enabled": False,
        "retrieval_mode": "hybrid",
        "result_limit": 30,
        "rerank_enabled": True,
    },
    "methodology": {
        "retrieval_threshold": 0.35,
        "rerank_threshold": 0.001,
        "qa_score_threshold": 0.05,
        "min_answer_words": 5,
        "section_diversity_enabled": True,
        "retrieval_mode": "hybrid",
        "result_limit": 30,
        "rerank_enabled": True,
    },
    "explanation": {
        "retrieval_threshold": 0.35,
        "rerank_threshold": 0.001,
        "qa_score_threshold": 0.05,
        "min_answer_words": 3,
        "section_diversity_enabled": False,
        "retrieval_mode": "hybrid",
        "result_limit": 30,
        "rerank_enabled": True,
    },
    "comparison": {
        "retrieval_threshold": 0.45,
        "rerank_threshold": 0.001,
        "qa_score_threshold": 0.08,
        "min_answer_words": 3,
        "section_diversity_enabled": False,
        "retrieval_mode": "hybrid",
        "result_limit": 30,
        "rerank_enabled": True,
    },
    "definition": {
        "retrieval_threshold": 0.45,
        "rerank_threshold": 0.001,
        "qa_score_threshold": 0.10,
        "min_answer_words": 3,
        "section_diversity_enabled": False,
        "retrieval_mode": "hybrid",
        "result_limit": 30,
        "rerank_enabled": True,
    },
    "general": {
        "retrieval_threshold": 0.45,
        "rerank_threshold": 0.001,
        "qa_score_threshold": 0.10,
        "min_answer_words": 3,
        "section_diversity_enabled": False,
        "retrieval_mode": "hybrid",
        "result_limit": 30,
        "rerank_enabled": True,
    },
    "custom": {
        "retrieval_threshold": 0.45,
        "rerank_threshold": 0.001,
        "qa_score_threshold": 0.0,
        "min_answer_words": 3,
        "section_diversity_enabled": False,
        "retrieval_mode": "hybrid",
        "result_limit": 30,
        "rerank_enabled": True,
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
