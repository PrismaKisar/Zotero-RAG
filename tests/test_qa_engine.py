"""Tests for the QA span picker.

The engine used to argmax the start and end logits independently and score the
result with a sigmoid over their mean. Both are wrong: the pair could be
inverted or unrelated, and the score was not a probability, so it was not
comparable across candidates. These tests pin the joint search and the
probability, using logits directly so no model is loaded.
"""

import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from qa_engine import (
    MAX_ANSWER_TOKENS,
    MAX_WINDOWS_PER_FORWARD,
    QAEngine,
    _best_span,
    _softmax,
)


def context_ids(length, context_slice):
    """Sequence ids marking ``context_slice`` as context (1) and the rest as question (0)."""
    ids = [0] * length
    for i in range(*context_slice):
        ids[i] = 1
    return ids


def test_softmax_sums_to_one_and_survives_large_logits():
    probs = _softmax(np.array([1000.0, 999.0, -1000.0]))
    assert np.isclose(probs.sum(), 1.0)
    assert probs[0] > probs[1] > probs[2]


def test_span_never_ends_before_it_starts():
    # Start peaks late, end peaks early: independent argmaxes would invert the span.
    start = np.array([0.0, 0.0, 0.0, 9.0])
    end = np.array([0.0, 9.0, 0.0, 0.0])
    start_token, end_token, _ = _best_span(start, end, context_ids(4, (0, 4)))
    assert start_token <= end_token


def test_span_picks_the_jointly_best_pair():
    start = np.array([0.0, 5.0, 0.0, 4.0])
    end = np.array([0.0, 0.0, 6.0, 0.0])
    assert _best_span(start, end, context_ids(4, (0, 4)))[:2] == (1, 2)


def test_span_stays_inside_the_context():
    # The strongest logits sit on question tokens 0-1, which must be ignored.
    start = np.array([9.0, 0.0, 1.0, 0.0])
    end = np.array([0.0, 9.0, 0.0, 1.0])
    start_token, end_token, _ = _best_span(start, end, context_ids(4, (2, 4)))
    assert (start_token, end_token) == (2, 3)


def test_span_length_is_bounded():
    length = MAX_ANSWER_TOKENS + 10
    start = np.zeros(length)
    end = np.zeros(length)
    start[0] = 9.0
    end[-1] = 9.0
    start_token, end_token, _ = _best_span(start, end, context_ids(length, (0, length)))
    assert end_token - start_token <= MAX_ANSWER_TOKENS


def test_score_is_a_probability_and_ranks_confidence():
    ids = context_ids(4, (0, 4))
    confident = _best_span(np.array([9.0, 0.0, 0.0, 0.0]), np.array([0.0, 9.0, 0.0, 0.0]), ids)[2]
    flat = _best_span(np.zeros(4), np.zeros(4), ids)[2]
    assert 0.0 <= flat < confident <= 1.0


def test_no_context_token_yields_no_span():
    assert _best_span(np.zeros(4), np.zeros(4), [0, 0, None, None]) is None


# Windowed inference: batch_size caps the (question, context) pairs handed to the
# tokenizer, not the windows a long context expands into, so the tensor reaching
# the model was unbounded. These pin that splitting it changes memory and nothing
# else - the logits must match a single pass row for row.

class _CountingModel:
    """Returns logits derived from the input ids, so order and content are checkable."""

    def __init__(self):
        self.forward_sizes = []

    def __call__(self, **window):
        ids = window["input_ids"]
        self.forward_sizes.append(ids.shape[0])
        return types.SimpleNamespace(start_logits=ids.float(), end_logits=ids.float() * 2)


def _engine_with(model):
    engine = QAEngine.__new__(QAEngine)
    engine.model = model
    return engine


def _inputs(rows, width=3):
    ids = torch.arange(rows * width, dtype=torch.long).reshape(rows, width)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_windowed_inference_matches_a_single_forward_pass():
    inputs = _inputs(MAX_WINDOWS_PER_FORWARD * 2 + 3)

    split_start, split_end = _engine_with(_CountingModel())._logits_in_windows(inputs)
    whole = _CountingModel()(**inputs)

    assert np.array_equal(split_start, whole.start_logits.numpy())
    assert np.array_equal(split_end, whole.end_logits.numpy())


def test_no_forward_pass_exceeds_the_window_cap():
    model = _CountingModel()
    rows = MAX_WINDOWS_PER_FORWARD * 2 + 3

    _engine_with(model)._logits_in_windows(_inputs(rows))

    assert max(model.forward_sizes) <= MAX_WINDOWS_PER_FORWARD
    assert sum(model.forward_sizes) == rows


def test_a_batch_below_the_cap_still_runs_in_one_pass():
    model = _CountingModel()
    _engine_with(model)._logits_in_windows(_inputs(MAX_WINDOWS_PER_FORWARD - 1))
    assert model.forward_sizes == [MAX_WINDOWS_PER_FORWARD - 1]
