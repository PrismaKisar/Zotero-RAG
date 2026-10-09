"""Tests for the QA span picker.

The engine used to argmax the start and end logits independently and score the
result with a sigmoid over their mean. Both are wrong: the pair could be
inverted or unrelated, and the score was not a probability, so it was not
comparable across candidates. These tests pin the joint search and the
probability, using logits directly so no model is loaded.
"""

import types

import numpy as np
import torch

from zotero_rag.answering.qa_engine import (
    CONTEXT_STRIDE,
    MAX_ANSWER_TOKENS,
    MAX_SEQUENCE_TOKENS,
    MAX_WINDOWS_PER_FORWARD,
    QAEngine,
    best_span,
    softmax,
)


def context_ids(length, context_slice):
    """Sequence ids marking ``context_slice`` as context (1) and the rest as question (0)."""
    ids = [0] * length
    for i in range(*context_slice):
        ids[i] = 1
    return ids


def test_softmax_sums_to_one_and_survives_large_logits():
    probs = softmax(np.array([1000.0, 999.0, -1000.0]))
    assert np.isclose(probs.sum(), 1.0)
    assert probs[0] > probs[1] > probs[2]


def test_span_never_ends_before_it_starts():
    # Start peaks late, end peaks early: independent argmaxes would invert the span.
    start = np.array([0.0, 0.0, 0.0, 9.0])
    end = np.array([0.0, 9.0, 0.0, 0.0])
    start_token, end_token, _ = best_span(start, end, context_ids(4, (0, 4)))
    assert start_token <= end_token


def test_span_picks_the_jointly_best_pair():
    start = np.array([0.0, 5.0, 0.0, 4.0])
    end = np.array([0.0, 0.0, 6.0, 0.0])
    assert best_span(start, end, context_ids(4, (0, 4)))[:2] == (1, 2)


def test_span_stays_inside_the_context():
    # The strongest logits sit on question tokens 0-1, which must be ignored.
    start = np.array([9.0, 0.0, 1.0, 0.0])
    end = np.array([0.0, 9.0, 0.0, 1.0])
    start_token, end_token, _ = best_span(start, end, context_ids(4, (2, 4)))
    assert (start_token, end_token) == (2, 3)


def test_span_length_is_bounded():
    length = MAX_ANSWER_TOKENS + 10
    start = np.zeros(length)
    end = np.zeros(length)
    start[0] = 9.0
    end[-1] = 9.0
    start_token, end_token, _ = best_span(start, end, context_ids(length, (0, length)))
    assert end_token - start_token <= MAX_ANSWER_TOKENS


def test_score_is_a_probability_and_ranks_confidence():
    ids = context_ids(4, (0, 4))
    confident = best_span(np.array([9.0, 0.0, 0.0, 0.0]), np.array([0.0, 9.0, 0.0, 0.0]), ids)[2]
    flat = best_span(np.zeros(4), np.zeros(4), ids)[2]
    assert 0.0 <= flat < confident <= 1.0


def test_no_answer_mass_lowers_the_score_but_never_moves_the_span():
    """[CLS] at position 0 is the model's "no answer": it counts, but is never a span."""
    ids = [None, 0, None, 1, 1, None]
    start = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0])
    end = np.array([0.0, 0.0, 0.0, 0.0, 4.0, 0.0])
    no_answer_start, no_answer_end = start.copy(), end.copy()
    no_answer_start[0] = no_answer_end[0] = 8.0
    answerable = best_span(start, end, ids)
    unanswerable = best_span(no_answer_start, no_answer_end, ids)
    assert answerable[:2] == unanswerable[:2] == (3, 4)
    assert unanswerable[2] < 0.01 < answerable[2]


def test_no_context_token_yields_no_span():
    assert best_span(np.zeros(4), np.zeros(4), [0, 0, None, None]) is None


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


# score_windows is the one place a (question, context) batch becomes windows and
# logits: extract_answers reads spans off it and benchmark/studies/qa_score_distribution.py
# reads scores off it, so the two cannot drift apart in how they tokenise.

class _FakeEncoding(dict):
    """The parts of a fast tokenizer's BatchEncoding that score_windows reads."""

    def __init__(self, sample_mapping, width=3):
        rows = len(sample_mapping)
        ids = torch.arange(rows * width, dtype=torch.long).reshape(rows, width)
        super().__init__(input_ids=ids, attention_mask=torch.ones_like(ids),
                         overflow_to_sample_mapping=torch.tensor(sample_mapping),
                         offset_mapping=torch.zeros(rows, width, 2, dtype=torch.long))
        self.moved_to = None

    def sequence_ids(self, k):
        return [None, 0, 1]

    def to(self, device):
        self.moved_to = device
        return self


class _FakeTokenizer:
    def __init__(self, sample_mapping):
        self.encoding = _FakeEncoding(sample_mapping)
        self.calls = []

    def __call__(self, questions, contexts, **kwargs):
        self.calls.append((questions, contexts, kwargs))
        return self.encoding


def test_score_windows_maps_every_window_back_to_its_context():
    engine = _engine_with(_CountingModel())
    engine.device = "cpu"
    # the second context is long enough to overflow into two windows
    engine.tokenizer = _FakeTokenizer([0, 1, 1])

    windows = engine.score_windows("q", ["short", "long"])

    assert windows.sample_mapping == [0, 1, 1]
    assert windows.sequence_ids == [[None, 0, 1]] * 3
    assert windows.offsets.shape == (3, 3, 2)
    assert np.array_equal(windows.start_logits, np.arange(9, dtype=np.float32).reshape(3, 3))
    assert engine.tokenizer.encoding.moved_to == "cpu"
    # neither bookkeeping field reaches the model
    assert set(engine.tokenizer.encoding) == {"input_ids", "attention_mask"}


def test_score_windows_asks_every_context_the_same_question_in_overflowing_windows():
    engine = _engine_with(_CountingModel())
    engine.device = "cpu"
    engine.tokenizer = _FakeTokenizer([0, 1])

    engine.score_windows("q", ["a", "b"])

    [(questions, contexts, kwargs)] = engine.tokenizer.calls
    assert (questions, contexts) == (["q", "q"], ["a", "b"])
    assert kwargs["truncation"] == "only_second"
    assert kwargs["return_overflowing_tokens"] and kwargs["return_offsets_mapping"]
    assert (kwargs["max_length"], kwargs["stride"]) == (MAX_SEQUENCE_TOKENS, CONTEXT_STRIDE)


def test_extract_answers_reads_each_candidate_s_span_off_its_windows():
    from zotero_rag.models import Chunk, RerankedChunk

    engine = _engine_with(_CountingModel())
    engine.device = "cpu"
    engine.batch_size = 8
    engine.tokenizer = _FakeTokenizer([0, 1])
    # the only context token of each window covers the first five characters
    engine.tokenizer.encoding["offset_mapping"][:, 2] = torch.tensor([0, 5])
    candidates = [RerankedChunk(Chunk(text=text, page_number=1, chunk_index=i,
                                      title=f"T{i}", pdf_hash="h"), 0.5, 0.9)
                  for i, text in enumerate(["alpha beta", "gamma delta"])]

    answers = engine.extract_answers("q", candidates, {"qa_score_threshold": 0.0})

    assert sorted(a.text for a in answers) == ["alpha", "gamma"]
    assert all((a.start_char, a.end_char) == (0, 5) for a in answers)


def _two_candidate_engine(texts, titles):
    from zotero_rag.models import Chunk, RerankedChunk

    engine = _engine_with(_CountingModel())
    engine.device = "cpu"
    engine.batch_size = 8
    engine.tokenizer = _FakeTokenizer([0, 1])
    engine.tokenizer.encoding["offset_mapping"][:, 2] = torch.tensor([0, 5])
    candidates = [RerankedChunk(Chunk(text=text, page_number=1, chunk_index=i,
                                      title=title, pdf_hash="h"), 0.5, 0.9)
                  for i, (text, title) in enumerate(zip(texts, titles))]
    return engine, candidates


def test_extract_answers_drops_spans_below_the_threshold():
    engine, candidates = _two_candidate_engine(["alpha beta", "gamma delta"], ["A", "B"])
    assert engine.extract_answers("q", candidates, {"qa_score_threshold": 0.99}) == []


def test_extract_answers_keeps_one_of_two_identical_spans_from_one_paper():
    engine, candidates = _two_candidate_engine(["Alpha beta", "alpha  gamma"], ["A", "A"])
    answers = engine.extract_answers("q", candidates, {"qa_score_threshold": 0.0})
    assert [a.text for a in answers] == ["Alpha"]


def test_the_extractive_reader_is_its_model_and_never_quotes_or_fails(monkeypatch):
    """The Reader seam's shared fields: the ablation reads them on every reader."""
    monkeypatch.setattr(QAEngine, "_load_model", lambda self: None)
    engine = QAEngine(model_name="deepset/roberta-base-squad2", device="cpu")
    assert engine.reader_kind == "deepset/roberta-base-squad2"
    assert engine.last_quote_stats == {"cited": 0, "matched": 0}
    assert engine.generation_failures == 0
