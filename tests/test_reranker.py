"""Tests that the reranker reports the cross-encoder probabilities unchanged,
and cuts candidates relative to each question's best one.

``CrossEncoder.predict`` is already called with a Sigmoid activation, so the
extra sigmoid the code used to apply on top squashed every score into
[0.5, 0.73] — a relevant passage and a distractor came out nearly identical and
the threshold stopped discriminating. The cut that reads those scores is two
decisions, not one: a gate on the best candidate says whether the question is
answered at all, a margin below it says what else is worth keeping. These tests
pin both, and the scores end to end, with a stub model, so no checkpoint is
downloaded.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

import reranker as rr
from models import Chunk


class FakeCrossEncoder:
    """Returns queued probabilities and records the batch sizes it was called with."""

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device
        self.probabilities = []
        self.batch_sizes = []

    def predict(self, pairs, activation_fn=None, show_progress_bar=False):
        self.batch_sizes.append(len(pairs))
        head, self.probabilities = self.probabilities[:len(pairs)], self.probabilities[len(pairs):]
        return head


@pytest.fixture
def reranker(monkeypatch):
    monkeypatch.setattr(rr, "CrossEncoder", FakeCrossEncoder)
    return rr.Reranker(device="cpu")


def candidates(count):
    return [
        (Chunk(text=f"passage {i}", page_number=1, chunk_index=i, title="t", pdf_hash="h"), 0.5)
        for i in range(count)
    ]


def test_scores_are_reported_as_the_model_returned_them(reranker):
    reranker.model.probabilities = [0.97, 0.02]
    results = reranker.rerank("q", candidates(2))
    assert [r.rerank_score for r in results] == [0.97, 0.02]


def test_by_default_nothing_is_cut(reranker):
    """The benchmarks that only want the scores must not have to ask for them."""
    reranker.model.probabilities = [0.0001, 0.9]
    assert len(reranker.rerank("q", candidates(2))) == 2


def test_no_candidate_reaching_the_gate_leaves_the_question_unanswered(reranker):
    """Many middling candidates are still not an answer."""
    reranker.model.probabilities = [0.86, 0.87]
    assert reranker.rerank("q", candidates(2), gate=0.88, margin=0.02) == []


def test_the_margin_is_measured_from_this_question_s_best(reranker):
    reranker.model.probabilities = [0.95, 0.94, 0.92]
    kept = reranker.rerank("q", candidates(3), gate=0.88, margin=0.02)
    assert [round(r.rerank_score, 2) for r in kept] == [0.95, 0.94]


def test_a_low_scoring_question_keeps_its_own_crowd(reranker):
    """The same margin, one question lower down: an absolute cut would empty it."""
    reranker.model.probabilities = [0.89, 0.88]
    kept = reranker.rerank("q", candidates(2), gate=0.88, margin=0.02)
    assert [round(r.rerank_score, 2) for r in kept] == [0.89, 0.88]


def test_scoring_respects_the_configured_batch_size(monkeypatch):
    monkeypatch.setattr(rr, "CrossEncoder", FakeCrossEncoder)
    reranker = rr.Reranker(device="cpu", batch_size=3)
    reranker.model.probabilities = [0.5] * 7
    reranker.rerank("q", candidates(7))
    assert reranker.model.batch_sizes == [3, 3, 1]


def test_no_candidates_short_circuits(reranker):
    assert reranker.rerank("q", []) == []


# Filtering and ordering are separable uses of the same score. On the benchmarks
# the cut roughly doubles attribution precision while the ordering puts gold
# evidence below where retrieval had it, so the two are varied apart.

def scored_candidates(retrieval_scores):
    return [
        (Chunk(text=f"passage {i}", page_number=1, chunk_index=i, title="t", pdf_hash="h"), s)
        for i, s in enumerate(retrieval_scores)
    ]


def test_by_default_the_cross_encoder_decides_the_order(reranker):
    reranker.model.probabilities = [0.1, 0.9]
    out = reranker.rerank("q", scored_candidates([0.9, 0.1]))
    assert [c.chunk.chunk_index for c in out] == [1, 0]


def test_ordering_by_retrieval_keeps_the_cut_but_not_the_ranking(reranker):
    reranker.model.probabilities = [0.1, 0.9]
    out = reranker.rerank("q", scored_candidates([0.9, 0.1]), order_by_retrieval=True)
    assert [c.chunk.chunk_index for c in out] == [0, 1]


def test_ordering_by_retrieval_still_drops_what_the_margin_rejects(reranker):
    """The intervention keeps the filter; only the sort key changes."""
    reranker.model.probabilities = [0.9, 0.001]
    out = reranker.rerank("q", scored_candidates([0.1, 0.9]), gate=0.5, margin=0.1,
                          order_by_retrieval=True)
    assert [c.chunk.chunk_index for c in out] == [0]


def test_the_rerank_score_is_still_reported_when_ordering_by_retrieval(reranker):
    """Attribution and the ablation read the score, so reordering must not hide it."""
    reranker.model.probabilities = [0.1, 0.9]
    out = reranker.rerank("q", scored_candidates([0.9, 0.1]), order_by_retrieval=True)
    assert [round(c.rerank_score, 3) for c in out] == [0.1, 0.9]


def test_the_absolute_cut_keeps_what_the_margin_would_drop(reranker):
    reranker.model.probabilities = [0.99, 0.91]
    relative = reranker.rerank("q", scored_candidates([0.5, 0.5]), gate=0.9, margin=0.02)
    reranker.model.probabilities = [0.99, 0.91]
    absolute = reranker.rerank("q", scored_candidates([0.5, 0.5]), gate=0.9, margin=0.02,
                               absolute_cut=True)
    assert len(relative) == 1 and len(absolute) == 2
