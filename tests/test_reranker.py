"""Tests that the reranker reports the cross-encoder probabilities unchanged.

``CrossEncoder.predict`` is already called with a Sigmoid activation, so the
extra sigmoid the code used to apply on top squashed every score into
[0.5, 0.73] — a relevant passage and a distractor came out nearly identical and
the threshold stopped discriminating. These tests pin the scores end to end with
a stub model, so no checkpoint is downloaded.
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
    results = reranker.rerank("q", candidates(2), threshold=0.0)
    assert [r.rerank_score for r in results] == [0.97, 0.02]


def test_a_distractor_stays_below_a_threshold_the_double_sigmoid_would_have_cleared(reranker):
    reranker.model.probabilities = [0.0001, 0.02]
    kept = reranker.rerank("q", candidates(2), threshold=0.001)
    assert [r.rerank_score for r in kept] == [0.02]


def test_variations_take_the_per_candidate_maximum(reranker):
    reranker.model.probabilities = [0.10, 0.80, 0.90, 0.20]
    results = reranker.rerank("q", candidates(2), threshold=0.0, query_variations=["a", "b"])
    assert sorted(r.rerank_score for r in results) == [0.80, 0.90]


def test_scoring_respects_the_configured_batch_size(monkeypatch):
    monkeypatch.setattr(rr, "CrossEncoder", FakeCrossEncoder)
    reranker = rr.Reranker(device="cpu", batch_size=3)
    reranker.model.probabilities = [0.5] * 7
    reranker.rerank("q", candidates(7), threshold=0.0)
    assert reranker.model.batch_sizes == [3, 3, 1]


def test_no_candidates_short_circuits(reranker):
    assert reranker.rerank("q", [], threshold=0.0) == []


# Filtering and ordering are separable uses of the same score. On the benchmarks
# the threshold roughly doubles attribution precision while the ordering puts
# gold evidence below where retrieval had it, so the two are varied apart.

def scored_candidates(retrieval_scores):
    return [
        (Chunk(text=f"passage {i}", page_number=1, chunk_index=i, title="t", pdf_hash="h"), s)
        for i, s in enumerate(retrieval_scores)
    ]


def test_by_default_the_cross_encoder_decides_the_order(reranker):
    reranker.model.probabilities = [0.1, 0.9]
    out = reranker.rerank("q", scored_candidates([0.9, 0.1]), threshold=0.0)
    assert [c.chunk.chunk_index for c in out] == [1, 0]


def test_ordering_by_retrieval_keeps_the_threshold_but_not_the_ranking(reranker):
    reranker.model.probabilities = [0.1, 0.9]
    out = reranker.rerank("q", scored_candidates([0.9, 0.1]), threshold=0.0,
                          order_by_retrieval=True)
    assert [c.chunk.chunk_index for c in out] == [0, 1]


def test_ordering_by_retrieval_still_drops_what_the_threshold_rejects(reranker):
    """The intervention keeps the filter; only the sort key changes."""
    reranker.model.probabilities = [0.9, 0.001]
    out = reranker.rerank("q", scored_candidates([0.1, 0.9]), threshold=0.5,
                          order_by_retrieval=True)
    assert [c.chunk.chunk_index for c in out] == [0]


def test_the_rerank_score_is_still_reported_when_ordering_by_retrieval(reranker):
    """Attribution and the ablation read the score, so reordering must not hide it."""
    reranker.model.probabilities = [0.1, 0.9]
    out = reranker.rerank("q", scored_candidates([0.9, 0.1]), threshold=0.0,
                          order_by_retrieval=True)
    assert [round(c.rerank_score, 3) for c in out] == [0.1, 0.9]
