"""Tests for EmbeddingManager runtime wiring: device, prompts, and batch size.

The dense model runs on sentence-transformers (torch), so it takes the resolved
device directly; BM25 stays on FastEmbed, whose only accelerator switch is the
CUDA flag. These tests pin that wiring without downloading any model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

import embedding_manager as em

DIM = 768


class FakeDense:
    """Stand-in for SentenceTransformer that records which encoder was called."""

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device
        self.max_seq_length = None
        self.calls = []

    def get_embedding_dimension(self):
        return DIM

    def encode_document(self, texts, batch_size=32):
        self.calls.append(("document", len(texts)))
        return np.zeros((len(texts), DIM), dtype=np.float32)

    def encode_query(self, text):
        self.calls.append(("query", 1))
        return np.zeros(DIM, dtype=np.float32)


class FakeSparseVector:
    """Minimal stand-in for a fastembed SparseEmbedding result."""

    def __init__(self):
        self.indices = np.array([0])
        self.values = np.array([1.0])


class FakeSparse:
    """Stand-in for SparseTextEmbedding that records its CUDA flag."""

    def __init__(self, model_name, cuda=False):
        self.model_name = model_name
        self.cuda = cuda

    def embed(self, texts):
        return iter([FakeSparseVector() for _ in texts])


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(em, "SentenceTransformer", FakeDense)
    monkeypatch.setattr(em, "SparseTextEmbedding", FakeSparse)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_dense_model_runs_on_the_resolved_device(device):
    assert em.EmbeddingManager(device=device).dense_model.device == device


@pytest.mark.parametrize(("device", "cuda"), [("cuda", True), ("cpu", False), ("mps", False)])
def test_only_cuda_turns_on_the_sparse_cuda_flag(device, cuda):
    """'mps' is a torch device; FastEmbed has no MPS path, so it must not set cuda."""
    assert em.EmbeddingManager(device=device).sparse_model.cuda is cuda


def test_dense_model_reads_the_reranker_window_and_reports_its_size():
    manager = em.EmbeddingManager(device="cpu")
    assert manager.dense_model.max_seq_length == em.MAX_TOKENS
    assert manager.vector_size == DIM


@pytest.mark.parametrize("requested", [None, 0])
def test_missing_batch_size_falls_back_to_the_default(requested):
    manager = em.EmbeddingManager(device="cpu", encode_batch_size=requested)
    assert manager.encode_batch_size == em.DEFAULT_ENCODE_BATCH_SIZE


def test_encode_chunks_uses_the_document_prompt_in_configured_batches():
    manager = em.EmbeddingManager(device="cpu", encode_batch_size=3)

    encoded = manager.encode_chunks(None, [f"chunk {i}" for i in range(7)])

    assert manager.dense_model.calls == [("document", 3), ("document", 3), ("document", 1)]
    assert len(encoded["dense"]) == len(encoded["sparse"]) == 7
    assert encoded["sparse"][0] == {"indices": [0], "values": [1.0]}


def test_encode_query_uses_the_query_prompt():
    manager = em.EmbeddingManager(device="cpu")

    encoded = manager.encode_query("what is attention?")

    assert manager.dense_model.calls == [("query", 1)]
    assert len(encoded["dense"]) == DIM
    assert encoded["sparse"] == {"indices": [0], "values": [1.0]}


def test_each_model_default_is_named_in_one_module_only():
    """A second copy of a default drifts: the app and the YAML runner once
    disagreed with the pipeline about which dense model was the default."""
    pkg = Path(__file__).resolve().parent.parent / "zotero_rag"
    for name, owner in (("google/embeddinggemma-300m", "embedding_manager.py"),
                        ("deepset/deberta-v3-large-squad2", "qa_engine.py"),
                        ("Alibaba-NLP/gte-reranker-modernbert-base", "reranker.py"),
                        ("llama3.2:3b", "contextualizer.py")):
        holders = [p.name for p in pkg.glob("*.py") if name in p.read_text()]
        assert holders == [owner], (name, holders)
