"""Tests for EmbeddingManager runtime wiring: device flag and batch size.

FastEmbed runs ONNX Runtime rather than torch, so the only accelerator switch it
exposes is the CUDA flag; every other device lands on the CPU provider. These
tests pin that mapping, and the batch size, without downloading any model.
"""

import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

import embedding_manager as em


class FakeEmbedding:
    """Stand-in for TextEmbedding/SparseTextEmbedding that records its kwargs."""

    instances: ClassVar[list["FakeEmbedding"]] = []

    def __init__(self, model_name, cuda=False, providers=None):
        self.model_name = model_name
        self.cuda = cuda
        self.providers = providers
        FakeEmbedding.instances.append(self)

    @staticmethod
    def list_supported_models():
        return [{"model": "BAAI/bge-base-en-v1.5", "dim": 768}]

    def embed(self, texts):
        return iter([[0.0] * 768 for _ in texts])


class FakeSparseVector:
    """Minimal stand-in for a fastembed SparseEmbedding result."""

    def __init__(self):
        self.indices = np.array([0])
        self.values = np.array([1.0])


@pytest.fixture
def fake_fastembed(monkeypatch):
    FakeEmbedding.instances = []
    monkeypatch.setattr(em, "TextEmbedding", FakeEmbedding)
    monkeypatch.setattr(em, "SparseTextEmbedding", FakeEmbedding)
    return FakeEmbedding


def _dense(fake):
    return fake.instances[0]


def _sparse(fake):
    return fake.instances[1]


def test_cuda_device_enables_the_cuda_flag(fake_fastembed):
    em.EmbeddingManager(device="cuda")
    assert _dense(fake_fastembed).cuda is True
    assert _sparse(fake_fastembed).cuda is True


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_non_cuda_devices_leave_the_cuda_flag_off(fake_fastembed, device):
    """'mps' is a torch device; FastEmbed has no MPS path, so it must not set cuda."""
    em.EmbeddingManager(device=device)
    assert _dense(fake_fastembed).cuda is False


@pytest.mark.parametrize("requested", [None, 0])
def test_missing_batch_size_falls_back_to_the_default(fake_fastembed, requested):
    manager = em.EmbeddingManager(device="cpu", encode_batch_size=requested)
    assert manager.encode_batch_size == em.DEFAULT_ENCODE_BATCH_SIZE


def test_explicit_batch_size_is_honoured(fake_fastembed):
    assert em.EmbeddingManager(device="cpu", encode_batch_size=7).encode_batch_size == 7


def test_encode_chunks_respects_the_batch_size(fake_fastembed):
    """No hidden probing run: the configured size is the size actually used."""
    manager = em.EmbeddingManager(device="cpu", encode_batch_size=3)
    seen_batches = []

    def record_dense(texts):
        seen_batches.append(len(texts))
        return iter([[0.0] * 768 for _ in texts])

    manager.dense_model.embed = record_dense
    manager.sparse_model.embed = lambda texts: iter([FakeSparseVector() for _ in texts])

    manager.encode_chunks(None, [f"chunk {i}" for i in range(7)])

    assert seen_batches == [3, 3, 1]
