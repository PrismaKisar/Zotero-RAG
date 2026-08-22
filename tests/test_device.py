"""Tests for the shared compute-device resolver."""

import pytest

from zotero_rag.device import SUPPORTED_DEVICES, resolve_device


@pytest.mark.parametrize("name", SUPPORTED_DEVICES)
def test_explicit_device_is_returned_verbatim(name):
    assert resolve_device(name) == name


def test_explicit_device_is_case_insensitive():
    assert resolve_device("MPS") == "mps"


def test_unsupported_device_is_rejected():
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("tpu")


def test_autodetect_returns_a_supported_device():
    assert resolve_device(None) in SUPPORTED_DEVICES


def test_autodetect_prefers_cuda_then_mps_then_cpu(monkeypatch):
    """The whole point of the shared resolver: one ordering, not one per module."""
    import torch

    def configure(cuda: bool, mps: bool):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    configure(cuda=True, mps=True)
    assert resolve_device(None) == "cuda"

    configure(cuda=False, mps=True)
    assert resolve_device(None) == "mps"

    configure(cuda=False, mps=False)
    assert resolve_device(None) == "cpu"
