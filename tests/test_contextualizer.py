"""Chunk contextualization: a document summary, then one context line per chunk.

A failed generation raises rather than indexing a raw chunk under a
configuration that says it was contextualized. Pinned with a stub Ollama client.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

import contextualizer as cx


class FakeClient:
    """Answers ``generate`` from a queue; an Exception in it is raised instead."""

    def __init__(self, host=None):
        self.host = host
        self.replies = []
        self.calls = []

    def generate(self, model, prompt=None, options=None, keep_alive=None):
        self.calls.append({"model": model, "prompt": prompt, "keep_alive": keep_alive})
        if prompt is None:  # the cache flush
            return {}
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"response": reply}


def _contextualizer(monkeypatch, replies):
    monkeypatch.setattr(cx.ollama, "Client", FakeClient)
    contextualizer = cx.ChunkContextualizer(ollama_url="http://ollama:1")
    contextualizer.client.replies = list(replies)
    return contextualizer


def test_each_chunk_is_prefixed_with_its_context_line(monkeypatch):
    contextualizer = _contextualizer(monkeypatch, ["summary", '"ctx a"', "ctx b"])

    assert contextualizer.contextualize("doc", ["a", "b"]) == ["ctx a\n\na", "ctx b\n\nb"]
    assert contextualizer.client.host == "http://ollama:1"
    assert "summary" in contextualizer.client.calls[1]["prompt"]


def test_a_failed_chunk_raises_instead_of_being_kept_raw(monkeypatch):
    contextualizer = _contextualizer(monkeypatch, ["summary", RuntimeError("down"), "ctx b"])
    with pytest.raises(RuntimeError, match="down"):
        contextualizer.contextualize("doc", ["a", "b"])


def test_a_failed_summary_raises(monkeypatch):
    contextualizer = _contextualizer(monkeypatch, [RuntimeError("down"), "ctx a"])
    with pytest.raises(RuntimeError, match="down"):
        contextualizer.contextualize("doc", ["a"])


def test_an_empty_context_line_leaves_the_chunk_as_it_was(monkeypatch):
    contextualizer = _contextualizer(monkeypatch, ["summary", "  "])
    assert contextualizer.contextualize("doc", ["a"]) == ["a"]


def test_the_model_is_unloaded_once_the_document_is_done(monkeypatch):
    contextualizer = _contextualizer(monkeypatch, ["summary", "ctx a"])
    contextualizer.contextualize("doc", ["a"])
    assert contextualizer.client.calls[-1] == {
        "model": cx.DEFAULT_CONTEXT_MODEL, "prompt": None, "keep_alive": 0}
