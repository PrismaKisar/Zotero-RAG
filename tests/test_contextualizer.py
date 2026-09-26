"""Chunk contextualization: a document summary, then one context line per chunk.

It is best-effort by design: a failed summary leaves the prompt without one, a
failed chunk is indexed raw. These pin that behaviour with a stub Ollama client.
"""

import sys
from pathlib import Path

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


def test_a_failed_chunk_is_kept_raw(monkeypatch):
    contextualizer = _contextualizer(monkeypatch, ["summary", RuntimeError("down"), "ctx b"])
    assert contextualizer.contextualize("doc", ["a", "b"]) == ["a", "ctx b\n\nb"]


def test_a_failed_summary_still_contextualizes_the_chunks(monkeypatch):
    contextualizer = _contextualizer(monkeypatch, [RuntimeError("down"), "ctx a"])
    assert contextualizer.contextualize("doc", ["a"]) == ["ctx a\n\na"]


def test_an_empty_context_line_leaves_the_chunk_as_it_was(monkeypatch):
    contextualizer = _contextualizer(monkeypatch, ["summary", "  "])
    assert contextualizer.contextualize("doc", ["a"]) == ["a"]


def test_the_model_is_unloaded_once_the_document_is_done(monkeypatch):
    contextualizer = _contextualizer(monkeypatch, ["summary", "ctx a"])
    contextualizer.contextualize("doc", ["a"])
    assert contextualizer.client.calls[-1] == {
        "model": cx.DEFAULT_CONTEXT_MODEL, "prompt": None, "keep_alive": 0}
