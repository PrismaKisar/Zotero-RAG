"""Tests for the abstractive reader's citation parsing and attribution.

Attribution is the whole point: the generated text is scored by Answer F1, but
evidence precision/recall are read off which chunk each Answer carries, so a
misparsed citation silently scores the wrong passage. A fake Ollama client keeps
these tests off the network.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

import generative_reader as gr
from generative_reader import GenerativeReader, parse_response
from models import Chunk, RerankedChunk


class FakeClient:
    """Serves a canned reply and records the prompt it was asked to complete."""

    def __init__(self, host=None, reply="", models=("llama3.2:3b",), fail=False):
        self.host = host
        self.reply = reply
        self.models = models
        self.fail = fail
        self.prompt = None

    def list(self):
        entries = [type("Entry", (), {"model": name})() for name in self.models]
        return type("Listing", (), {"models": entries})()

    def generate(self, model, prompt, options=None, keep_alive=None):
        self.prompt = prompt
        if self.fail:
            raise ConnectionError("ollama went away")
        return {"response": self.reply}


@pytest.fixture
def reader(monkeypatch):
    def build(reply="", **kwargs):
        monkeypatch.setattr(gr.ollama, "Client",
                            lambda host=None: FakeClient(host=host, reply=reply, **kwargs))
        return GenerativeReader()
    return build


def candidates(count):
    return [
        RerankedChunk(
            chunk=Chunk(text=f"passage {i}", page_number=i, chunk_index=i,
                        title="paper", pdf_hash="h", sentences=[(f"passage {i}", f"c{i}")]),
            retrieval_score=0.5,
            rerank_score=0.9 - i / 100,
        )
        for i in range(count)
    ]


def test_parse_response_collects_citations_in_order_without_duplicates():
    text, cited = parse_response("The model uses BERT [2] and also [1], plus [2] again.", 3)
    assert cited == [1, 0]
    assert "[" not in text


def test_parse_response_drops_indices_the_prompt_never_offered():
    _, cited = parse_response("Answer [1] and [9] and [0].", 2)
    assert cited == [0]


def test_parse_response_strips_a_trailing_sources_line():
    text, cited = parse_response("They use QASPER.\nSources: [1], [2]", 2)
    assert text == "They use QASPER."
    assert cited == [0, 1]


def test_parse_response_reports_a_declined_answer_as_empty():
    assert parse_response("NO ANSWER", 3) == ("", [])


def test_answers_are_attributed_to_every_cited_chunk(reader):
    engine = reader(reply="They evaluate on QASPER [1] and SQuAD [3].")
    answers = engine.extract_answers("Which datasets?", candidates(4), {})

    assert [a.context for a in answers] == ["passage 0", "passage 2"]
    # the same generated text is what Answer F1 scores, on every copy
    assert {a.text for a in answers} == {"They evaluate on QASPER and SQuAD."}
    assert [a.page_number for a in answers] == [0, 2]
    assert answers[0].sentence_coords == ["c0"]


def test_an_uncited_answer_falls_back_to_the_top_ranked_chunk(reader):
    engine = reader(reply="They evaluate on QASPER.")
    answers = engine.extract_answers("Which datasets?", candidates(3), {})
    assert [a.context for a in answers] == ["passage 0"]


def test_a_declined_answer_yields_nothing(reader):
    engine = reader(reply="NO ANSWER")
    assert engine.extract_answers("Which datasets?", candidates(3), {}) == []


def test_a_failed_generation_yields_nothing_instead_of_raising(reader):
    engine = reader(fail=True)
    assert engine.extract_answers("Which datasets?", candidates(3), {}) == []


def test_no_candidates_short_circuits(reader):
    engine = reader(reply="anything")
    assert engine.extract_answers("Which datasets?", [], {}) == []


def test_the_prompt_carries_only_the_top_chunks_numbered_from_one(reader):
    engine = reader(reply="Answer [1].")
    engine.max_context_chunks = 2
    engine.extract_answers("Which datasets?", candidates(5), {})

    prompt = engine.client.prompt
    assert "[1] passage 0" in prompt
    assert "[2] passage 1" in prompt
    assert "passage 2" not in prompt


def test_a_missing_ollama_model_fails_fast_with_the_pull_command(monkeypatch):
    monkeypatch.setattr(gr.ollama, "Client",
                        lambda host=None: FakeClient(host=host, models=()))
    with pytest.raises(RuntimeError, match="ollama pull"):
        GenerativeReader()
