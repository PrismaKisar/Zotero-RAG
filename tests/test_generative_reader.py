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
from generative_reader import (
    GenerativeReader,
    locate_quote,
    parse_quotes,
    parse_response,
)
from models import Chunk, RerankedChunk


class FakeClient:
    """Serves a canned reply and records the prompt it was asked to complete."""

    def __init__(self, host=None, reply="", models=None, fail=False, fail_times=0):
        self.host = host
        self.reply = reply
        # Fails the first ``fail_times`` calls and then succeeds, which is the
        # shape of the Ollama runner dying and being restarted under us.
        self.fail_times = fail_times
        self.calls = 0
        # Tracks the default rather than naming a model: the reader refuses to
        # start against a model Ollama does not serve, so a hardcoded name here
        # turns every test in this file red the next time the default changes.
        self.models = (gr.DEFAULT_GENERATIVE_MODEL,) if models is None else models
        self.fail = fail
        self.prompt = None
        self.think = None

    def list(self):
        entries = [type("Entry", (), {"model": name})() for name in self.models]
        return type("Listing", (), {"models": entries})()

    def generate(self, model, prompt, think=None, options=None, keep_alive=None):
        self.prompt = prompt
        self.think = think
        self.calls += 1
        if self.fail or self.calls <= self.fail_times:
            raise ConnectionError("ollama went away")
        return {"response": self.reply}


@pytest.fixture
def reader(monkeypatch):
    def build(reply="", quote=gr.QUOTE_OFF, max_context_chunks=None, **kwargs):
        monkeypatch.setattr(gr.ollama, "Client",
                            lambda host=None: FakeClient(host=host, reply=reply, **kwargs))
        # Reader settings are named explicitly; **kwargs stays the FakeClient's,
        # so a typo in a reader setting fails here instead of being swallowed as
        # client config and silently testing the default.
        reader_kwargs = ({} if max_context_chunks is None
                         else {"max_context_chunks": max_context_chunks})
        return GenerativeReader(citation_quote=quote, **reader_kwargs)
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


def test_the_reasoning_mode_is_pinned_off(reader):
    """Left to the server's default, Qwen3.5 may deliberate before answering.

    That would multiply latency and make the reader a different component from
    the one measured, so the flag is sent explicitly on every call rather than
    left unset - and unset is what this test would catch.
    """
    engine = reader(reply="Answer [1].")
    engine.extract_answers("Which datasets?", candidates(3), {})

    assert engine.client.think is False


TWO_SENTENCES = "Alpha holds under mild assumptions. Beta follows from alpha."


def quotable(text=TWO_SENTENCES):
    """One candidate whose chunk has two locatable sentences."""
    sentences = [("Alpha holds under mild assumptions.", "cA"),
                 ("Beta follows from alpha.", "cB")]
    return [RerankedChunk(
        chunk=Chunk(text=text, page_number=1, chunk_index=0, title="paper",
                    pdf_hash="h", sentences=sentences),
        retrieval_score=0.5, rerank_score=0.9)]


REPLY_WITH_QUOTE = ('Beta is implied [1].\n'
                    'Quotes:\n'
                    '[1] "Beta follows from alpha."')


def test_parse_quotes_reads_the_block_after_the_marker():
    assert parse_quotes(REPLY_WITH_QUOTE, 1) == {0: "Beta follows from alpha."}


def test_parse_quotes_ignores_a_citation_in_the_answer_body():
    """An answer line starting "[2] ..." is a citation, not a quoted sentence."""
    raw = '[2] is where this comes from.\nQuotes:\n[2] "Beta follows from alpha."'
    assert parse_quotes(raw, 3) == {1: "Beta follows from alpha."}


def test_parse_quotes_drops_indices_the_prompt_never_offered():
    assert parse_quotes('Quotes:\n[9] "nope"\n[1] "yes"', 2) == {0: "yes"}


def test_parse_quotes_is_empty_when_the_model_never_wrote_the_block():
    assert parse_quotes("Just an answer [1].", 2) == {}


def test_parse_response_strips_the_quote_block_from_the_answer_text():
    """Left in, the quoted sentences would be scored by Answer F1 as answer text."""
    text, cited = parse_response(REPLY_WITH_QUOTE, 1)
    assert text == "Beta is implied."
    assert cited == [0]


def test_locate_quote_finds_a_verbatim_sentence():
    assert locate_quote("Beta follows from alpha.", TWO_SENTENCES) == (36, 60)


def test_locate_quote_ignores_how_the_extractor_broke_the_lines():
    """PDF extraction splits sentences across lines; that is not a bad quote."""
    assert locate_quote("Beta follows from alpha.",
                        "Alpha holds.\nBeta follows\n  from alpha.") is not None


def test_locate_quote_reports_a_sentence_that_is_not_there():
    assert locate_quote("Gamma refutes beta.", TWO_SENTENCES) is None


def test_locate_quote_does_not_forgive_rewording():
    """The metric measures verbatim reproduction, so a near-miss must not count."""
    assert locate_quote("Beta follows from Alpha.", TWO_SENTENCES) is None


def test_a_verified_quote_narrows_the_highlight_to_that_sentence(reader):
    engine = reader(reply=REPLY_WITH_QUOTE, quote=gr.QUOTE_LENIENT)
    answer = engine.extract_answers("Why beta?", quotable(), {})[0]

    assert (answer.start_char, answer.end_char) == (36, 60)
    # the coordinates have to narrow too, or the page still shows the paragraph
    assert answer.sentence_coords == ["cB"]


def test_lenient_keeps_an_unverifiable_citation_and_marks_the_whole_chunk(reader):
    engine = reader(reply='Something [1].\nQuotes:\n[1] "Gamma refutes beta."',
                    quote=gr.QUOTE_LENIENT)
    answer = engine.extract_answers("Why beta?", quotable(), {})[0]

    assert (answer.start_char, answer.end_char) == (0, len(TWO_SENTENCES))
    assert answer.sentence_coords == ["cA", "cB"]


def test_strict_discards_a_citation_whose_quote_is_not_in_the_chunk(reader):
    engine = reader(reply='Something [1].\nQuotes:\n[1] "Gamma refutes beta."',
                    quote=gr.QUOTE_STRICT)
    assert engine.extract_answers("Why beta?", quotable(), {}) == []


def test_strict_keeps_a_citation_whose_quote_checks_out(reader):
    engine = reader(reply=REPLY_WITH_QUOTE, quote=gr.QUOTE_STRICT)
    assert len(engine.extract_answers("Why beta?", quotable(), {})) == 1


def test_the_match_rate_counts_every_citation_not_only_the_surviving_ones(reader):
    """Strict drops the failures, so the denominator cannot be the answers."""
    reply = ('Both [1] and [2].\nQuotes:\n'
             '[1] "Beta follows from alpha."\n[2] "Gamma refutes beta."')
    engine = reader(reply=reply, quote=gr.QUOTE_STRICT)
    engine.extract_answers("Why beta?", quotable() + quotable(), {})

    assert engine.last_quote_stats == {"cited": 2, "matched": 1}


def test_the_verified_chunks_are_exposed_for_offline_policy_scoring(reader):
    """One lenient generation has to be scorable under the strict policy too."""
    reply = ('Both [1] and [2].\nQuotes:\n'
             '[1] "Beta follows from alpha."\n[2] "Gamma refutes beta."')
    engine = reader(reply=reply, quote=gr.QUOTE_LENIENT)
    engine.extract_answers("Why beta?", quotable() + quotable("Other text."), {})

    assert engine.last_verified_contexts == {TWO_SENTENCES}


def test_verified_chunks_do_not_leak_from_the_previous_question(reader):
    engine = reader(reply=REPLY_WITH_QUOTE, quote=gr.QUOTE_LENIENT)
    engine.extract_answers("Why beta?", quotable(), {})
    engine.extract_answers("Why beta?", [], {})

    assert engine.last_verified_contexts == set()


def test_the_off_arm_reports_no_citations_to_have_verified(reader):
    """A rate of 0/n here would score the control as fabricating every citation."""
    engine = reader(reply="Answer [1].")
    engine.extract_answers("Why beta?", quotable(), {})

    assert engine.last_quote_stats == {"cited": 0, "matched": 0}


def test_quote_stats_do_not_leak_from_the_previous_question(reader):
    engine = reader(reply=REPLY_WITH_QUOTE, quote=gr.QUOTE_LENIENT)
    engine.extract_answers("Why beta?", quotable(), {})
    engine.extract_answers("Why beta?", [], {})

    assert engine.last_quote_stats == {"cited": 0, "matched": 0}


def test_the_quote_rule_is_absent_from_the_prompt_when_the_mode_is_off(reader):
    """The off arm must be the reader already measured, prompt included."""
    engine = reader(reply="Answer [1].")
    engine.extract_answers("Why beta?", quotable(), {})
    assert "Quotes:" not in engine.client.prompt

    engine = reader(reply="Answer [1].", quote=gr.QUOTE_LENIENT)
    engine.extract_answers("Why beta?", quotable(), {})
    assert "Quotes:" in engine.client.prompt


def test_an_unknown_quote_mode_fails_at_construction(reader):
    with pytest.raises(ValueError, match="citation_quote"):
        reader(quote="verbatim")


def test_every_answer_style_produces_its_own_rule(reader, monkeypatch):
    """A style silently falling back to another would compare a reader to itself."""
    rules = set()
    for style in gr.ANSWER_STYLES:
        monkeypatch.setattr(gr.ollama, "Client",
                            lambda host=None: FakeClient(host=host, reply="Answer [1]."))
        engine = GenerativeReader(answer_style=style)
        engine.extract_answers("Why beta?", quotable(), {})
        rules.add(engine.client.prompt)
    assert len(rules) == len(gr.ANSWER_STYLES)


def test_the_default_answer_style_is_the_rule_as_shipped(reader):
    """The axis has to vary against the reader the campaign already measured."""
    engine = reader(reply="Answer [1].")
    engine.extract_answers("Why beta?", quotable(), {})
    assert "at most two sentences" in engine.client.prompt


def test_the_phrase_style_forbids_a_sentence(reader):
    engine = reader(reply="Answer [1].")
    engine.answer_style = "phrase"
    engine.extract_answers("Why beta?", quotable(), {})
    assert "Do not\n  write a sentence" in engine.client.prompt


def test_an_unknown_answer_style_fails_at_construction(monkeypatch):
    monkeypatch.setattr(gr.ollama, "Client", lambda host=None: FakeClient(host=host))
    with pytest.raises(ValueError, match="answer_style"):
        GenerativeReader(answer_style="haiku")


def test_the_reader_identity_carries_every_setting_that_changes_the_prompt(monkeypatch):
    """Two prompts are two readers; a shared load would score one of them twice."""
    monkeypatch.setattr(gr.ollama, "Client", lambda host=None: FakeClient(host=host))
    plain = GenerativeReader()
    brief = GenerativeReader(answer_style="phrase")
    wide = GenerativeReader(max_context_chunks=30)

    assert len({plain.reader_kind, brief.reader_kind, wide.reader_kind}) == 3


def test_a_generation_that_fails_once_is_retried(reader):
    """The Ollama runner dies and restarts; the next call finds it back up.

    Without the retry the blip enters the results as an unanswered question,
    which a benchmark reports as a property of the configuration.
    """
    engine = reader(reply="Answer [1].", fail_times=1)

    answers = engine.extract_answers("Which datasets?", candidates(3), {})

    assert engine.client.calls == 2
    assert len(answers) == 1
    assert engine.generation_failures == 0


def test_a_generation_that_keeps_failing_is_counted_not_retried_forever(reader):
    """Two failures means the service is down, not blipping."""
    engine = reader(reply="Answer [1].", fail_times=5)

    assert engine.extract_answers("Which datasets?", candidates(3), {}) == []
    assert engine.client.calls == 2
    assert engine.generation_failures == 1


def test_generation_failures_accumulate_across_questions(reader):
    """The count is what tells a run how many of its zeros came from the service."""
    engine = reader(reply="Answer [1].", fail_times=99)

    engine.extract_answers("q1", candidates(3), {})
    engine.extract_answers("q2", candidates(3), {})

    assert engine.generation_failures == 2


def test_an_empty_context_window_fails_at_construction(monkeypatch):
    """Zero chunks answers every question from an empty prompt, and scores as
    an ignorant reader rather than as the misconfiguration it is."""
    monkeypatch.setattr(gr.ollama, "Client", lambda host=None: FakeClient(host=host))
    with pytest.raises(ValueError, match="max_context_chunks"):
        GenerativeReader(max_context_chunks=0)


def test_the_context_window_truncates_the_candidates_reaching_the_prompt(reader):
    """The axis has to bound the prompt, not just the identity string.

    A citation past the window must also be dropped rather than indexed into the
    untruncated list, or [4] would silently attribute the answer to a passage
    the model was never shown.
    """
    engine = reader(reply="Answer [1] and [4].", max_context_chunks=3)

    engine.extract_answers("Which datasets?", candidates(10), {})
    prompt = engine.client.prompt

    assert "passage 2" in prompt
    assert "passage 3" not in prompt


def test_a_missing_ollama_model_fails_fast_with_the_pull_command(monkeypatch):
    monkeypatch.setattr(gr.ollama, "Client",
                        lambda host=None: FakeClient(host=host, models=()))
    with pytest.raises(RuntimeError, match="ollama pull"):
        GenerativeReader()
