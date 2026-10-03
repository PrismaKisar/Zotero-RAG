import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from hybrid_reader import HybridReader
from models import Answer, Chunk, RerankedChunk


def _candidate(text):
    return RerankedChunk(Chunk(text=text, page_number=0, chunk_index=0, title="t",
                               pdf_hash="h"), retrieval_score=0.5, rerank_score=0.9)


CANDIDATES = [_candidate("alpha chunk"), _candidate("beta chunk"), _candidate("gamma chunk")]


def _answer(text, context, start=0, end=None):
    return Answer(text=text, context=context, page_number=0, title="t", start_char=start,
                  end_char=len(context) if end is None else end)


class _Generative:
    reader_kind = "generative:off:one_sentence:8"
    generation_failures = 0

    def __init__(self, cited):
        self.cited = cited
        self.last_quote_stats = {"cited": 0, "matched": 0}

    def extract_answers(self, question, candidates, config, color=(1, 1, 0),
                        progress_callback=None):
        return [_answer("generated", context) for context in self.cited]


class _Extractive:
    reader_kind = "deberta"

    def __init__(self, spans):
        self.spans = spans
        self.seen = None

    def extract_answers(self, question, candidates, config, color=(1, 1, 0),
                        progress_callback=None):
        self.seen = [c.chunk.text for c in candidates]
        return [s for s in self.spans if s.context in self.seen]


def test_the_extractive_reader_only_reads_the_chunks_the_answer_cites():
    extractive = _Extractive([])
    HybridReader(_Generative(["beta chunk"]), extractive).extract_answers("q", CANDIDATES, {})
    assert extractive.seen == ["beta chunk"]


def test_the_mark_is_the_span_and_the_text_is_the_generated_answer():
    extractive = _Extractive([_answer("beta", "beta chunk", 0, 4)])
    answers = HybridReader(_Generative(["beta chunk"]), extractive).extract_answers(
        "q", CANDIDATES, {})
    assert [(a.text, a.start_char, a.end_char) for a in answers] == [("generated", 0, 4)]


def test_a_cited_chunk_without_a_span_keeps_the_whole_chunk():
    extractive = _Extractive([_answer("beta", "beta chunk", 0, 4)])
    answers = HybridReader(_Generative(["alpha chunk", "beta chunk"]), extractive).extract_answers(
        "q", CANDIDATES, {})
    assert [(a.context, a.end_char) for a in answers] == [("alpha chunk", 11), ("beta chunk", 4)]


def test_a_declined_answer_runs_no_extraction():
    extractive = _Extractive([])
    assert HybridReader(_Generative([]), extractive).extract_answers("q", CANDIDATES, {}) == []
    assert extractive.seen is None


def test_the_identity_and_the_failure_count_are_the_generative_readers():
    generative = types.SimpleNamespace(reader_kind="g", last_quote_stats={"cited": 1},
                                       generation_failures=3)
    reader = HybridReader(generative, _Extractive([]))
    assert reader.reader_kind == "hybrid:g:deberta"
    assert reader.generation_failures == 3
    reader.generation_failures = 0
    assert generative.generation_failures == 0
