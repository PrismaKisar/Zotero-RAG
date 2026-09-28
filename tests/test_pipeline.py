"""Stage timings recorded by ZoteroRAG.answer_question.

ZoteroRAG.__init__ loads three transformer models, so these build the instance
with from_components and pass only the collaborators the call actually touches.
The point under test is where the mark() calls sit, not what the models do.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from models import Answer, Chunk, RerankedChunk

from zotero_rag.pipeline import ZoteroRAG

STAGES = ("retrieval", "rerank", "read")


class _Qdrant:
    def __init__(self, results, indexed=None):
        self._results = results
        # (pdf_hash, chunk_index) -> Chunk, standing in for the whole collection
        self._indexed = indexed or {}
        self.fetched = None

    def search(self, embedding, result_limit, mode, pdf_hashes=None):
        return list(self._results)

    def fetch_chunks(self, ids):
        self.fetched = list(ids)
        return [self._indexed[i] for i in ids if i in self._indexed]


def _chunk(index):
    return Chunk(text=f"text-{index}", page_number=1, chunk_index=index,
                 title="T", pdf_hash="h")


def _rag(results, answers=None):
    return ZoteroRAG.from_components(
        qdrant_manager=_Qdrant(results),
        embedding_manager=types.SimpleNamespace(encode_query=lambda q: [0.0]),
        reader=types.SimpleNamespace(extract_answers=lambda *a, **k: answers or []),
        reranker=types.SimpleNamespace(
            rerank=lambda question, candidates, gate=0.0, margin=1.0, progress_callback=None,
            order_by_retrieval=False: [
                RerankedChunk(chunk=c, retrieval_score=s, rerank_score=s)
                for c, s in candidates]),
        pdf_cache=types.SimpleNamespace(get_pdf_path=lambda h: None))


def _neighbour_rag(span, indexed):
    rag = _rag([(_chunk(5), 0.9)])
    rag.qdrant_manager = _Qdrant([(_chunk(5), 0.9)], indexed=indexed)
    rag.answer_question("q", overrides={"retrieval_neighbours": span})
    return rag


def test_neighbour_expansion_is_off_unless_asked_for():
    """The default must stay 0: an unaccepted intervention does not ship."""
    rag = _rag([(_chunk(5), 0.9)])
    rag.answer_question("q")

    assert rag.qdrant_manager.fetched is None
    assert [c.chunk.chunk_index for c in rag.last_trace.retrieved] == [5]


def test_tied_retrieval_scores_are_broken_by_chunk_id_not_by_arrival_order():
    """Qdrant returns equal RRF scores in no stable order; the pipeline must not."""
    rag = _rag([(_chunk(9), 0.5), (_chunk(2), 0.5), (_chunk(7), 0.8)])
    rag.answer_question("q")

    assert [c.chunk.chunk_index for c in rag.last_trace.retrieved] == [7, 2, 9]


def test_neighbour_expansion_adds_the_adjacent_chunks():
    indexed = {("h", 4): _chunk(4), ("h", 6): _chunk(6)}
    rag = _neighbour_rag(1, indexed)

    assert sorted(c.chunk.chunk_index for c in rag.last_trace.retrieved) == [4, 5, 6]


def test_a_neighbour_inherits_the_score_of_the_hit_that_pulled_it_in():
    """It was never scored against the query, so any other number is invented."""
    rag = _neighbour_rag(1, {("h", 4): _chunk(4)})

    by_index = {c.chunk.chunk_index: c.retrieval_score for c in rag.last_trace.retrieved}
    assert by_index[4] == by_index[5] == 0.9


def test_a_neighbour_past_the_end_of_the_document_is_simply_missing():
    """Qdrant has no chunk 6, and that must not fail the query."""
    rag = _neighbour_rag(1, {("h", 4): _chunk(4)})

    assert sorted(c.chunk.chunk_index for c in rag.last_trace.retrieved) == [4, 5]


def test_neighbour_expansion_never_asks_for_a_negative_index():
    rag = _rag([(_chunk(0), 0.9)])
    rag.qdrant_manager = _Qdrant([(_chunk(0), 0.9)], indexed={("h", 1): _chunk(1)})
    rag.answer_question("q", overrides={"retrieval_neighbours": 2})

    assert all(index >= 0 for _, index in rag.qdrant_manager.fetched)


def test_a_wider_span_reaches_further():
    indexed = {("h", i): _chunk(i) for i in (3, 4, 6, 7)}
    rag = _neighbour_rag(2, indexed)

    assert sorted(c.chunk.chunk_index for c in rag.last_trace.retrieved) == [3, 4, 5, 6, 7]


def test_a_neighbour_already_retrieved_is_not_added_twice():
    hits = [(_chunk(5), 0.9), (_chunk(6), 0.4)]
    indexed = {("h", 4): _chunk(4), ("h", 6): _chunk(6), ("h", 7): _chunk(7)}
    rag = _rag(hits)
    rag.qdrant_manager = _Qdrant(hits, indexed=indexed)
    rag.answer_question("q", overrides={"retrieval_neighbours": 1})

    assert sorted(c.chunk.chunk_index for c in rag.last_trace.retrieved) == [4, 5, 6, 7]
    assert ("h", 6) not in rag.qdrant_manager.fetched
    # and chunk 6 keeps its own retrieval score, not the neighbour's inherited one
    by_index = {c.chunk.chunk_index: c.retrieval_score for c in rag.last_trace.retrieved}
    assert by_index[6] == 0.4


def test_a_neighbour_never_outranks_a_genuine_hit():
    """The defect the first neighbour campaign measured with, now fixed.

    Chunk 6 was retrieved and scored 0.4. Chunk 4 was never scored at all and
    inherits 0.9 from chunk 5. Ranking on the score alone put the unscored
    chunk above the scored one and pushed real hits out of the top ten, so the
    recall the campaign reported was partly an artefact of the ordering.
    """
    hits = [(_chunk(5), 0.9), (_chunk(6), 0.4)]
    indexed = {("h", 4): _chunk(4), ("h", 7): _chunk(7)}
    rag = _rag(hits)
    rag.qdrant_manager = _Qdrant(hits, indexed=indexed)
    rag.answer_question("q", overrides={"retrieval_neighbours": 1})

    assert [c.chunk.chunk_index for c in rag.last_trace.retrieved] == [5, 6, 4, 7]
    assert [c.is_neighbour for c in rag.last_trace.retrieved] == [False, False, True, True]


def test_without_neighbours_nothing_is_flagged():
    """The flag must not perturb the rows the campaign already measured."""
    rag = _rag([(_chunk(5), 0.9), (_chunk(6), 0.4)])
    rag.answer_question("q")

    assert [c.is_neighbour for c in rag.last_trace.retrieved] == [False, False]


def test_the_rerank_bypass_also_puts_neighbours_last():
    """With reranking off nothing rescores, so the bypass order is the order."""
    hits = [(_chunk(5), 0.9), (_chunk(6), 0.4)]
    indexed = {("h", 4): _chunk(4), ("h", 7): _chunk(7)}
    rag = _rag(hits)
    rag.qdrant_manager = _Qdrant(hits, indexed=indexed)
    rag.answer_question("q",
                        overrides={"retrieval_neighbours": 1,
                                   "rerank_enabled": False})

    assert [c.chunk.chunk_index for c in rag.last_trace.reranked] == [5, 6, 4, 7]


def test_a_full_call_times_every_stage():
    rag = _rag([(_chunk(1), 0.9)], answers=[
        Answer(text="a", context="text-1", page_number=1, score=1.0,
               title="T", pdf_hash="h")])

    rag.answer_question("q")

    assert set(rag.last_trace.stage_times) == set(STAGES)
    assert all(v >= 0.0 for v in rag.last_trace.stage_times.values())


def test_retrieval_is_timed_even_when_it_comes_back_empty():
    """An empty retrieval returns early, and that path still has to be attributable.

    The mark sits in the ``finally`` for exactly this: if it sat after the early
    return, the one case worth diagnosing - retrieval found nothing - would be
    the one case with no retrieval timing.
    """
    rag = _rag([])

    assert rag.answer_question("q") == []
    assert set(rag.last_trace.stage_times) == {"retrieval"}


def test_stage_times_do_not_leak_from_the_previous_call():
    rag = _rag([(_chunk(1), 0.9)], answers=[])
    rag.answer_question("q")
    assert "rerank" in rag.last_trace.stage_times

    rag.qdrant_manager = _Qdrant([])
    rag.answer_question("q")
    assert "rerank" not in rag.last_trace.stage_times


def test_stage_times_sum_to_roughly_the_whole_call():
    """Guards against a stage of work sitting outside every mark() interval."""
    import time

    rag = _rag([(_chunk(1), 0.9)], answers=[])
    started = time.perf_counter()
    rag.answer_question("q")
    total = time.perf_counter() - started

    assert sum(rag.last_trace.stage_times.values()) <= total
    assert sum(rag.last_trace.stage_times.values()) >= total * 0.5


def test_upsert_with_qdrant_down_raises_instead_of_failing_every_pdf():
    """Qdrant down is one error, not one recorded failure per PDF."""
    class _Down:
        def ping(self):
            raise ConnectionError("Qdrant service not reachable")

    rag = ZoteroRAG.from_components(
        qdrant_manager=_Down(),
        pdf_cache=types.SimpleNamespace(remove_pdf=lambda h: True),
        pdf_processor=types.SimpleNamespace(remove_tei_cache=lambda h: None))
    item = types.SimpleNamespace(title="T", pdf_hash="h", newly_cached=True)
    with pytest.raises(ConnectionError):
        rag.upsert_pdfs([item])


class _IndexQdrant:
    """Registry by title and hash; records what upsert_chunks receives."""

    def __init__(self, titles=None, indexed=()):
        self.titles = dict(titles or {})
        self.indexed = set(indexed)
        self.upserted = None

    def ping(self):
        pass

    def find_pdf_hash_by_title(self, title, filter_by_model=True):
        return self.titles.get(title)

    def is_pdf_indexed(self, pdf_hash):
        return pdf_hash in self.indexed

    def register_pdf_title(self, pdf_hash, title):
        return title

    def upsert_chunks(self, chunks, dense_embeddings, sparse_embeddings,
                      progress_callback=None):
        self.upserted = chunks
        return len(chunks), []


class _Processor:
    def __init__(self, documents):
        # pdf_hash -> (document_text, [(section, text), ...]); a missing hash fails
        self.documents = documents
        self.tei_removed = []

    def extract_text_chunks(self, pdf_hash, pdf_title=None):
        document_text, sections = self.documents[pdf_hash]
        return [types.SimpleNamespace(text=text, page_number=1, chunk_index=i,
                                      section=section, sentences=[])
                for i, (section, text) in enumerate(sections)], document_text

    def remove_tei_cache(self, pdf_hash):
        self.tei_removed.append(pdf_hash)


class _Contextualizer:
    def contextualize(self, document_text, all_texts):
        return [f"ctx|{t}" for t in all_texts]


class _Embedder:
    def __init__(self):
        self.encoded = None

    def encode_chunks(self, progress_callback, all_texts):
        self.encoded = list(all_texts)
        return {"dense": [[0.0]] * len(all_texts), "sparse": [{}] * len(all_texts)}


class _Cache:
    def __init__(self):
        self.removed = []

    def remove_pdf(self, pdf_hash):
        self.removed.append(pdf_hash)
        return True


def _indexing_rag(qdrant, processor, contextualizer=None):
    return ZoteroRAG.from_components(qdrant_manager=qdrant, pdf_processor=processor,
                                     embedding_manager=_Embedder(), pdf_cache=_Cache(),
                                     contextualizer=contextualizer)


def _pdf(pdf_hash, title, newly_cached=True):
    return types.SimpleNamespace(pdf_hash=pdf_hash, title=title, newly_cached=newly_cached)


def test_upsert_skips_duplicates_and_indexed_pdfs_and_rolls_back_failures():
    qdrant = _IndexQdrant(titles={"Taken": "other"}, indexed={"old"})
    processor = _Processor({"new": ("doc", [("intro", "a")])})
    rag = _indexing_rag(qdrant, processor)

    result = rag.upsert_pdfs([_pdf("dup", "Taken"), _pdf("old", "Old"),
                              _pdf("new", "New"), _pdf("broken", "Broken")])

    assert result.duplicate_titles == ["Taken"]
    assert [e["pdf_hash"] for e in result.already_indexed] == ["old"]
    assert [f["pdf_hash"] for f in result.failed_pdfs] == ["broken"]
    assert result.processed_pdfs == 1 and result.indexed_chunks == 1
    assert [c.text for c in qdrant.upserted] == ["a"]
    # the duplicate is removed once, at skip time; the failure by the rollback
    assert rag.pdf_cache.removed == ["dup", "broken"]
    assert processor.tei_removed == ["broken"]


def test_upsert_raises_when_nothing_could_be_extracted():
    rag = _indexing_rag(_IndexQdrant(), _Processor({"e": ("doc", [])}))
    with pytest.raises(ValueError, match="No text could be extracted"):
        rag.upsert_pdfs([_pdf("e", "Empty")])


def test_contextualization_embeds_the_contextualized_texts_but_indexes_the_originals():
    processor = _Processor({
        "ok": ("doc ok", [("intro", "a"), ("intro", "b")]),
        "blank": ("   ", [("intro", "c")]),
    })
    qdrant = _IndexQdrant()
    rag = _indexing_rag(qdrant, processor, _Contextualizer())

    rag.upsert_pdfs([_pdf("ok", "Ok"), _pdf("blank", "Blank")])

    # a PDF without document text keeps its original chunks
    assert rag.embedding_manager.encoded == ["ctx|a", "ctx|b", "c"]
    assert [c.text for c in qdrant.upserted] == ["a", "b", "c"]


def test_contextualization_is_skipped_when_disabled():
    rag = _indexing_rag(_IndexQdrant(), _Processor({"ok": ("doc", [("intro", "a")])}))
    rag.upsert_pdfs([_pdf("ok", "Ok")])
    assert rag.embedding_manager.encoded == ["a"]


class _ReversedFetch(_Qdrant):
    """Qdrant returns fetched points in no promised order."""

    def fetch_chunks(self, ids):
        return list(reversed(super().fetch_chunks(ids)))


def _scattered_neighbours_rag():
    hits = [(_chunk(9), 0.4), (_chunk(5), 0.9)]
    indexed = {("h", i): _chunk(i) for i in (4, 6, 8, 10)}
    rag = _rag(hits)
    rag.qdrant_manager = _ReversedFetch(hits, indexed=indexed)
    seen = []
    rerank = rag.reranker.rerank
    rag.reranker = types.SimpleNamespace(
        rerank=lambda question, candidates, *a, **k: seen.extend(candidates) or rerank(
            question, candidates, *a, **k))
    rag.answer_question("q", overrides={"retrieval_neighbours": 1})
    return rag, seen


def test_the_trace_orders_neighbours_best_first_whatever_the_fetch_order():
    rag, _ = _scattered_neighbours_rag()
    assert [c.chunk.chunk_index for c in rag.last_trace.retrieved] == [5, 9, 6, 4, 10, 8]


def test_the_reranker_sees_neighbours_in_fetch_order():
    """Rerank ties keep their input order, so this order reaches the results."""
    _, seen = _scattered_neighbours_rag()
    assert [c.chunk_index for c, _ in seen] == [5, 9, 10, 8, 6, 4]
