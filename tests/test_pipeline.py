"""Stage timings recorded by ZoteroRAG.answer_question.

ZoteroRAG.__init__ loads three transformer models, so these build the instance
with __new__ and attach only the collaborators answer_question actually touches.
The point under test is where the mark() calls sit, not what the models do.
"""

import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from models import Answer, Chunk, RerankedChunk

from zotero_rag.pipeline import ZoteroRAG

STAGES = ("expansion", "retrieval", "rerank", "read")


class _Qdrant:
    def __init__(self, results, indexed=None):
        self._results = results
        # (pdf_hash, chunk_index) -> Chunk, standing in for the whole collection
        self._indexed = indexed or {}
        self.fetched = None

    def open_connection(self):
        pass

    def close_connection(self):
        pass

    def search_batch(self, embeddings, threshold, result_limit, mode, pdf_hashes=None):
        return [self._results for _ in embeddings]

    def fetch_chunks(self, ids):
        self.fetched = list(ids)
        return [self._indexed[i] for i in ids if i in self._indexed]


def _chunk(index):
    return Chunk(text=f"text-{index}", page_number=1, chunk_index=index,
                 title="T", pdf_hash="h")


def _rag(results, answers=None):
    rag = ZoteroRAG.__new__(ZoteroRAG)
    rag.last_candidates = []
    rag.last_reranked = []
    rag.last_stage_times = {}
    rag.query_colors = [(1.0, 1.0, 0.0)]
    rag.query_color_map = {}
    rag.qdrant_manager = _Qdrant(results)
    rag.embedding_manager = types.SimpleNamespace(encode_query=lambda q: [0.0])
    rag.qa_engine = types.SimpleNamespace(
        enable_question_expansion=False,
        extract_answers=lambda *a, **k: answers or [])
    rag.reranker = types.SimpleNamespace(
        rerank=lambda question, candidates, threshold, progress_callback=None,
        query_variations=None, order_by_retrieval=False: [
            RerankedChunk(chunk=c, retrieval_score=s, rerank_score=s)
            for c, s in candidates])
    rag.pdf_cache = types.SimpleNamespace(get_pdf_path=lambda h: None)
    return rag


def _neighbour_rag(span, indexed):
    rag = _rag([(_chunk(5), 0.9)])
    rag.qdrant_manager = _Qdrant([(_chunk(5), 0.9)], indexed=indexed)
    rag.answer_question("q", num_paraphrases=0, overrides={"retrieval_neighbours": span})
    return rag


def test_neighbour_expansion_is_off_unless_asked_for():
    """The default must stay 0: an unaccepted intervention does not ship."""
    rag = _rag([(_chunk(5), 0.9)])
    rag.answer_question("q", num_paraphrases=0)

    assert rag.qdrant_manager.fetched is None
    assert [c["chunk"].chunk_index for c in rag.last_candidates] == [5]


def test_neighbour_expansion_adds_the_adjacent_chunks():
    indexed = {("h", 4): _chunk(4), ("h", 6): _chunk(6)}
    rag = _neighbour_rag(1, indexed)

    assert sorted(c["chunk"].chunk_index for c in rag.last_candidates) == [4, 5, 6]


def test_a_neighbour_inherits_the_score_of_the_hit_that_pulled_it_in():
    """It was never scored against the query, so any other number is invented."""
    rag = _neighbour_rag(1, {("h", 4): _chunk(4)})

    by_index = {c["chunk"].chunk_index: c["retrieval_score"] for c in rag.last_candidates}
    assert by_index[4] == by_index[5] == 0.9


def test_a_neighbour_past_the_end_of_the_document_is_simply_missing():
    """Qdrant has no chunk 6, and that must not fail the query."""
    rag = _neighbour_rag(1, {("h", 4): _chunk(4)})

    assert sorted(c["chunk"].chunk_index for c in rag.last_candidates) == [4, 5]


def test_neighbour_expansion_never_asks_for_a_negative_index():
    rag = _rag([(_chunk(0), 0.9)])
    rag.qdrant_manager = _Qdrant([(_chunk(0), 0.9)], indexed={("h", 1): _chunk(1)})
    rag.answer_question("q", num_paraphrases=0, overrides={"retrieval_neighbours": 2})

    assert all(index >= 0 for _, index in rag.qdrant_manager.fetched)


def test_a_wider_span_reaches_further():
    indexed = {("h", i): _chunk(i) for i in (3, 4, 6, 7)}
    rag = _neighbour_rag(2, indexed)

    assert sorted(c["chunk"].chunk_index for c in rag.last_candidates) == [3, 4, 5, 6, 7]


def test_a_neighbour_already_retrieved_is_not_added_twice():
    hits = [(_chunk(5), 0.9), (_chunk(6), 0.4)]
    indexed = {("h", 4): _chunk(4), ("h", 6): _chunk(6), ("h", 7): _chunk(7)}
    rag = _rag(hits)
    rag.qdrant_manager = _Qdrant(hits, indexed=indexed)
    rag.answer_question("q", num_paraphrases=0, overrides={"retrieval_neighbours": 1})

    assert sorted(c["chunk"].chunk_index for c in rag.last_candidates) == [4, 5, 6, 7]
    assert ("h", 6) not in rag.qdrant_manager.fetched
    # and chunk 6 keeps its own retrieval score, not the neighbour's inherited one
    by_index = {c["chunk"].chunk_index: c["retrieval_score"] for c in rag.last_candidates}
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
    rag.answer_question("q", num_paraphrases=0, overrides={"retrieval_neighbours": 1})

    order = sorted(rag.last_candidates,
                   key=lambda c: (c["is_neighbour"], -c["retrieval_score"]))
    assert [c["chunk"].chunk_index for c in order] == [5, 6, 4, 7]


def test_without_neighbours_nothing_is_flagged():
    """The flag must not perturb the rows the campaign already measured."""
    rag = _rag([(_chunk(5), 0.9), (_chunk(6), 0.4)])
    rag.answer_question("q", num_paraphrases=0)

    assert [c["is_neighbour"] for c in rag.last_candidates] == [False, False]


def test_the_rerank_bypass_also_puts_neighbours_last():
    """With reranking off nothing rescores, so the bypass order is the order."""
    hits = [(_chunk(5), 0.9), (_chunk(6), 0.4)]
    indexed = {("h", 4): _chunk(4), ("h", 7): _chunk(7)}
    rag = _rag(hits)
    rag.qdrant_manager = _Qdrant(hits, indexed=indexed)
    rag.answer_question("q", num_paraphrases=0,
                        overrides={"retrieval_neighbours": 1,
                                   "rerank_enabled": False})

    assert [c.chunk.chunk_index for c in rag.last_reranked] == [5, 6, 4, 7]


def test_a_full_call_times_every_stage():
    rag = _rag([(_chunk(1), 0.9)], answers=[
        Answer(text="a", context="text-1", page_number=1, score=1.0,
               title="T", pdf_hash="h")])

    rag.answer_question("q", num_paraphrases=0)

    assert set(rag.last_stage_times) == set(STAGES)
    assert all(v >= 0.0 for v in rag.last_stage_times.values())


def test_retrieval_is_timed_even_when_it_comes_back_empty():
    """An empty retrieval returns early, and that path still has to be attributable.

    The mark sits in the ``finally`` for exactly this: if it sat after the early
    return, the one case worth diagnosing - retrieval found nothing - would be
    the one case with no retrieval timing.
    """
    rag = _rag([])

    assert rag.answer_question("q", num_paraphrases=0) == []
    assert set(rag.last_stage_times) == {"expansion", "retrieval"}


def test_stage_times_do_not_leak_from_the_previous_call():
    rag = _rag([(_chunk(1), 0.9)], answers=[])
    rag.answer_question("q", num_paraphrases=0)
    assert "rerank" in rag.last_stage_times

    rag.qdrant_manager = _Qdrant([])
    rag.answer_question("q", num_paraphrases=0)
    assert "rerank" not in rag.last_stage_times


def test_stage_times_sum_to_roughly_the_whole_call():
    """Guards against a stage of work sitting outside every mark() interval."""
    import time

    rag = _rag([(_chunk(1), 0.9)], answers=[])
    started = time.perf_counter()
    rag.answer_question("q", num_paraphrases=0)
    total = time.perf_counter() - started

    assert sum(rag.last_stage_times.values()) <= total
    assert sum(rag.last_stage_times.values()) >= total * 0.5
