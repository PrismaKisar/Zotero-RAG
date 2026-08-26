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
    def __init__(self, results):
        self._results = results

    def open_connection(self):
        pass

    def close_connection(self):
        pass

    def search_batch(self, embeddings, threshold, result_limit, mode, pdf_hashes=None):
        return [self._results for _ in embeddings]


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
        query_variations=None: [
            RerankedChunk(chunk=c, retrieval_score=s, rerank_score=s)
            for c, s in candidates])
    rag.pdf_cache = types.SimpleNamespace(get_pdf_path=lambda h: None)
    return rag


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
