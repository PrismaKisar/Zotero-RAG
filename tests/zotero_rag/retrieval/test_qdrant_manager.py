"""Unit tests for QdrantManager's query construction.

Only the pure request-building half is covered here: it needs no server, and it
is where a scoped search can go silently wrong - a filter that lands in the
wrong place returns fewer results rather than an error.
"""

from typing import ClassVar

import pytest
from qdrant_client import models as qmodels

from zotero_rag.models import Chunk, Reference
from zotero_rag.retrieval.qdrant_manager import QdrantManager

DENSE = [0.1] * 4
SPARSE = qmodels.SparseVector(indices=[1], values=[1.0])


def build(mode, pdf_hashes=None):
    return QdrantManager._build_query_requests(
        DENSE, SPARSE, result_limit=30, mode=mode, pdf_hashes=pdf_hashes)


def hashes_in(flt):
    assert flt is not None
    return flt.must[0].match.any


def test_no_filter_when_no_documents_are_named():
    assert QdrantManager._pdf_hash_filter(None) is None
    assert QdrantManager._pdf_hash_filter([]) is None


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_every_branch_scopes_to_the_named_documents(mode):
    """A filter on only some branches would let the others fill their depth
    with out-of-scope chunks."""
    assert all(hashes_in(r.filter) == ["a", "b"] for r in build(mode, ["a", "b"]))


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_every_mode_searches_the_whole_corpus_by_default(mode):
    assert all(r.filter is None for r in build(mode))


def test_hybrid_is_one_request_per_vector():
    assert [r.using for r in build("hybrid")] == ["", "text-sparse"]


def test_each_branch_is_fetched_deeper_than_the_cut_to_settle_ties():
    assert all(r.limit == 60 for r in build("hybrid"))


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_no_request_cuts_by_cosine_score(mode):
    """retrieval_threshold was removed: result_limit is the only stage-1 cut."""
    assert all(r.score_threshold is None for r in build(mode))


def test_dense_search_is_exact_not_approximate():
    """A library past Qdrant's indexing threshold must not turn approximate."""
    assert build("dense")[0].params.exact is True
    assert build("hybrid")[0].params.exact is True


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown retrieval_mode"):
        build("bm25")


def _hit(index, score):
    return (Chunk(text=str(index), page_number=0, chunk_index=index, title="t", pdf_hash="h"),
            score)


def test_a_branch_breaks_ties_by_chunk_id_not_arrival_order():
    ranked = QdrantManager.rank_branch([_hit(9, 0.5), _hit(2, 0.5), _hit(7, 0.8)], limit=2)
    assert [c.chunk_index for c, _ in ranked] == [7, 2]


def test_fusion_scores_as_qdrant_does():
    """Qdrant's RRF: k=2, ranks from zero, so first in both branches is 1.0."""
    fused = QdrantManager.rrf_fuse([[_hit(1, 0.9), _hit(2, 0.8)],
                                    [_hit(1, 5.0), _hit(3, 4.0)]], limit=3)
    assert [(c.chunk_index, round(s, 4)) for c, s in fused] == [(1, 1.0), (2, 0.3333), (3, 0.3333)]


def test_fused_ties_are_broken_by_chunk_id_at_the_cut():
    fused = QdrantManager.rrf_fuse([[_hit(5, 0.9)], [_hit(4, 0.9)]], limit=1)
    assert [c.chunk_index for c, _ in fused] == [4]


class _FakeClient:
    """Stands in for QdrantClient: counts connects and collection creations."""

    instances: ClassVar[list] = []

    def __init__(self, url, fail=None):
        self.collections = set()
        self.fail = fail
        self.closed = False
        _FakeClient.instances.append(self)

    def get_collections(self):
        if self.fail:
            raise self.fail

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, **kwargs):
        self.collections.add(collection_name)

    def create_payload_index(self, **kwargs):
        pass

    def delete_collection(self, name):
        self.collections.discard(name)

    def scroll(self, **kwargs):
        if self.fail:
            raise self.fail
        return [], None

    def close(self):
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch):
    _FakeClient.instances = []
    monkeypatch.setattr("zotero_rag.retrieval.qdrant_manager.qc.QdrantClient", _FakeClient)
    return _FakeClient


def _transport_error():
    import httpx
    from qdrant_client.http.exceptions import ResponseHandlingException
    return ResponseHandlingException(httpx.ConnectError("refused"))


def test_client_connects_once_and_creates_both_collections(fake_client):
    manager = QdrantManager("m", collection_suffix="_t")
    assert manager.list_indexed_pdfs() == []
    assert manager.list_indexed_pdfs() == []
    [client] = fake_client.instances
    assert client.collections == {manager.chunk_collection, manager.registry_collection}


def test_clearing_recreates_the_chunk_collection_on_next_use(fake_client):
    manager = QdrantManager("m")
    manager.clear_chunk_collection()
    [client] = fake_client.instances
    assert manager.chunk_collection not in client.collections
    manager.list_indexed_pdfs()
    assert manager.chunk_collection in client.collections


def test_unreachable_service_raises_connection_error_and_is_retried(fake_client, monkeypatch):
    monkeypatch.setattr("zotero_rag.retrieval.qdrant_manager.qc.QdrantClient",
                        lambda url: _FakeClient(url, fail=RuntimeError("down")))
    manager = QdrantManager("m")
    with pytest.raises(ConnectionError, match="Qdrant service not reachable"):
        manager.list_indexed_pdfs()
    assert fake_client.instances[0].closed
    with pytest.raises(ConnectionError):
        manager.ping()
    assert len(fake_client.instances) == 2  # a failed connect is not cached


def test_service_lost_mid_session_raises_connection_error(fake_client):
    """The app tells "Qdrant is not running" apart by this exception type."""
    manager = QdrantManager("m")
    manager.list_indexed_pdfs()
    fake_client.instances[0].fail = _transport_error()
    with pytest.raises(ConnectionError, match="Qdrant service not reachable"):
        manager.list_indexed_pdfs()


def test_search_returns_chunks_with_their_scores_in_order(fake_client):
    from types import SimpleNamespace
    manager = QdrantManager("m")
    manager.list_indexed_pdfs()
    points = [SimpleNamespace(payload={"text": "a", "pdf_hash": "h", "chunk_index": 1}, score=0.9),
              SimpleNamespace(payload={"text": "b", "pdf_hash": "h", "chunk_index": 2}, score=0.4)]
    fake_client.instances[0].query_batch_points = (
        lambda collection_name, requests: [SimpleNamespace(points=points)])

    hits = manager.search({"dense": DENSE, "sparse": SPARSE}, mode="dense")

    assert [(c.text, c.chunk_index, s) for c, s in hits] == [("a", 1, 0.9), ("b", 2, 0.4)]


def _registry_client(fake_client, models):
    from types import SimpleNamespace
    manager = QdrantManager("m")
    manager.list_indexed_pdfs()
    client = fake_client.instances[0]
    client.calls = []
    client.delete = lambda collection_name, points_selector: client.calls.append(
        ("delete", collection_name))
    client.set_payload = lambda collection_name, payload, points: client.calls.append(
        ("set_payload", payload["models"]))
    client.retrieve = lambda **kw: ([SimpleNamespace(payload={"models": models})]
                                    if models is not None else [])
    return manager, client


def test_deleting_the_last_model_drops_the_registry_entry(fake_client):
    manager, client = _registry_client(fake_client, ["m"])
    assert manager.delete_pdf_from_index("h") is False
    assert client.calls == [("delete", manager.chunk_collection),
                            ("delete", manager.registry_collection)]


def test_deleting_one_of_two_models_keeps_the_other(fake_client):
    manager, client = _registry_client(fake_client, ["m", "other"])
    assert manager.delete_pdf_from_index("h") is True
    assert client.calls[-1] == ("set_payload", ["other"])


def test_deleting_an_unregistered_pdf_raises(fake_client):
    manager, _ = _registry_client(fake_client, None)
    with pytest.raises(ValueError, match="Failed to delete"):
        manager.delete_pdf_from_index("h")


def test_non_transport_failures_are_not_reported_as_unreachable(fake_client):
    from qdrant_client.http.exceptions import ResponseHandlingException
    manager = QdrantManager("m")
    manager.list_indexed_pdfs()
    fake_client.instances[0].fail = ResponseHandlingException(ValueError("bad payload"))
    with pytest.raises(ResponseHandlingException):
        manager.list_indexed_pdfs()


# ---- Embedded storage: the whole round trip, no server ----------------------
DIM = 4


def _chunk(pdf_hash, index, text):
    return Chunk(text=text, page_number=0, chunk_index=index, title=pdf_hash, pdf_hash=pdf_hash)


@pytest.fixture
def open_manager(tmp_path):
    """Managers on one storage folder, all closed at teardown even if the test fails."""
    opened = []

    def open_one():
        manager = QdrantManager("m", qdrant_url=None, qdrant_path=str(tmp_path / "qdrant"),
                                vector_size=DIM)
        opened.append(manager)
        return manager
    yield open_one
    for manager in opened:
        manager.close()


def test_embedded_storage_indexes_searches_and_persists(open_manager):
    manager = open_manager()
    manager.register_pdf_title("p1", "Paper one")
    chunks = [_chunk("p1", 0, "graphs"), _chunk("p1", 1, "tables")]
    manager.upsert_chunks(chunks,
                          dense_embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]],
                          sparse_embeddings=[{"indices": [1], "values": [1.0]},
                                             {"indices": [2], "values": [1.0]}])

    hits = manager.search({"dense": [1, 0, 0, 0], "sparse": {"indices": [1], "values": [1.0]}},
                          result_limit=2)
    assert hits[0][0].text == "graphs"
    assert manager.is_pdf_indexed("p1")
    manager.close()

    assert [p["pdf_hash"] for p in open_manager().list_indexed_pdfs()] == ["p1"]


def test_a_second_process_on_the_same_storage_is_told_why(open_manager):
    open_manager().ping()
    with pytest.raises(ConnectionError, match="in use"):
        open_manager().ping()


def test_a_papers_reference_is_kept_in_the_registry(open_manager):
    manager = open_manager()
    manager.register_pdf_title("p1", "Paper one")
    reference = Reference(["Dasigi", "Lo"], "2021", zotero_key="KEY")
    manager.set_reference("p1", reference)
    assert manager.reference("p1") == reference
    assert manager.register_pdf_title("p1", "Paper one") == "Paper one"  # a later registration keeps it
    assert manager.reference("p1") == reference
    assert manager.reference("unknown") is None


def test_a_reference_for_a_paper_with_no_registry_entry_is_skipped(open_manager):
    manager = open_manager()
    manager.set_reference("never-registered", Reference(["X"], "2020"))
    assert manager.reference("never-registered") is None
