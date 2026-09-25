"""Unit tests for QdrantManager's query construction.

Only the pure request-building half is covered here: it needs no server, and it
is where a scoped search can go silently wrong - a filter that lands in the
wrong place returns fewer results rather than an error.
"""

import pytest
from qdrant_client import models as qmodels

from zotero_rag.qdrant_manager import QdrantManager

DENSE = [0.1] * 4
SPARSE = qmodels.SparseVector(indices=[1], values=[1.0])


def build(mode, pdf_hashes=None):
    return QdrantManager._build_query_request(
        DENSE, SPARSE, result_limit=30, mode=mode, pdf_hashes=pdf_hashes)


def hashes_in(flt):
    assert flt is not None
    return flt.must[0].match.any


def test_no_filter_when_no_documents_are_named():
    assert QdrantManager._pdf_hash_filter(None) is None
    assert QdrantManager._pdf_hash_filter([]) is None


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_every_mode_scopes_to_the_named_documents(mode):
    assert hashes_in(build(mode, ["a", "b"]).filter) == ["a", "b"]


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_every_mode_searches_the_whole_corpus_by_default(mode):
    assert build(mode).filter is None


def test_hybrid_filters_inside_each_prefetch_not_only_after_fusion():
    """Filtering only after RRF would let both prefetches fill their limit with
    out-of-scope chunks and fuse down to an empty result."""
    request = build("hybrid", ["a"])
    assert len(request.prefetch) == 2
    assert all(hashes_in(p.filter) == ["a"] for p in request.prefetch)


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_no_request_cuts_by_cosine_score(mode):
    """retrieval_threshold was removed: result_limit is the only stage-1 cut."""
    request = build(mode)
    assert request.score_threshold is None
    assert all(p.score_threshold is None for p in request.prefetch or [])


def test_dense_search_is_exact_not_approximate():
    """A library past Qdrant's indexing threshold must not turn approximate."""
    assert build("dense").params.exact is True
    dense_prefetch = build("hybrid").prefetch[0]
    assert dense_prefetch.using == "" and dense_prefetch.params.exact is True


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown retrieval_mode"):
        build("bm25")


class _FakeClient:
    """Stands in for QdrantClient: counts connects and collection creations."""

    instances = []

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
    monkeypatch.setattr("zotero_rag.qdrant_manager.qc.QdrantClient", _FakeClient)
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
    monkeypatch.setattr("zotero_rag.qdrant_manager.qc.QdrantClient",
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


def test_non_transport_failures_are_not_reported_as_unreachable(fake_client):
    from qdrant_client.http.exceptions import ResponseHandlingException
    manager = QdrantManager("m")
    manager.list_indexed_pdfs()
    fake_client.instances[0].fail = ResponseHandlingException(ValueError("bad payload"))
    with pytest.raises(ResponseHandlingException):
        manager.list_indexed_pdfs()
