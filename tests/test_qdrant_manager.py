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
        DENSE, SPARSE, threshold=0.5, result_limit=30, mode=mode, pdf_hashes=pdf_hashes)


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


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown retrieval_mode"):
        build("bm25")
