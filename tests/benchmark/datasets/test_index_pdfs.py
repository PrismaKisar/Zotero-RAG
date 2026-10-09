
def test_collection_suffix_isolates_chunk_and_registry_collections():
    """Two corpora must not share a retrieval pool, nor a title registry."""
    from zotero_rag.retrieval.qdrant_manager import QdrantManager

    default = QdrantManager("m")
    qasa = QdrantManager("m", collection_suffix="_qasa")

    assert qasa.chunk_collection == default.chunk_collection + "_qasa"
    assert qasa.registry_collection == default.registry_collection + "_qasa"
    assert qasa.chunk_collection != default.chunk_collection


def test_index_pdfs_passes_the_collection_suffix(monkeypatch, tmp_path):
    """Each split indexes into its own collection."""
    import sys
    import types

    captured = {}

    class _FakeRag:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def ingest_pdfs_from_folder(self, folder_path):
            captured["ingested_from"] = folder_path
            return types.SimpleNamespace(failed_pdfs=[], ingested_pdfs=[])

        def upsert_pdfs(self, pdfs):
            captured["upserted"] = pdfs

    monkeypatch.setitem(sys.modules, "zotero_rag.pipeline",
                        types.SimpleNamespace(ZoteroRAG=_FakeRag))

    from benchmark.datasets.index_pdfs import index_pdfs

    index_pdfs(tmp_path, tmp_path, "http://grobid", "http://qdrant")
    assert captured["qdrant_collection_suffix"] == ""

    index_pdfs(tmp_path, tmp_path, "http://grobid", "http://qdrant",
               qdrant_collection_suffix="_qasa")
    assert captured["qdrant_collection_suffix"] == "_qasa"


def test_each_pdf_is_upserted_on_its_own(monkeypatch, tmp_path):
    """One failed PDF must not cost every other PDF its index."""
    import sys
    import types

    upserts = []

    class _FakeRag:
        def __init__(self, **kwargs):
            pass

        def ingest_pdfs_from_folder(self, folder_path):
            pdfs = [types.SimpleNamespace(title=t, pdf_hash=t + "-hash") for t in ("a", "b")]
            return types.SimpleNamespace(failed_pdfs=[], ingested_pdfs=pdfs)

        def upsert_pdfs(self, pdfs):
            upserts.append([p.title for p in pdfs])
            return types.SimpleNamespace(indexed_chunks=3)

    monkeypatch.setitem(sys.modules, "zotero_rag.pipeline",
                        types.SimpleNamespace(ZoteroRAG=_FakeRag))

    from benchmark.datasets.index_pdfs import index_pdfs

    assert index_pdfs(tmp_path, tmp_path, "g", "q") == {"a": "a-hash", "b": "b-hash"}
    assert upserts == [["a"], ["b"]]
