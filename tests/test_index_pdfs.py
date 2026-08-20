
def test_collection_suffix_isolates_chunk_and_registry_collections():
    """Two corpora must not share a retrieval pool, nor a title registry."""
    from qdrant_manager import QdrantManager

    default = QdrantManager()
    qasa = QdrantManager(collection_suffix="_qasa")

    assert qasa.chunk_collection == default.chunk_collection + "_qasa"
    assert qasa.registry_collection == default.registry_collection + "_qasa"
    assert qasa.chunk_collection != default.chunk_collection


def test_index_pdfs_defaults_contextualization_off(monkeypatch, tmp_path):
    """The baseline index must be reproducible: no silent LLM enrichment."""
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

    from benchmark.index_pdfs import index_pdfs

    index_pdfs(tmp_path, tmp_path, "http://grobid", "http://qdrant")
    assert captured["use_chunk_contextualization"] is False
    assert captured["qdrant_collection_suffix"] == ""

    index_pdfs(tmp_path, tmp_path, "http://grobid", "http://qdrant",
               qdrant_collection_suffix="_qasa", contextualize=True)
    assert captured["use_chunk_contextualization"] is True
    assert captured["qdrant_collection_suffix"] == "_qasa"


def test_zotero_rag_name_resolves_to_the_package():
    """Importing a benchmark module must not rebind `zotero_rag` to a module file.

    No module inside the package shares the package's name any more, so nothing
    can shadow it; this pins that down, because a re-introduced `zotero_rag.py`
    would silently break every `from zotero_rag.<x> import ...` in the process.
    """
    import benchmark.index_pdfs  # noqa: F401 - its sys.path setup is the point
    import zotero_rag

    assert hasattr(zotero_rag, "__path__"), "zotero_rag got shadowed by zotero_rag.py"

    from zotero_rag.question_presets import PRESETS

    assert PRESETS
