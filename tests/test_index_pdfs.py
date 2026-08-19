
from benchmark.index_pdfs import build_ingest_items


def test_build_ingest_items_titles_by_paper_id(tmp_path):
    for paper_id in ("1605.03481", "1601.02403"):
        (tmp_path / f"{paper_id}.pdf").write_bytes(b"%PDF-1.4")

    items = build_ingest_items(tmp_path)

    assert [item.title for item in items] == ["1601.02403", "1605.03481"]
    assert all(item.source.path.endswith(".pdf") for item in items)


def test_build_ingest_items_ignores_non_pdf_files(tmp_path):
    (tmp_path / "notes.txt").write_text("not a pdf")
    (tmp_path / "1601.02403.pdf").write_bytes(b"%PDF-1.4")

    items = build_ingest_items(tmp_path)

    assert [item.title for item in items] == ["1601.02403"]


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

        def _ingest_pdfs(self, items):
            return types.SimpleNamespace(failed_pdfs=[], ingested_pdfs=[])

        def upsert_pdfs(self, pdfs):
            captured["upserted"] = pdfs

    monkeypatch.setitem(sys.modules, "zotero_rag.zotero_rag",
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
    """Importing a benchmark module must not rebind `zotero_rag` to the module file.

    The directory and one of its modules share a name, so a sys.path entry added
    ahead of the repo root makes `zotero_rag` the file and breaks every
    `from zotero_rag.<x> import ...` later in the process.
    """
    import benchmark.index_pdfs  # noqa: F401 - its sys.path setup is the point
    import zotero_rag

    assert hasattr(zotero_rag, "__path__"), "zotero_rag got shadowed by zotero_rag.py"

    from zotero_rag.question_presets import PRESETS

    assert PRESETS
