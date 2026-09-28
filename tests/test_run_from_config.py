"""Unit tests for zotero_rag/run_from_config.py, with a stand-in for ZoteroRAG.

The real class pulls in torch and transformers, so `pipeline` is stubbed in
`sys.modules` before the runner is imported: what is under test here is the
runner's own wiring - which source it ingests from, when it clears the index,
and where it writes - not the pipeline behind it.
"""

import json
import sys
import types
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

calls = {}


class _FakeRag:
    def __init__(self, **kwargs):
        calls.clear()
        calls["init"] = kwargs

    def clear_index(self):
        calls["cleared"] = True
        return True

    def ingest_pdfs_from_folder(self, folder_path):
        calls["folder"] = folder_path
        return types.SimpleNamespace(failed_pdfs=[], ingested_pdfs=[])

    def ingest_pdfs_from_zotero(self, zotero_collection=None, zotero_data_dir=None):
        calls["zotero"] = (zotero_collection, zotero_data_dir)
        return types.SimpleNamespace(failed_pdfs=[], ingested_pdfs=[])

    def upsert_pdfs(self, pdfs):
        return types.SimpleNamespace(indexed_chunks=0, processed_pdfs=0)

    def answer_question(self, **kwargs):
        calls.setdefault("questions", []).append(kwargs["question"])
        calls.setdefault("answer_kwargs", []).append(kwargs)
        return []


sys.modules.setdefault("pipeline", types.SimpleNamespace(ZoteroRAG=_FakeRag))

from run_from_config import load_config, run_from_config


def _write_config(tmp_path, **overrides):
    config = {
        "output_base_dir": str(tmp_path / "out"),
        "questions": [{"question": "What are transformers?"}],
        "create_highlighted_pdfs": False,
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_load_config_fills_in_the_documented_defaults(tmp_path):
    config = load_config(str(_write_config(tmp_path)))
    assert config["dense_model_name"] == "google/embeddinggemma-300m"
    assert config["rebuild_index"] is False


def test_zotero_source_forwards_collection_and_data_dir(tmp_path):
    path = _write_config(tmp_path, zotero_collection="ML Papers",
                         zotero_data_dir="/tmp/zotero")
    run_from_config(str(path))
    assert calls["zotero"] == ("ML Papers", "/tmp/zotero")
    assert "cleared" not in calls


def test_folder_source_ingests_the_configured_folder(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    path = _write_config(tmp_path, source_type="folder", folder_path=str(pdf_dir))
    run_from_config(str(path))
    assert calls["folder"] == str(pdf_dir)


def test_rebuild_index_clears_before_ingesting(tmp_path):
    run_from_config(str(_write_config(tmp_path, rebuild_index=True)))
    assert calls["cleared"] is True


def test_constructor_gets_no_source_keywords(tmp_path):
    run_from_config(str(_write_config(tmp_path, source_type="folder",
                                      folder_path=str(tmp_path))))
    assert not {"source_type", "folder_path", "zotero_collection"} & set(calls["init"])
    assert calls["init"]["output_base_dir"] == str(tmp_path / "out")


def test_results_file_is_written_even_without_highlighted_pdfs(tmp_path):
    path = _write_config(tmp_path, results_file="results.json")
    run_from_config(str(path))
    written = tmp_path / "out" / "highlighted_results" / "results.json"
    assert json.loads(written.read_text()) == {"What are transformers?": []}


def test_a_question_reaches_the_pipeline_with_its_colour_and_no_settings(tmp_path):
    path = _write_config(tmp_path, questions=[{"question": "q", "highlight_color": [0, 1, 0],
                                               "rerank_gate": 0.1}])
    run_from_config(str(path))
    assert calls["answer_kwargs"] == [{"question": "q", "highlight_color": (0, 1, 0)}]


def test_the_reader_and_reranker_models_are_not_configurable(tmp_path):
    run_from_config(str(_write_config(tmp_path, qa_model="x", reranker_model="y")))
    assert not {"qa_model", "reranker_model"} & set(calls["init"])


def test_answer_to_dict_keeps_what_the_results_file_documents():
    from models import Answer
    from run_from_config import answer_to_dict
    answer = Answer(text="t", context="c", page_number=2, title="T", score=0.5, query="q",
                    sentence_coords=["1,1,1,1,1"], pdf_path="/p.pdf", pdf_hash="h")
    assert answer_to_dict(answer) == {
        "text": "t", "context": "c", "pdf_path": "/p.pdf", "page_number": 2, "title": "T",
        "section": "body", "start_char": 0, "end_char": 0, "score": 0.5, "query": "q",
        "color": (1, 1, 0), "retrieval_score": 0.0, "rerank_score": 0.0}


def test_highlighting_writes_one_pdf_per_source_pdf(tmp_path, monkeypatch):
    import run_from_config as runner
    from models import Answer

    answers = [Answer(text=t, context=t, page_number=0, title=t, pdf_path=p)
               for t, p in (("a", "/x/one.pdf"), ("b", "/x/two.pdf"), ("c", "/x/one.pdf"))]
    monkeypatch.setattr(_FakeRag, "answer_question", lambda self, **kw: answers)
    written = {}
    monkeypatch.setattr(runner, "highlight_pdf",
                        lambda group, out: written.setdefault(Path(out).name,
                                                              [a.text for a in group]))

    run_from_config(str(_write_config(tmp_path, create_highlighted_pdfs=True)))

    assert written == {"one_highlighted.pdf": ["a", "c"], "two_highlighted.pdf": ["b"]}


def test_missing_questions_is_reported_not_crashed(tmp_path):
    path = _write_config(tmp_path)
    config = yaml.safe_load(path.read_text())
    del config["questions"]
    path.write_text(yaml.safe_dump(config))
    assert run_from_config(str(path)) == {}
