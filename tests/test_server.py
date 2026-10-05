"""The HTTP API, run in-process on a stand-in pipeline."""

import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import fitz
import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))


class FakeRAG:
    """Stands in for ZoteroRAG, which pulls in torch: under test is the API, not the pipeline."""

    answers: ClassVar[list] = []

    def __init__(self, **config):
        self.titles = ["Paper"]

    def get_indexed_pdfs(self):
        return [{"title": t, "pdf_hash": t} for t in self.titles]

    def consistency_check(self, pdfs):
        return True

    def delete_pdf_by_title(self, title):
        if title not in self.titles:
            return False
        self.titles.remove(title)
        return True

    def answer_question(self, question, progress_callback=None, rerank_callback=None):
        rerank_callback(1, 2, "half")
        return FakeRAG.answers

    def ingest_pdfs_from_upload(self, uploads):
        self.uploads = [(u.name, u.read()) for u in uploads]
        return types.SimpleNamespace(failed_pdfs=[], ingested_pdfs=["cached"])

    def upsert_pdfs(self, target_pdfs, progress_callback=None):
        progress_callback("encoding", 1, 1, "done")
        return types.SimpleNamespace(processed_pdfs=1, indexed_chunks=7, already_indexed=[],
                                     title_overrides=[], duplicate_titles=["Twin"], failed_pdfs=[])


# Stubbed only for this import, so no other test module finds it in place of its own.
stubbed = sys.modules.setdefault("pipeline", types.SimpleNamespace(ZoteroRAG=FakeRAG))
import server
from models import Answer

if sys.modules["pipeline"] is stubbed and stubbed.ZoteroRAG is FakeRAG:
    del sys.modules["pipeline"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ZoteroRAG", FakeRAG)
    server.state.update(config=server.Config(output_base_dir=str(tmp_path)), rag=None, runs={})
    yield TestClient(server.app, raise_server_exceptions=False)
    FakeRAG.answers = []


def events(response):
    return [json.loads(line) for line in response.text.splitlines()]


def pdf(tmp_path):
    doc = fitz.open()
    doc.new_page()
    path = str(tmp_path / "paper.pdf")
    doc.save(path)
    return path


def test_the_page_and_its_icons_are_served(client):
    assert "<title>Zotero RAG</title>" in client.get("/").text
    for path in ("/favicon.ico", "/brand/logo.svg", "/brand/apple-touch-icon.png"):
        assert client.get(path).status_code == 200, path


def test_collection_labels_spell_the_full_path_and_query_by_name():
    labels = server.collection_labels([{"id": 1, "name": "ML", "parent_id": None},
                                       {"id": 2, "name": "NLP", "parent_id": 1},
                                       {"id": 3, "name": "QA", "parent_id": 2}])
    assert labels == {"Whole library": None, "ML": "ML", "ML / NLP": "NLP", "ML / NLP / QA": "QA"}


def test_library_lists_titles_and_removes_them(client):
    assert client.get("/api/library").json() == {"titles": ["Paper"], "consistent": True}
    report = client.post("/api/library/remove", json={"titles": ["Paper", "Ghost"]}).json()
    assert [r["level"] for r in report] == ["success", "error"]
    assert client.get("/api/library").json()["titles"] == []


def test_ask_streams_progress_then_the_answers(client, tmp_path):
    FakeRAG.answers = [Answer(text="42", context="it is 42", page_number=2, title="Paper",
                              start_char=6, end_char=8, pdf_path=pdf(tmp_path))]
    stream = events(client.post("/api/ask", json={"question": "what?"}))

    assert stream[0] == {"type": "progress", "stage": "rerank", "message": "half", "fraction": 0.5}
    run, [answer] = stream[-1]["result"]["run"], stream[-1]["result"]["answers"]
    assert stream[-1]["type"] == "done"
    assert (answer["text"], answer["page"], answer["start"], answer["has_pdf"]) == ("42", 3, 6, True)
    assert client.get(f"/api/answers/{run}/0/pdf").headers["content-type"] == "application/pdf"
    assert client.get(f"/api/answers/{run}/1/pdf").status_code == 404
    assert "earlier session" in client.get("/api/answers/gone/0/pdf").json()["detail"]


def test_a_failing_pipeline_streams_an_error(client, monkeypatch):
    def broken(*args, **kwargs):
        raise ConnectionError("Qdrant down")
    monkeypatch.setattr(FakeRAG, "answer_question", broken)

    assert events(client.post("/api/ask", json={"question": "what?"})) == \
        [{"type": "error", "message": "Qdrant down"}]


def test_services_down_answer_503_with_the_reason(client, monkeypatch):
    def broken(self):
        raise ConnectionError("Qdrant down")
    monkeypatch.setattr(FakeRAG, "get_indexed_pdfs", broken)

    response = client.get("/api/library")
    assert (response.status_code, response.json()) == (503, {"detail": "Qdrant down"})


def test_upload_indexes_the_files_and_reports(client):
    stream = events(client.post("/api/index/upload",
                                files=[("files", ("../a.pdf", b"%PDF-1", "application/pdf"))]))

    assert stream[0]["stage"] == "encoding"
    assert [r["level"] for r in stream[-1]["result"]] == ["success", "warning"]
    assert server.state["rag"].uploads == [("a.pdf", b"%PDF-1")]


def test_saving_the_config_rebuilds_the_pipeline(client):
    first = server.rag()
    config = client.get("/api/config").json() | {"model_device": "cpu"}

    assert client.put("/api/config", json=config).json()["model_device"] == "cpu"
    assert server.rag() is not first
    assert client.put("/api/config", json=config | {"model_device": "tpu"}).status_code == 422


def test_highlight_writes_one_pdf_per_paper_and_serves_it(client, tmp_path):
    source = pdf(tmp_path)
    answer = Answer(text="x", context="x", page_number=0, title="Paper",
                    sentence_coords=["1,10,10,50,10"], pdf_path=source)
    server.state["runs"]["r"] = [answer, answer]

    result = client.post("/api/highlight", json={"run": "r", "answers": [0, 1], "color": "#ff0000"}).json()

    assert result == {"files": ["Paper_highlighted.pdf"], "failed": []}
    assert answer.color == (1, 1, 0)
    assert client.get("/api/highlighted/Paper_highlighted.pdf").status_code == 200
    assert client.get("/api/highlighted/..%2Fpaper.pdf").status_code == 404
    assert client.post("/api/highlight", json={"run": "r", "answers": [5], "color": "#ff0000"}).status_code == 404
