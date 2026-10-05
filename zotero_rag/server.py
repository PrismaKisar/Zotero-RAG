"""
server.py - HTTP API and web page for Zotero RAG System

Serves ``web/index.html`` and the JSON API it calls. The long operations (ask,
index) stream newline-delimited JSON: progress events, then one ``done`` or
``error`` event.

Run with: uv run uvicorn server:app --app-dir zotero_rag
"""

import io
import json
import os
import queue
import threading
import uuid
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from highlighter import highlight_pdf
from pdf_utils import sanitize_filename
from pipeline import ZoteroRAG
from pydantic import BaseModel, Field
from zotero_db import ZoteroDatabase

WHOLE_LIBRARY = "Whole library"
ZOTERO_DIR = os.environ.get("ZOTERO_DATA_DIR")  # None: look in the usual places
WEB = Path(__file__).resolve().parent / "web"


class Config(BaseModel):
    grobid_url: str = os.environ.get("GROBID_URL", "http://localhost:8070")
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    output_base_dir: str = os.environ.get("OUTPUT_DIR", "./output")
    model_device: str | None = Field(None, pattern="^(cpu|cuda|mps)$")
    use_chunk_contextualization: bool = False


class Question(BaseModel):
    question: str = Field(min_length=1)


class Titles(BaseModel):
    titles: list[str]


class Collection(BaseModel):
    label: str


class Highlight(BaseModel):
    run: str
    answers: list[int] = Field(min_length=1)
    color: str = Field(pattern="^#[0-9a-fA-F]{6}$")


# ponytail: one user, one pipeline, one lock; the pipeline is not thread-safe.
LOCK = threading.Lock()
# ponytail: every run kept in memory, so old chat messages can still open and
# highlight; a restart forgets them (the page says "ask again"). Cap it if a
# session ever grows large.
state = {"config": Config(), "rag": None, "runs": {}}


def rag() -> ZoteroRAG:
    """The pipeline for the current config, built on first use."""
    if state["rag"] is None:
        state["rag"] = ZoteroRAG(**state["config"].model_dump())
    return state["rag"]


def highlighted_dir() -> Path:
    return Path(state["config"].output_base_dir) / "highlighted_results"


def collection_labels(collections: list[dict]) -> dict[str, str | None]:
    """Label (the full path) -> the collection name Zotero is queried by."""
    by_id = {c["id"]: c for c in collections}

    def path(c):
        parent = by_id.get(c["parent_id"])
        return f"{path(parent)} / {c['name']}" if parent else c["name"]

    labels = {path(c): c["name"] for c in collections}
    return {WHOLE_LIBRARY: None, **dict(sorted(labels.items()))}


def answer_json(index: int, answer) -> dict:
    return {
        "id": index, "text": answer.text, "context": answer.context,
        "start": answer.start_char, "end": answer.end_char,
        "title": answer.title, "page": answer.page_number + 1,
        "section": answer.section or "body", "has_pdf": bool(answer.pdf_path),
        "scores": {"retrieval": answer.retrieval_score,
                   "rerank": answer.rerank_score, "qa": answer.score},
    }


def index_report(ingested, upserted) -> list[dict]:
    """What an ingest and its upsert did, as ``{level, text}`` notices."""
    report = [{"level": "warning", "text": f"Not cached: {f.get('title', '?')}: {f.get('error', '?')}"}
              for f in ingested.failed_pdfs]
    if upserted is None:
        return [*report, {"level": "info", "text": "No PDF to index."}]
    if upserted.processed_pdfs:
        report.append({"level": "success", "text": f"Indexed {upserted.processed_pdfs} PDF(s), "
                                                   f"{upserted.indexed_chunks} chunks."})
    if upserted.already_indexed:
        report.append({"level": "info",
                       "text": f"{len(upserted.already_indexed)} PDF(s) already in the library."})
    if upserted.title_overrides:
        report.append({"level": "info",
                       "text": f"{len(upserted.title_overrides)} PDF(s) kept their existing title."})
    if upserted.duplicate_titles:
        report.append({"level": "warning",
                       "text": "Title already in use: " + ", ".join(upserted.duplicate_titles)})
    report += [{"level": "warning", "text": f"Failed: {f.get('title', '?')}: {f.get('error', '?')}"}
               for f in upserted.failed_pdfs]
    return report


def stream(work) -> StreamingResponse:
    """Run ``work(progress)`` under the lock in a thread, streaming its events as NDJSON.

    ``progress(stage, current, total, message)`` emits a progress event; the
    return value becomes the ``done`` event, an exception the ``error`` one.
    """
    events = queue.Queue()

    def progress(stage, current, total, message):
        events.put({"type": "progress", "stage": stage, "message": message,
                    "fraction": current / total if total else 0.0})

    def run():
        try:
            with LOCK:
                events.put({"type": "done", "result": work(progress)})
        except Exception as e:  # noqa: BLE001 - API boundary: report to the page, never crash
            events.put({"type": "error", "message": str(e)})

    def lines():
        while True:
            event = events.get()
            yield json.dumps(event) + "\n"
            if event["type"] != "progress":
                return

    threading.Thread(target=run, daemon=True).start()
    return StreamingResponse(lines(), media_type="application/x-ndjson")


def index(ingest):
    def work(progress):
        ingested = ingest()
        upserted = (rag().upsert_pdfs(target_pdfs=ingested.ingested_pdfs, progress_callback=progress)
                    if ingested.ingested_pdfs else None)
        return index_report(ingested, upserted)
    return stream(work)


app = FastAPI(title="Zotero RAG", docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.exception_handler(Exception)
async def unavailable(request, error):
    """Services down or the pipeline failing: a message the page can show."""
    return JSONResponse({"detail": str(error)}, status_code=503)


@app.get("/api/config")
def get_config() -> Config:
    return state["config"]


@app.put("/api/config")
def put_config(config: Config) -> Config:
    with LOCK:
        state.update(config=config, rag=None, runs={})
    return config


@app.get("/api/library")
def library():
    with LOCK:
        pdfs = rag().get_indexed_pdfs()
        consistent = rag().consistency_check(pdfs)
    return {"titles": sorted(item["title"] for item in pdfs), "consistent": consistent}


@app.post("/api/library/remove")
def remove(body: Titles):
    with LOCK:
        return [{"level": "success", "text": f"Removed {title}."} if rag().delete_pdf_by_title(title)
                else {"level": "error", "text": f"Could not remove {title}."} for title in body.titles]


@app.delete("/api/library")
def clear():
    with LOCK:
        cleared = rag().clear_index()
    return [{"level": "success", "text": "Library cleared."} if cleared
            else {"level": "error", "text": "Could not clear the library."}]


@app.get("/api/zotero/collections")
def zotero_collections() -> list[str]:
    return list(collection_labels(ZoteroDatabase(ZOTERO_DIR).list_collections()))


@app.post("/api/index/zotero")
def index_zotero(body: Collection):
    labels = collection_labels(ZoteroDatabase(ZOTERO_DIR).list_collections())
    if body.label not in labels:
        raise HTTPException(404, f"No Zotero collection {body.label!r}")
    return index(lambda: rag().ingest_pdfs_from_zotero(labels[body.label], ZOTERO_DIR))


@app.post("/api/index/upload")
async def index_upload(files: list[UploadFile]):
    # Read now: the request's files are closed before the stream runs.
    uploads = []
    for file in files:
        upload = io.BytesIO(await file.read())
        upload.name = os.path.basename(file.filename or "upload.pdf")
        uploads.append(upload)
    return index(lambda: rag().ingest_pdfs_from_upload(uploads))


@app.post("/api/ask")
def ask(body: Question):
    def work(progress):
        answers = rag().answer_question(
            question=body.question.strip(),
            rerank_callback=lambda *a: progress("rerank", *a),
            progress_callback=lambda *a: progress("read", *a))
        run = uuid.uuid4().hex
        state["runs"][run] = answers
        return {"run": run, "answers": [answer_json(i, a) for i, a in enumerate(answers)]}
    return stream(work)


def run_answers(run: str, indices: list[int]) -> list:
    """The answers of ``run`` at ``indices``; 404 if the server no longer has them."""
    answers = state["runs"].get(run, [])
    if not all(0 <= i < len(answers) for i in indices):
        raise HTTPException(404, "This answer is from an earlier session: ask again.")
    return [answers[i] for i in indices]


@app.get("/api/answers/{run}/{index}/pdf")
def answer_pdf(run: str, index: int):
    [answer] = run_answers(run, [index])
    if not answer.pdf_path:
        raise HTTPException(404, "This answer has no PDF.")
    return FileResponse(answer.pdf_path, media_type="application/pdf")


@app.post("/api/highlight")
def highlight(body: Highlight):
    """Highlight the chosen answers, one PDF per paper."""
    rgb = tuple(int(body.color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    by_pdf = defaultdict(list)
    for answer in run_answers(body.run, body.answers):
        by_pdf[answer.pdf_path].append(replace(answer, color=rgb))

    highlighted_dir().mkdir(parents=True, exist_ok=True)
    files, failed = [], []
    with LOCK:
        for group in by_pdf.values():
            title = group[0].title or Path(group[0].pdf_path or "untitled").stem
            path = highlighted_dir() / f"{sanitize_filename(title)}_highlighted.pdf"
            if highlight_pdf(group, str(path)):
                files.append(path.name)
            else:
                failed.append(title)
    return {"files": files, "failed": failed}


@app.get("/api/highlighted/{name}")
def highlighted(name: str):
    path = highlighted_dir() / name
    if name != Path(name).name or not path.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(path, media_type="application/pdf", filename=name)


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
