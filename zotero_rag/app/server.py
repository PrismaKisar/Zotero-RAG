"""
server.py - HTTP API and web page for Zotero RAG System

Serves ``web/index.html`` and the JSON API it calls. The long operations (ask,
index) stream newline-delimited JSON: progress events, then one ``done`` or
``error`` event.

Run with: uv run uvicorn server:app --app-dir zotero_rag
"""

import functools
import io
import json
import logging
import os
import queue
import threading
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from logging.handlers import RotatingFileHandler
from pathlib import Path

import platformdirs
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from zotero_rag.answering.highlighter import highlight_pdf
from zotero_rag.app.chat_store import ChatStore
from zotero_rag.ingest.pdf_utils import sanitize_filename
from zotero_rag.ingest.zotero_db import ZoteroDatabase
from zotero_rag.models import Answer, Reference, from_stored
from zotero_rag.pipeline import ZoteroRAG

WHOLE_LIBRARY = "Whole library"
ZOTERO_DIR = os.environ.get("ZOTERO_DATA_DIR")  # None: look in the usual places
WEB = Path(__file__).resolve().parent / "web"


def data_dir() -> str:
    """OUTPUT_DIR if set, else this user's data folder on macOS, Linux or Windows.

    Chats, the PDF and TEI caches and the highlighted PDFs live there, outside
    any checkout, as an installed app keeps them.
    """
    return os.environ.get("OUTPUT_DIR") or platformdirs.user_data_dir("Zotero RAG", appauthor=False)


class Config(BaseModel):
    """Where the services are, from the environment: the page has no settings."""
    grobid_url: str = os.environ.get("GROBID_URL", "http://localhost:8070")
    qdrant_url: str | None = os.environ.get("QDRANT_URL")  # None: embedded, in the data folder
    output_base_dir: str = data_dir()


class Chat(BaseModel):
    title: str
    items: list[dict]  # the page's own records: under test is that they come back as sent


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
state = {"config": Config(), "rag": None}


def rag() -> ZoteroRAG:
    """The pipeline for the current config, built on first use."""
    if state["rag"] is None:
        state["rag"] = ZoteroRAG(**state["config"].model_dump())
    return state["rag"]


def store() -> ChatStore:
    return _store(Path(state["config"].output_base_dir) / "zotero_rag.db")


@functools.cache
def _store(path: Path) -> ChatStore:
    """One store per database file, its schema checked once."""
    return ChatStore(path)


def save_run(run: str, answers: list[Answer]) -> None:
    store().save_run(run, [asdict(a) for a in answers])


def load_run(run: str) -> list[Answer]:
    return [from_stored(Answer, {**a, "color": tuple(a["color"])}) for a in store().run(run) or []]


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


def answer_json(index: int, answer, reference: Reference) -> dict:
    page = answer.page_number + 1
    return {
        "id": index, "text": answer.text, "context": answer.context,
        "start": answer.start_char, "end": answer.end_char,
        "title": answer.title, "page": page,
        "cite": reference.cite(),
        "zotero": reference.zotero_link(page),
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


LOG_HANDLERS = ("zotero-rag-file", "zotero-rag-console")


def configure_logging(folder: Path) -> None:
    """The app's log in its data folder (up to 10 MB, then one previous
    generation) and in the terminal. Replaces its own handlers, so a second
    start in one process does not write every line twice."""
    folder.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for old in [h for h in root.handlers if h.get_name() in LOG_HANDLERS]:
        root.removeHandler(old)
        old.close()
    file = RotatingFileHandler(folder / "app.log", maxBytes=10_000_000, backupCount=1,
                               encoding="utf-8")
    console = logging.StreamHandler()
    for name, handler in zip(LOG_HANDLERS, (file, console), strict=True):
        handler.set_name(name)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # At startup, not at import: importing the module (the tests do) writes no log.
    configure_logging(Path(state["config"].output_base_dir) / "logs")
    yield
    if state["rag"] is not None:  # releases the embedded index's lock
        state["rag"].qdrant_manager.close()


app = FastAPI(title="Zotero RAG", docs_url="/api/docs", openapi_url="/api/openapi.json",
              lifespan=lifespan)


@app.exception_handler(Exception)
async def unavailable(request, error):
    """Services down or the pipeline failing: a message the page can show, the traceback in the log."""
    logging.getLogger(__name__).error("%s %s failed", request.method, request.url.path, exc_info=error)
    return JSONResponse({"detail": str(error)}, status_code=503)


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
        save_run(run, answers)
        references = {h: rag().reference(h) for h in {a.pdf_hash for a in answers}}
        return {"run": run, "answers": [answer_json(i, a, references[a.pdf_hash])
                                        for i, a in enumerate(answers)]}
    return stream(work)


def run_answers(run: str, indices: list[int]) -> list:
    """The answers of ``run`` at ``indices``; 404 if the server no longer has them."""
    answers = load_run(run)
    if not all(0 <= i < len(answers) for i in indices):
        raise HTTPException(404, "This answer is no longer available: ask again.")
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


@app.get("/api/chats")
def chats() -> list[dict]:
    return store().chats()


@app.put("/api/chats/{chat_id}")
def save_chat(chat_id: str, chat: Chat):
    store().save_chat(chat_id, chat.title, chat.items)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    store().delete_chat(chat_id)


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
