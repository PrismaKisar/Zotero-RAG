"""Chats, and the answers they highlight from, in one SQLite file.

SQLite rather than files per chat: one file to back up or delete, writes that
are atomic, and a chat list that needs no file read per chat. It ships with
Python, so it costs no dependency.

A chat's items are the page's own records, stored as one JSON column: the page
always sends a chat whole, and nothing queries inside it. A run is the full
answers of one question, kept so an answer can still be opened and highlighted
after a restart.
"""

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
create table if not exists chats (
    id text primary key, title text not null, items text not null, updated integer not null);
-- ponytail: every run is kept, so a chat's answers open forever; prune the runs no
-- chat refers to if the file ever grows large.
create table if not exists runs (id text primary key, answers text not null);
"""


class ChatStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """One connection per call, committed on success: the server runs requests in threads."""
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_chat(self, chat_id: str, title: str, items: list[dict]) -> None:
        with self._db() as db:
            db.execute("insert or replace into chats values (?, ?, ?, ?)",
                       (chat_id, title, json.dumps(items), time.time_ns()))

    def chats(self) -> list[dict]:
        """Every chat, last changed first."""
        with self._db() as db:
            rows = db.execute("select id, title, items from chats order by updated desc").fetchall()
        return [{"id": i, "title": t, "items": json.loads(items)} for i, t, items in rows]

    def delete_chat(self, chat_id: str) -> None:
        with self._db() as db:
            db.execute("delete from chats where id = ?", (chat_id,))

    def save_run(self, run: str, answers: list[dict]) -> None:
        with self._db() as db:
            db.execute("insert or replace into runs values (?, ?)", (run, json.dumps(answers)))

    def run(self, run: str) -> list[dict] | None:
        with self._db() as db:
            row = db.execute("select answers from runs where id = ?", (run,)).fetchone()
        return json.loads(row[0]) if row else None
