"""Unit tests for zotero_rag/zotero_db.py."""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from zotero_db import ZoteroDatabase


def test_find_zotero_dir_uses_valid_custom_dir(tmp_path):
    (tmp_path / "zotero.sqlite").touch()
    assert ZoteroDatabase.find_zotero_dir(str(tmp_path)) == str(tmp_path)


def test_find_zotero_dir_raises_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "nonexistent"))
    with pytest.raises(ValueError, match="Zotero directory not found"):
        ZoteroDatabase.find_zotero_dir(None)


def test_init_raises_when_database_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ZoteroDatabase(str(tmp_path))


def _build_zotero_db(db_path, pdf_path):
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, collectionName TEXT, parentCollectionID INTEGER);
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT);
        CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INTEGER, path TEXT, contentType TEXT);
        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        CREATE TABLE collectionItems (collectionID INTEGER, itemID INTEGER);

        INSERT INTO collections VALUES (1, 'My Collection', NULL);

        -- parent item (the paper, holds the title) and its PDF attachment
        INSERT INTO items VALUES (10, 'PARENTKEY');
        INSERT INTO items VALUES (11, 'ATTACHKEY');

        INSERT INTO fields VALUES (1, 'title');
        INSERT INTO itemDataValues VALUES (1, 'My Paper Title');
        INSERT INTO itemData VALUES (10, 1, 1);

        INSERT INTO collectionItems VALUES (1, 10);
    """)
    conn.execute(
        "INSERT INTO itemAttachments VALUES (11, 10, ?, 'application/pdf')", (pdf_path,)
    )
    conn.commit()
    conn.close()


@pytest.fixture
def zotero_dir(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    _build_zotero_db(str(tmp_path / "zotero.sqlite"), str(pdf_path))
    return tmp_path


def test_list_collections(zotero_dir):
    db = ZoteroDatabase(str(zotero_dir))
    collections = db.list_collections()
    assert collections == [{"id": 1, "name": "My Collection", "parent_id": None}]


def test_get_items_from_library(zotero_dir):
    db = ZoteroDatabase(str(zotero_dir))
    items = db.get_items()
    assert len(items) == 1
    assert items[0].title == "My Paper Title"
    assert items[0].source.path == str(zotero_dir / "paper.pdf")


def test_get_items_from_collection(zotero_dir):
    db = ZoteroDatabase(str(zotero_dir))
    items = db.get_items(collection_name="My Collection")
    assert len(items) == 1
    assert items[0].title == "My Paper Title"


def test_get_items_raises_for_unknown_collection(zotero_dir):
    db = ZoteroDatabase(str(zotero_dir))
    with pytest.raises(ValueError, match="not found"):
        db.get_items(collection_name="Nonexistent")
