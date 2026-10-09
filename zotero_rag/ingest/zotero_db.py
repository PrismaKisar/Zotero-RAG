"""Zotero database interaction utilities."""

import logging
import os
import sqlite3
from contextlib import closing

from zotero_rag.models import PathSource, PDFIngestItem, Reference

logger = logging.getLogger(__name__)


def _field(cursor: sqlite3.Cursor, item_id: int, name: str) -> str | None:
    """One field of a Zotero item (title, date, ...), or None if it has none."""
    cursor.execute(
        "SELECT v.value FROM itemData d "
        "JOIN itemDataValues v ON d.valueID = v.valueID "
        "JOIN fields f ON d.fieldID = f.fieldID "
        "WHERE d.itemID = ? AND f.fieldName = ?",
        (item_id, name))
    row = cursor.fetchone()
    return str(row[0]).strip() if row and row[0] else None


def _reference(cursor: sqlite3.Cursor, item_id: int, attachment_key: str,
               library_id: int) -> Reference:
    """The item's authors and year as Zotero holds them, curated by the user.

    Zotero stores a date as "2020-05-00 May 2020": the year leads, when known. An
    attachment in a group library is linked through its group, not "library".
    """
    cursor.execute("SELECT groupID FROM groups WHERE libraryID = ?", (library_id,))
    group = cursor.fetchone()
    cursor.execute(
        "SELECT c.lastName FROM itemCreators ic "
        "JOIN creators c ON ic.creatorID = c.creatorID "
        "JOIN creatorTypes t ON ic.creatorTypeID = t.creatorTypeID "
        "WHERE ic.itemID = ? AND t.creatorType = 'author' ORDER BY ic.orderIndex",
        (item_id,))
    authors = [name for (name,) in cursor.fetchall() if name]
    date = _field(cursor, item_id, "date") or ""
    year = date[:4] if date[:4].isdigit() else None
    return Reference(authors, year, zotero_key=attachment_key,
                     zotero_library=f"groups/{group[0]}" if group else "library")


class ZoteroDatabase:
    """Handles interaction with the Zotero SQLite database."""
    
    @staticmethod
    def find_zotero_dir(custom_dir: str | None = None) -> str:
        """Find Zotero data directory."""
        if custom_dir and os.path.exists(custom_dir):
            return custom_dir
        possible_paths = [
            os.path.expanduser(p) 
            for p in ["~/Zotero", "~/Documents/Zotero", "~/.zotero"]
        ]
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'zotero.sqlite')):
                return path
        raise ValueError("Zotero directory not found. Please specify zotero_data_dir")
    
    def __init__(self, zotero_data_dir: str | None = None):
        """Initialize Zotero database connection.
        
        Args:
            zotero_data_dir: Path to Zotero data directory. If None, auto-detect.
        """
        self.zotero_dir = self.find_zotero_dir(zotero_data_dir)
        self.storage_dir = os.path.join(self.zotero_dir, 'storage')
        self.db_path = os.path.join(self.zotero_dir, 'zotero.sqlite')
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Zotero database not found at {self.db_path}")

    def _connect(self) -> closing[sqlite3.Connection]:
        """A read-only connection, closed when the ``with`` block exits."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA query_only = ON")
        return closing(conn)

    def _attachment_path(self, key: str, path: str | None) -> str | None:
        """An attachment's file: ``storage:name`` lives in the attachment's storage folder."""
        if path and path.startswith('storage:'):
            return os.path.join(self.storage_dir, key, path.replace('storage:', ''))
        return path

    def list_collections(self) -> list[dict]:
        """Load collections from the Zotero database.

        Returns:
            List of dictionaries with 'id', 'name', and 'parent_id' keys.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT collectionID, collectionName, parentCollectionID "
                "FROM collections ORDER BY collectionName"
            ).fetchall()
        return [{'id': r[0], 'name': r[1], 'parent_id': r[2]} for r in rows]

    def get_items(self, collection_name: str | None = None) -> list[PDFIngestItem]:
        """Get PDF items from Zotero library or a specific collection.

        Args:
            collection_name: Name of the collection to filter by. If None, get all items.

        Returns:
            List of PDFIngestItem objects with title and source.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            if collection_name:
                cursor.execute(
                    "SELECT collectionID FROM collections WHERE collectionName = ?",
                    (collection_name,)
                )
                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"Collection '{collection_name}' not found.")
                cursor.execute("""
                SELECT a_items.key, ia.path, ia.parentItemID as sourceItemID, a_items.libraryID
                FROM collectionItems ci
                JOIN itemAttachments ia ON ci.itemID = ia.parentItemID
                JOIN items a_items ON ia.itemID = a_items.itemID
                WHERE ci.collectionID = ?
                AND ia.contentType = 'application/pdf'
                AND ia.path IS NOT NULL
                """, (result[0],))
            else:
                cursor.execute("""
                SELECT i.key, ia.path, COALESCE(ia.parentItemID, i.itemID) as sourceItemID, i.libraryID
                FROM items i
                JOIN itemAttachments ia ON i.itemID = ia.itemID
                WHERE ia.contentType = 'application/pdf'
                AND ia.path IS NOT NULL
                """)

            items: list[PDFIngestItem] = []
            for key, path, src_id, library_id in cursor.fetchall():
                pdf_path = self._attachment_path(key, path)
                if pdf_path and os.path.exists(pdf_path):
                    items.append(PDFIngestItem(
                        title=_field(cursor, src_id, "title") or "Unknown",
                        source=PathSource(pdf_path),
                        reference=_reference(cursor, src_id, key, library_id)))
            return items
