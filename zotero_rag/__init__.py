"""Zotero RAG pipeline package.

This file is not optional. The directory holds a module with the same name as
the directory itself (``zotero_rag/zotero_rag.py``), and Python resolves a
module file on *any* sys.path entry ahead of a namespace package on an earlier
one. Without ``__init__.py`` the name ``zotero_rag`` therefore binds to the file
as soon as anything puts this directory on sys.path, and every later
``from zotero_rag.<x> import ...`` fails with "is not a package" - which one
depends on import order, so the whole test suite passed or failed by accident.
Making this a regular package pins the name to the directory.
"""
