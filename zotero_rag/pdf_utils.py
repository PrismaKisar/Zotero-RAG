"""Shared helpers for PDF hashing and safe filenames."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import BinaryIO

logger = logging.getLogger(__name__)

_DEFAULT_NAME = "_All_Library"


def sanitize_filename(name: str | None, max_length: int = 200) -> str:
    """Convert a string into a filesystem-safe filename.
        Replaces spaces and slashes with underscores, removes unsafe characters, and truncates to a maximum length.

    Args:
        name: The input string to sanitize.
        max_length: The maximum length of the resulting filename (default is 200).
    
    Returns:
        A sanitized filename string.
    """
    if not name:
        return _DEFAULT_NAME

    sanitized = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    sanitized = re.sub(r"(?u)[^-\w.]", "", sanitized)
    sanitized = sanitized.strip("._")
    if not sanitized:
        sanitized = _DEFAULT_NAME

    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def compute_stream_hash(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash from a binary stream.
    
    Args:
        stream: A binary stream to read from.
        chunk_size: The size of chunks to read at a time (default is 1 MB).
        
    Returns:
        The SHA-256 hash of the stream as a hexadecimal string.
    """
    h = hashlib.sha256()
    can_seek = hasattr(stream, "seek")

    if can_seek:
        try:
            stream.seek(0)
        except Exception:
            can_seek = False

    for chunk in iter(lambda: stream.read(chunk_size), b""):
        if not chunk:
            break
        h.update(chunk)

    if can_seek:
        try:
            stream.seek(0)
        except Exception:
            logger.debug("Unable to reset stream position", exc_info=True)

    return h.hexdigest()


def compute_file_hash(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file.
    
    Args:
        file_path: The path to the file to hash.
        chunk_size: The size of chunks to read at a time (default is 1 MB).
        
    Returns:
        The SHA-256 hash of the file as a hexadecimal string.
    """
    with open(file_path, "rb") as f:
        return compute_stream_hash(f, chunk_size=chunk_size)
