"""Shared helpers for PDF hashing and safe filenames."""

from __future__ import annotations

import hashlib
import re
from typing import BinaryIO

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


def is_pdf_hash(name: str) -> bool:
    """Tell whether a bare filename is one of our SHA-256 cache names.

    Args:
        name: Filename without extension.

    Returns:
        True if the name is a 64-character hexadecimal digest.
    """
    return re.fullmatch(r"[a-fA-F0-9]{64}", name or "") is not None


def compute_stream_hash(stream: BinaryIO) -> str:
    """SHA-256 hex digest of a whole seekable stream, left rewound for the next reader."""
    stream.seek(0)
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
    stream.seek(0)
    return digest


def compute_file_hash(file_path: str) -> str:
    """SHA-256 hex digest of the file at ``file_path``."""
    with open(file_path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
