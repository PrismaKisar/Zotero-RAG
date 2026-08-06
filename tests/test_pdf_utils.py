"""Unit tests for zotero_rag/pdf_utils.py pure helpers."""

import hashlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from pdf_utils import (
    compute_file_hash,
    compute_stream_hash,
    sanitize_filename,
)


def test_sanitize_filename_replaces_spaces_and_slashes():
    assert sanitize_filename("my paper/v2\\final") == "my_paper_v2_final"


def test_sanitize_filename_strips_unsafe_characters():
    assert sanitize_filename("a:b*c?d") == "abcd"


def test_sanitize_filename_defaults_on_empty_input():
    assert sanitize_filename(None) == "_All_Library"
    assert sanitize_filename("") == "_All_Library"
    assert sanitize_filename("///") == "_All_Library"


def test_sanitize_filename_truncates_to_max_length():
    assert sanitize_filename("a" * 300, max_length=10) == "a" * 10


def test_compute_stream_hash_matches_hashlib():
    data = b"hello world" * 1000
    expected = hashlib.sha256(data).hexdigest()
    assert compute_stream_hash(io.BytesIO(data)) == expected


def test_compute_stream_hash_resets_seekable_stream_position():
    stream = io.BytesIO(b"some bytes")
    compute_stream_hash(stream)
    assert stream.tell() == 0


def test_compute_file_hash_matches_hashlib(tmp_path):
    data = b"file contents for hashing"
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(data)
    assert compute_file_hash(str(file_path)) == hashlib.sha256(data).hexdigest()
