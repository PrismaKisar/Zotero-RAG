"""Unit tests for zotero_rag/models.py: PDF sources and dataclass pickling."""

import io
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from models import Answer, Chunk, PathSource, UploadSource
from pdf_utils import compute_file_hash, compute_stream_hash


def test_upload_source_compute_hash_matches_stream_hash():
    data = b"uploaded pdf bytes"
    source = UploadSource(uploaded_file=io.BytesIO(data))
    assert source.compute_hash() == compute_stream_hash(io.BytesIO(data))


def test_upload_source_write_to_copies_bytes_and_resets_position(tmp_path):
    data = b"uploaded pdf bytes"
    upload = io.BytesIO(data)
    source = UploadSource(uploaded_file=upload)
    dest = tmp_path / "out.pdf"

    source.write_to(str(dest))

    assert dest.read_bytes() == data
    assert upload.tell() == 0


def test_path_source_compute_hash_matches_file_hash(tmp_path):
    src = tmp_path / "in.pdf"
    src.write_bytes(b"%PDF-1.4 content")
    source = PathSource(path=str(src))
    assert source.compute_hash() == compute_file_hash(str(src))


def test_path_source_write_to_copies_file(tmp_path):
    src = tmp_path / "in.pdf"
    src.write_bytes(b"%PDF-1.4 content")
    dest = tmp_path / "out.pdf"
    PathSource(path=str(src)).write_to(str(dest))
    assert dest.read_bytes() == src.read_bytes()


def test_chunk_survives_pickle_roundtrip():
    original = Chunk(
        text="some text", page_number=1, chunk_index=2, title="Title",
        pdf_hash="abc123", section="abstract", sentence_count=3,
        sentences=[("s1", "coords1")],
    )
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original


def test_answer_survives_pickle_roundtrip():
    original = Answer(
        text="answer", context="context", page_number=1, title="Title",
        section="body", start_char=0, end_char=6, score=0.9, query="q",
        sentence_coords=["c1"], retrieval_score=0.5, rerank_score=0.4,
        pdf_path="/tmp/x.pdf", pdf_hash="hash",
    )
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
