"""Unit tests for zotero_rag/models.py: PDF sources and dataclass pickling."""

import io
import pickle
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from models import Answer, Chunk, PathSource, UploadSource, ingest_items_from_folder
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


def _sentenced(text, sentences):
    return Chunk(text=text, page_number=1, chunk_index=0, title="T", pdf_hash="h",
                 sentences=sentences)


# Both readers ask the chunk which sentences a character span falls in, since
# the highlighter draws per sentence. The extractive one widens its span to them;
# the generative one only needs their coordinates.

def test_a_span_widens_to_the_whole_sentences_it_touches():
    chunk = _sentenced("One two. Three four. Five.",
                       [("One two.", "cA"), ("Three four.", "cB"), ("Five.", "cC")])
    expanded = chunk.expand_to_sentences(10, 14)  # inside "Three four."
    assert (expanded.text, expanded.start_char, expanded.end_char) == ("Three four.", 9, 20)
    assert expanded.sentence_coords == ["cB"]


def test_a_span_across_a_boundary_takes_both_sentences():
    chunk = _sentenced("One two. Three four. Five.",
                       [("One two.", "cA"), ("Three four.", "cB"), ("Five.", "cC")])
    expanded = chunk.expand_to_sentences(4, 12)
    assert expanded.text == "One two. Three four."
    assert expanded.sentence_coords == ["cA", "cB"]


def test_without_sentences_the_span_is_kept_as_is():
    expanded = _sentenced("plain text", []).expand_to_sentences(0, 5)
    assert (expanded.text, expanded.start_char, expanded.end_char) == ("plain", 0, 5)
    assert expanded.sentence_coords == []


def test_coords_of_the_sentences_a_span_touches():
    chunk = _sentenced("Alpha one.  Beta two. Gamma.",
                       [("Alpha one.", "cA"), ("Beta two.", "cB"), ("Gamma.", "")])
    assert chunk.sentence_coords_within((12, 16)) == ["cB"]
    assert chunk.sentence_coords_within((0, 16)) == ["cA", "cB"]


def test_no_span_means_every_sentence_with_coords():
    chunk = _sentenced("A. B.", [("A.", "cA"), ("B.", "")])
    assert chunk.sentence_coords_within(None) == ["cA"]


def test_a_sentence_missing_from_the_text_is_skipped_not_misplaced():
    chunk = _sentenced("Alpha one. Gamma.",
                       [("Alpha one.", "cA"), ("altered", "cX"), ("Gamma.", "cG")])
    assert chunk.sentence_coords_within((11, 17)) == ["cG"]


def test_answer_survives_pickle_roundtrip():
    original = Answer(
        text="answer", context="context", page_number=1, title="Title",
        section="body", start_char=0, end_char=6, score=0.9, query="q",
        sentence_coords=["c1"], retrieval_score=0.5, rerank_score=0.4,
        pdf_path="/tmp/x.pdf", pdf_hash="hash",
    )
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original


def test_ingest_items_from_folder_titles_by_file_stem(tmp_path):
    for paper_id in ("1605.03481", "1601.02403"):
        (tmp_path / f"{paper_id}.pdf").write_bytes(b"%PDF-1.4")

    items = ingest_items_from_folder(str(tmp_path))

    assert [item.title for item in items] == ["1601.02403", "1605.03481"]
    assert all(item.source.path.endswith(".pdf") for item in items)


def test_ingest_items_from_folder_ignores_non_pdf_files(tmp_path):
    (tmp_path / "notes.txt").write_text("not a pdf")
    (tmp_path / "1601.02403.pdf").write_bytes(b"%PDF-1.4")

    assert [item.title for item in ingest_items_from_folder(str(tmp_path))] == ["1601.02403"]


def test_ingest_items_from_folder_rejects_a_non_directory(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(ValueError, match="Not a directory"):
        ingest_items_from_folder(str(missing))
