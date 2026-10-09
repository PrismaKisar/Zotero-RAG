"""Data models for the Zotero RAG system."""

import os
import shutil
from dataclasses import dataclass, field, fields
from itertools import accumulate
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from zotero_rag.ingest.pdf_utils import compute_file_hash, compute_stream_hash

# The highlight colour when none is chosen, as RGB in 0-1.
YELLOW = (1.0, 1.0, 0.0)


@dataclass
class ExtractedChunk:
    """A chunk extracted from a TEI paragraph, before PDF metadata is attached."""
    text: str
    page_number: int
    chunk_index: int
    section: str
    sentences: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Chunk:
    """The unit of retrieval: one TEI paragraph, indexed and searched as a whole."""
    text: str
    page_number: int
    chunk_index: int
    title: str
    pdf_hash: str
    section: str = "body"  # section type: body, abstract, intro, etc.
    sentences: list[tuple[str, str]] = field(default_factory=list)  # List of (sentence_text, coords)

    @property
    def key(self) -> tuple[str, int]:
        """The chunk's identity: its paper and its position in it."""
        return self.pdf_hash, self.chunk_index

    def expand_to_sentences(self, start_char: int, end_char: int) -> "ExpandedAnswerSpan":
        """Widen a character span to the complete sentences it touches.

        Sentence positions are counted as if the sentences were joined by one
        space each, which is how the extractor builds ``text`` from them.

        Args:
            start_char: Start position of the span in ``text``.
            end_char: End position of the span in ``text``.

        Returns:
            The widened span, with the coordinates of its sentences.
        """
        if not self.sentences:
            return ExpandedAnswerSpan(
                text=self.text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
                sentence_coords=[],
            )

        # Where each sentence ends, counting the one space after it.
        bounds = list(accumulate(len(text) + 1 for text, _ in self.sentences))
        last = next((i for i, bound in enumerate(bounds) if end_char <= bound), len(bounds) - 1)
        first = next((i for i, bound in enumerate(bounds[:last + 1]) if start_char < bound), 0)

        involved_sentences = self.sentences[first:last + 1]
        expanded_text = " ".join(text for text, _ in involved_sentences)
        sentence_coords = [coords for _, coords in involved_sentences if coords]
        new_start = bounds[first - 1] if first else 0
        new_end = new_start + len(expanded_text)

        return ExpandedAnswerSpan(
            text=expanded_text,
            start_char=new_start,
            end_char=new_end,
            sentence_coords=sentence_coords,
        )

    def sentence_coords_within(self, span: tuple[int, int] | None) -> list[str]:
        """Coordinates of the sentences ``span`` touches, or all of them if it is None.

        The highlighter draws per sentence, so narrowing the character span without
        narrowing the coordinates would report a tight citation and still paint the
        whole paragraph.

        ponytail: sentence offsets are recovered by scanning ``text`` rather than
        stored, since nothing else needs them; a sentence the extractor altered is
        skipped instead of misplacing every sentence after it. This scan and the
        one-space arithmetic of ``expand_to_sentences`` are two readings of the same
        layout, kept apart because each reader was measured with its own.
        """
        if span is None:
            return [coords for _, coords in self.sentences if coords]
        start, end = span
        found_coords, cursor = [], 0
        for sentence, coords in self.sentences:
            at = self.text.find(sentence, cursor)
            if at < 0:
                continue
            cursor = at + len(sentence)
            if coords and at < end and cursor > start:
                found_coords.append(coords)
        return found_coords


@dataclass
class RerankedChunk:
    """A chunk with its retrieval and rerank scores."""
    chunk: Chunk
    retrieval_score: float
    rerank_score: float


@dataclass
class RetrievedChunk:
    """A retrieval candidate. A neighbour was pulled in beside a hit and never
    scored against the query: its score is the hit's, borrowed."""
    chunk: Chunk
    retrieval_score: float
    is_neighbour: bool = False


@dataclass
class QuestionTrace:
    """What one ``answer_question`` call did at each stage, for the benchmarks.

    ``retrieved`` is the retrieval order: hits best first, ties by chunk id,
    then neighbours best first. ``reranked`` is what the reader was given.
    ``stage_times`` holds wall-clock seconds per stage reached; total latency
    alone cannot say whether a slow reader or a slow retriever is responsible,
    and the two carry opposite fixes.
    """
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    reranked: list[RerankedChunk] = field(default_factory=list)
    stage_times: dict[str, float] = field(default_factory=dict)


@dataclass
class Answer:
    """Represents an extracted answer to a question."""
    text: str  # The answer text extracted from passage
    context: str  # Full text of the chunk the answer was extracted from
    page_number: int
    title: str
    section: str = "body"
    start_char: int = 0  # Character position in context where answer starts
    end_char: int = 0  # Character position in context where answer ends
    score: float = 0.0  # QA model confidence score
    query: str = ""
    color: tuple[float, float, float] = YELLOW
    sentence_coords: list[str] = field(default_factory=list)  # TEI coordinates for highlighting
    retrieval_score: float = 0.0  # Semantic search distance/score
    rerank_score: float = 0.0  # CrossEncoder reranking score
    pdf_path: str | None = None
    pdf_hash: str | None = None

    @classmethod
    def from_candidate(cls, candidate: "RerankedChunk", *, text: str, start_char: int,
                       end_char: int, score: float, query: str,
                       color: tuple[float, float, float],
                       sentence_coords: list[str]) -> "Answer":
        """An answer drawn from ``candidate``: where its chunk sits, and how it scored."""
        chunk = candidate.chunk
        return cls(text=text, context=chunk.text, page_number=chunk.page_number,
                   title=chunk.title, section=chunk.section, start_char=start_char,
                   end_char=end_char, score=score, query=query, color=color,
                   sentence_coords=sentence_coords, retrieval_score=candidate.retrieval_score,
                   rerank_score=candidate.rerank_score, pdf_hash=chunk.pdf_hash)


class Reader(Protocol):
    """The seam between reranking and answers: QAEngine and GenerativeReader.

    ``reader_kind`` names everything that changes what the reader does, so two
    readers of one kind are interchangeable and a loaded one can be reused.
    ``last_quote_stats`` describes the last call, with ``cited`` 0 when nothing
    was asked to quote; ``generation_failures`` counts the calls lost to the
    service since it was last reset. A reader that cannot fail or quote keeps
    both at zero rather than lacking them.
    """

    reader_kind: str
    last_quote_stats: dict[str, int]
    generation_failures: int

    def extract_answers(self, question: str, candidates: list[RerankedChunk],
                        config: dict, color: tuple[float, float, float] = YELLOW,
                        progress_callback=None) -> list[Answer]:
        ...


@dataclass
class ExpandedAnswerSpan:
    """Represents an answer span expanded to full sentences."""
    text: str
    start_char: int
    end_char: int
    sentence_coords: list[str] = field(default_factory=list)


def from_stored(cls, data: dict):
    """A dataclass from a dict saved by an earlier version: fields it no longer has are dropped."""
    return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


@dataclass
class Reference:
    """Who wrote a paper and when, and its Zotero attachment if it came from Zotero."""
    authors: list[str]  # surnames, in the paper's order
    year: str | None
    zotero_key: str | None = None
    zotero_library: str = "library"  # or "groups/<id>" for a group library

    def cite(self) -> str:
        """Author-year, as a reader expects it: "Dasigi et al., 2021"."""
        if len(self.authors) > 2:
            who = f"{self.authors[0]} et al."
        else:
            who = " and ".join(self.authors)
        return ", ".join(part for part in (who, self.year) if part)

    def zotero_link(self, page: int) -> str | None:
        """Opens the PDF in Zotero's own reader at ``page`` (1-based)."""
        if not self.zotero_key:
            return None
        return f"zotero://open-pdf/{self.zotero_library}/items/{self.zotero_key}?page={page}"


@dataclass
class CachedPDF:
    """Represents a cached PDF stored by content hash."""
    pdf_hash: str
    title: str
    cache_path: str
    newly_cached: bool = False
    reference: Reference | None = None


@dataclass
class IngestResult:
    """Summary of an ingestion operation."""
    ingested_pdfs: list[CachedPDF] = field(default_factory=list)
    failed_pdfs: list[dict[str, str]] = field(default_factory=list)


@dataclass
class UpsertResult:
    """Summary of an indexing operation."""
    indexed_chunks: int = 0
    processed_pdfs: int = 0
    already_indexed: list[dict[str, str]] = field(default_factory=list)
    title_overrides: list[dict[str, str]] = field(default_factory=list)
    duplicate_titles: list[str] = field(default_factory=list)
    failed_pdfs: list[dict[str, str]] = field(default_factory=list)


class PDFSource(Protocol):
    """Strategy interface for PDF ingestion sources."""

    def compute_hash(self) -> str:
        ...

    def write_to(self, dest_path: str) -> None:
        ...


@dataclass
class UploadSource:
    """PDF source backed by an in-memory upload."""
    uploaded_file: Any

    def compute_hash(self) -> str:
        return compute_stream_hash(self.uploaded_file)

    def write_to(self, dest_path: str) -> None:
        self.uploaded_file.seek(0)
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(self.uploaded_file, out_file)
        self.uploaded_file.seek(0)


@dataclass
class PathSource:
    """PDF source backed by a file path."""
    path: str

    def compute_hash(self) -> str:
        return compute_file_hash(self.path)

    def write_to(self, dest_path: str) -> None:
        shutil.copy2(self.path, dest_path)


@dataclass
class PDFIngestItem:
    """Represents a PDF ingestion source."""
    title: str
    source: PDFSource
    reference: Reference | None = None  # from Zotero; None: read off the PDF when parsed


def ingest_items_from_folder(folder_path: str) -> list[PDFIngestItem]:
    """One PDFIngestItem per PDF in ``folder_path``, titled by its filename stem.

    Args:
        folder_path: Path to the folder to read PDFs from (not recursive).

    Returns:
        PDFIngestItem objects, sorted by filename.

    Raises:
        ValueError: If the path is not a directory.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    return [PDFIngestItem(title=pdf.stem, source=PathSource(str(pdf)))
            for pdf in sorted(folder.glob("*.pdf"))]


def ingest_items_from_upload(uploaded_files: list[BinaryIO]) -> list[PDFIngestItem]:
    """One PDFIngestItem per upload, titled by its file name's stem; None entries are skipped.

    Args:
        uploaded_files: Binary streams carrying the upload's file name in ``name``.
    """
    return [PDFIngestItem(title=os.path.splitext(upload.name)[0], source=UploadSource(upload))
            for upload in uploaded_files if upload is not None]
