"""Data models for the Zotero RAG system."""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pdf_utils import compute_file_hash, compute_stream_hash


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
    sentence_count: int = 0  # number of sentences in this chunk
    sentences: list[tuple[str, str]] = field(default_factory=list)  # List of (sentence_text, coords)
    
    def __reduce__(self):
        """Custom pickle support for dataclass."""
        return (
            self.__class__,
            (
                self.text,
                self.page_number,
                self.chunk_index,
                self.title,
                self.pdf_hash,
                self.section,
                self.sentence_count,
                self.sentences,
            )
        )


@dataclass
class RerankedChunk:
    """A chunk with its retrieval and rerank scores."""
    chunk: Chunk
    retrieval_score: float
    rerank_score: float


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
    color: tuple[float, float, float] = field(default_factory=lambda: (1, 1, 0))
    sentence_coords: list[str] = field(default_factory=list)  # TEI coordinates for highlighting
    retrieval_score: float = 0.0  # Semantic search distance/score
    rerank_score: float = 0.0  # CrossEncoder reranking score
    pdf_path: str | None = None
    pdf_hash: str | None = None
    
    def __reduce__(self):
        """Custom pickle support for dataclass."""
        return (
            self.__class__,
            (self.text, self.context, self.page_number, self.title,
             self.section, self.start_char, self.end_char, self.score, self.query, self.color,
             self.sentence_coords, self.retrieval_score, self.rerank_score, self.pdf_path, self.pdf_hash)
        )


@dataclass
class ExpandedAnswerSpan:
    """Represents an answer span expanded to full sentences."""
    text: str
    start_char: int
    end_char: int
    sentence_coords: list[str] = field(default_factory=list)


@dataclass
class CachedPDF:
    """Represents a cached PDF stored by content hash."""
    pdf_hash: str
    title: str
    cache_path: str
    newly_cached: bool = False


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

    def compute_hash(self, buffer_size: int = 1024 * 1024) -> str:
        ...

    def write_to(self, dest_path: str, buffer_size: int = 1024 * 1024) -> None:
        ...


@dataclass
class UploadSource:
    """PDF source backed by an in-memory upload."""
    uploaded_file: Any

    def compute_hash(self, buffer_size: int = 1024 * 1024) -> str:
        return compute_stream_hash(self.uploaded_file, buffer_size=buffer_size)

    def write_to(self, dest_path: str, buffer_size: int = 1024 * 1024) -> None:
        self.uploaded_file.seek(0)
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(self.uploaded_file, out_file, length=buffer_size)
        self.uploaded_file.seek(0)


@dataclass
class PathSource:
    """PDF source backed by a file path."""
    path: str

    def compute_hash(self, buffer_size: int = 1024 * 1024) -> str:
        return compute_file_hash(self.path, buffer_size=buffer_size)

    def write_to(self, dest_path: str, buffer_size: int = 1024 * 1024) -> None:
        shutil.copy2(self.path, dest_path)


@dataclass
class PDFIngestItem:
    """Represents a PDF ingestion source."""
    title: str
    source: PDFSource


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
