"""Data models for the Zotero RAG system."""

from dataclasses import dataclass, field
import hashlib
import shutil
from typing import Any, Dict, List, Tuple, Optional, Protocol


@dataclass
class Paragraph:
    """Represents a paragraph-level chunk for QA."""
    text: str
    page_num: int
    para_idx: int
    title: str
    pdf_hash: str
    section: str = "body"  # section type: body, abstract, intro, etc.
    sentence_count: int = 0  # number of sentences in this paragraph
    sentences: List[Tuple[str, str]] = field(default_factory=list)  # List of (sentence_text, coords)
    
    def __reduce__(self):
        """Custom pickle support for dataclass."""
        return (
            self.__class__,
            (
                self.text,
                self.page_num,
                self.para_idx,
                self.title,
                self.pdf_hash,
                self.section,
                self.sentence_count,
                self.sentences,
            )
        )


@dataclass
class Answer:
    """Represents an extracted answer to a question."""
    text: str  # The answer text extracted from passage
    context: str  # Full paragraph context
    page_num: int
    title: str
    section: str = "body"
    start_char: int = 0  # Character position in context where answer starts
    end_char: int = 0  # Character position in context where answer ends
    score: float = 0.0  # QA model confidence score
    query: str = ""
    color: Tuple[float, float, float] = field(default_factory=lambda: (1, 1, 0))
    sentence_coords: List[str] = field(default_factory=list)  # TEI coordinates for highlighting
    retrieval_score: float = 0.0  # Semantic search distance/score
    rerank_score: float = 0.0  # CrossEncoder reranking score
    pdf_path: Optional[str] = None
    
    def __reduce__(self):
        """Custom pickle support for dataclass."""
        return (
            self.__class__,
            (self.text, self.context, self.pdf_path, self.page_num, self.title,
             self.section, self.start_char, self.end_char, self.score, self.query, self.color, 
             self.sentence_coords, self.retrieval_score, self.rerank_score)
        )


@dataclass
class IngestResult:
    """Summary of an ingestion operation."""

    ingested_titles: List[str] = field(default_factory=list)
    duplicate_title_titles: List[str] = field(default_factory=list)
    failed_uploads: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class UpsertResult:
    """Summary of an indexing operation."""

    indexed_chunks: int = 0
    processed_pdfs: int = 0
    already_indexed_titles: List[str] = field(default_factory=list)
    failed_pdfs: List[Dict[str, str]] = field(default_factory=list)


class PDFSource(Protocol):
    """Strategy interface for PDF ingestion sources."""

    def compute_hash(self, chunk_size: int = 1024 * 1024) -> str:
        ...

    def write_to(self, dest_path: str, chunk_size: int = 1024 * 1024) -> None:
        ...


@dataclass
class UploadSource:
    """PDF source backed by an in-memory upload."""

    uploaded_file: Any

    # def compute_hash(self, chunk_size: int = 1024 * 1024) -> str: TODO: non penso servano più
    #     h = hashlib.sha256()
    #     self.uploaded_file.seek(0)
    #     for chunk in iter(lambda: self.uploaded_file.read(chunk_size), b""):
    #         h.update(chunk)
    #     self.uploaded_file.seek(0)
    #     return h.hexdigest()

    def write_to(self, dest_path: str, chunk_size: int = 1024 * 1024) -> None:
        self.uploaded_file.seek(0)
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(self.uploaded_file, out_file, length=chunk_size)
        self.uploaded_file.seek(0)


@dataclass
class PathSource:
    """PDF source backed by a file path."""

    path: str

    # def compute_hash(self, chunk_size: int = 1024 * 1024) -> str:  TODO: non penso servano più
    #     h = hashlib.sha256()
    #     with open(self.path, "rb") as f:
    #         for chunk in iter(lambda: f.read(chunk_size), b""):
    #             h.update(chunk)
    #     return h.hexdigest()

    def write_to(self, dest_path: str, chunk_size: int = 1024 * 1024) -> None:
        shutil.copy2(self.path, dest_path)


@dataclass
class PDFIngestItem:
    """Represents a PDF ingestion source."""
    title: str
    source: PDFSource