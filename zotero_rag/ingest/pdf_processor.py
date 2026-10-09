"""PDF processing using GROBID service."""

import logging
import os
import shutil
import tempfile
import threading
import xml.etree.ElementTree as ET

import requests
from grobid_client.grobid_client import GrobidClient

from zotero_rag.models import ExtractedChunk, Reference

logger = logging.getLogger(__name__)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def reference_from_tei(tei_root: ET.Element) -> Reference:
    """Authors and year of the paper itself, from the TEI header.

    The header's own biblStruct only: the bibliography at the back holds the
    same elements for every paper the text cites.
    """
    paper = tei_root.find(".//tei:teiHeader//tei:sourceDesc/tei:biblStruct", TEI_NS)
    if paper is None:
        return Reference([], None)
    surnames = [s.text.strip() for s in paper.findall("tei:analytic/tei:author/tei:persName/tei:surname", TEI_NS)
                if s.text and s.text.strip()]
    date = paper.find(".//tei:date[@when]", TEI_NS)
    year = date.get("when")[:4] if date is not None else None
    return Reference(surnames, year)


# Heading keyword -> section type; the first keyword the heading contains wins.
SECTION_KEYWORDS = {
    'abstract': 'abstract', 'introduction': 'introduction',
    'method': 'methods', 'procedure': 'methods',
    'result': 'results', 'discussion': 'discussion',
    'conclusion': 'conclusion',
}


def section_type(head_text: str) -> str:
    """The section a lowercased heading names, 'body' when no keyword matches."""
    return next((value for key, value in SECTION_KEYWORDS.items() if key in head_text), 'body')


def _paragraphs(tei_root: ET.Element):
    """(section type, <p> element) in reading order: the abstract, then each body division."""
    abstract = tei_root.find('.//tei:abstract', TEI_NS)
    if abstract is not None:
        for p_elem in abstract.findall('.//tei:p', TEI_NS):
            yield "abstract", p_elem

    body = tei_root.find('.//tei:body', TEI_NS)
    if body is None:
        return
    for division in body.findall('tei:div', TEI_NS):
        head = division.find('tei:head', TEI_NS)
        section = section_type("".join(head.itertext()).strip().lower() if head is not None else "")
        for p_elem in division.findall('.//tei:p', TEI_NS):
            yield section, p_elem


def _paragraph_content(p_elem: ET.Element) -> tuple[str, list[tuple[str, str]], int]:
    """A <p>'s text, its (sentence, coords) pairs, and the 0-based page of its first box."""
    sentences = []
    page_number = 0
    for s in p_elem.findall('.//tei:s', TEI_NS):
        sentence_text = " ".join("".join(s.itertext()).split())
        coords = s.get('coords', '')
        if sentence_text:
            sentences.append((sentence_text, coords))
            if page_number == 0 and coords:
                try:
                    page_number = int(coords.split(',')[0]) - 1
                except ValueError:
                    pass
    return ' '.join(text for text, _ in sentences), sentences, page_number


class PDFProcessor:
    """Handles PDF parsing and text extraction using GROBID."""

    # Minimum word count before a paragraph is treated as standalone.
    MIN_PARAGRAPH_WORDS = 10
    
    # Serialize calls to GROBID to avoid exhausting its internal pool
    GROBID_LOCK = threading.Lock()
    
    def __init__(self, 
                grobid_url: str = "http://localhost:8070", 
                grobid_timeout: int = 180, 
                output_base_dir: str | None = None):
        """Initialize PDF processor.
        
        Args:
            grobid_url: URL of the GROBID service.
            grobid_timeout: Timeout in seconds for GROBID requests.
            output_base_dir: Base directory for storing cache TEI XML outputs.
        """
        self.grobid_url = grobid_url
        self.grobid_timeout = grobid_timeout
        self.grobid_client = None  # Lazy initialization only when needed

        self.pdf_cache_dir = os.path.join(output_base_dir, "pdf_cache")
        self.tei_cache_dir = os.path.join(output_base_dir, "tei_cache")
        os.makedirs(self.tei_cache_dir, exist_ok=True)

        logger.info(f"Initialized PDFProcessor with folder: {self.tei_cache_dir}")
    
    def is_alive(self) -> bool:
        """Quick health check for the GROBID service."""
        try:
            resp = requests.get(f"{self.grobid_url}/api/isalive", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
        
    def remove_tei_cache(self, pdf_hash: str) -> bool:
        """Remove the cached TEI XML for a given PDF hash.

        Args:
            pdf_hash: Hash of the PDF whose cache should be removed.

        Returns:
            True if a cache file was removed, False otherwise.
        """
        if not pdf_hash:
            logger.warning("PDF hash cannot be empty for cache removal")
            return False

        cache_paths = [
            self._tei_cache_path(pdf_hash),
            os.path.join(self.tei_cache_dir, f"{pdf_hash}.tei"),
        ]

        removed = False
        for cache_path in cache_paths:
            if not os.path.exists(cache_path):
                continue
            try:
                os.remove(cache_path)
                logger.info(f"Removed cached TEI XML file: {cache_path}")
                removed = True
            except Exception as e:
                raise OSError(
                    f"Error removing cached TEI XML file '{cache_path}': {e}"
                ) from e

        if not removed:
            logger.warning(
                f"Cache item not found for removal (hash: {pdf_hash})"
            )

        return removed

    def clear_tei_cache(self, deleted_pdfs: dict[str, str]):
        """Clear cached TEI XML files for deleted PDFs.
        
        Args:
            deleted_pdfs: Dictionary mapping PDF hashes to their titles for logging.
        """
        if not os.path.isdir(self.tei_cache_dir):
            logger.warning(f"Cache directory does not exist for clearing: {self.tei_cache_dir}")
            return
        
        for pdf_hash in deleted_pdfs:
            if not pdf_hash:
                logger.warning("Skipping cache clearing for entry with empty hash")
                continue
            self.remove_tei_cache(pdf_hash)

    def _tei_cache_path(self, pdf_hash: str) -> str:
        return os.path.join(self.tei_cache_dir, f"{pdf_hash}.tei.xml")

    def _parse_pdf(self, pdf_path: str, pdf_hash: str) -> ET.Element | None:
        """Parse a single PDF using GROBID and return TEI XML root.

        Args:
            pdf_path: Path to the PDF file.
            pdf_hash: Hash naming the PDF's TEI cache entry.

        Returns:
            XML Element root of the TEI document, or None if parsing failed.
        """
        try:
            cache_path = self._tei_cache_path(pdf_hash)

            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        return ET.fromstring(f.read())
                except Exception:
                    logger.debug("Cached TEI for %s unreadable, reprocessing", pdf_hash, exc_info=True)

            content = self._run_grobid(pdf_path)
            if content is None:
                logger.warning(f"GROBID client did not produce TEI for {pdf_path}")
                return None
            # Persist to cache for reuse
            try:
                with open(cache_path, "wb") as out_f:
                    out_f.write(content)
            except Exception:
                logger.debug("Unable to write TEI cache %s", cache_path, exc_info=True)
            return ET.fromstring(content)
        except ConnectionError:
            raise
        except Exception as e:  # noqa: BLE001 - GROBID failure: skip this PDF
            logger.error(f"Error parsing PDF with GROBID: {e}")
            return None

    def _run_grobid(self, pdf_path: str) -> bytes | None:
        """GROBID's TEI for the PDF, or None if it produced none; ConnectionError if it is down."""
        # Only check GROBID availability if we need to parse (cache miss)
        if not self.is_alive():
            message = (
                f"GROBID service not reachable at {self.grobid_url}. "
                "Start GROBID and retry."
            )
            logger.error(message)
            raise ConnectionError(message)

        # Lazy initialization of GROBID client only when needed
        if self.grobid_client is None:
            self.grobid_client = GrobidClient(grobid_server=self.grobid_url)

        with self.GROBID_LOCK:
            in_dir = tempfile.mkdtemp(prefix="grobid_in_")
            out_dir = tempfile.mkdtemp(prefix="grobid_out_")
            try:
                base = os.path.basename(pdf_path)
                shutil.copy2(pdf_path, os.path.join(in_dir, base))

                # Process with sentence segmentation and coordinates
                self.grobid_client.process(
                    service="processFulltextDocument",
                    input_path=in_dir,
                    output=out_dir,
                    n=1,
                    tei_coordinates=True,
                    segment_sentences=True
                )

                tei_path = os.path.join(out_dir, os.path.splitext(base)[0] + ".tei.xml")
                if not os.path.exists(tei_path):
                    candidates = [f for f in os.listdir(out_dir) if f.endswith(".tei.xml")]
                    if not candidates:
                        return None
                    tei_path = os.path.join(out_dir, candidates[0])
                with open(tei_path, "rb") as f:
                    return f.read()
            finally:
                shutil.rmtree(in_dir, ignore_errors=True)
                shutil.rmtree(out_dir, ignore_errors=True)

    def _extract_chunks_from_tei(self, tei_root: ET.Element) -> list[ExtractedChunk]:
        """One chunk per paragraph, in reading order.

        A paragraph too short to stand alone is merged into the next one, and a
        short one left at the end into the last chunk.
        """
        chunks: list[ExtractedChunk] = []
        pending_text, pending_sentences = "", []
        for section, p_elem in _paragraphs(tei_root):
            text, sentences, page_number = _paragraph_content(p_elem)
            if not text:
                continue

            if pending_text:
                text = f"{pending_text} {text}"
                sentences = pending_sentences + sentences
                pending_text, pending_sentences = "", []

            if len(text.split()) < self.MIN_PARAGRAPH_WORDS:
                pending_text, pending_sentences = text, sentences
                continue

            chunks.append(ExtractedChunk(text=text, page_number=page_number,
                                         chunk_index=len(chunks), section=section,
                                         sentences=sentences))

        if pending_text and chunks:
            chunks[-1].text = f"{chunks[-1].text} {pending_text}"
            chunks[-1].sentences.extend(pending_sentences)

        return chunks

    def extract_text_chunks(self, pdf_hash: str, pdf_title: str | None = None) -> list[ExtractedChunk]:
        """Extract chunks from PDF using GROBID.
        
        Args:
            pdf_hash: Hash of the cached PDF file.
            pdf_title: Optional display title for logging.
            
        Returns:
            List of ExtractedChunk objects.
        """
        return self._extract_chunks_from_tei(self._tei(pdf_hash, pdf_title))

    def extract_reference(self, pdf_hash: str) -> Reference:
        """Authors and year from the PDF's header; the TEI is cached, so no new GROBID call."""
        return reference_from_tei(self._tei(pdf_hash))

    def _tei(self, pdf_hash: str, pdf_title: str | None = None) -> ET.Element:
        """The cached PDF's TEI, parsed by GROBID on first use; ValueError if it cannot be."""
        if not pdf_hash:
            raise ValueError("pdf_hash cannot be empty")

        pdf_path = os.path.join(self.pdf_cache_dir, f"{pdf_hash}.pdf")

        tei_root = self._parse_pdf(pdf_path, pdf_hash=pdf_hash)
        if tei_root is None:
            title_info = f" ({pdf_title})" if pdf_title else ""
            message = (
                f"GROBID parsing failed for {pdf_path}{title_info}; no chunks extracted"
            )
            logger.warning(message)
            raise ValueError(message)
        return tei_root