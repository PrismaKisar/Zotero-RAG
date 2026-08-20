"""PDF processing using GROBID service."""

import logging
import os
import shutil
import tempfile
import threading
import xml.etree.ElementTree as ET
from typing import ClassVar

import requests
from grobid_client.grobid_client import GrobidClient
from models import ExtractedChunk
from pdf_utils import compute_file_hash, is_pdf_hash

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF parsing and text extraction using GROBID."""
    
    # Section types to include in chunking (can be customized)
    CONTENT_SECTIONS: ClassVar[dict[str, bool]] = {
        'body': True,
        'abstract': True,
        'introduction': True,
        'conclusion': True,
        'results': True,
        'methods': True,
        'discussion': True,
    }

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

        self.output_base_dir = output_base_dir 
        self.pdf_cache_dir = os.path.join(self.output_base_dir, "pdf_cache") or "pdf_cache"
        self.tei_cache_dir = os.path.join(output_base_dir, "tei_cache") or "tei_cache"
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
            os.path.join(self.tei_cache_dir, f"{pdf_hash}.tei.xml"),
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

    def _parse_pdf(self, pdf_path: str, pdf_hash: str | None = None) -> ET.Element | None:
        """Parse a single PDF using GROBID and return TEI XML root.
        
        Args:
            pdf_path: Path to the PDF file.
            pdf_hash: Precomputed hash for TEI cache naming.
            
        Returns:
            XML Element root of the TEI document, or None if parsing failed.
        """
        try:
            if not pdf_hash:
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                if is_pdf_hash(base_name):
                    pdf_hash = base_name.lower()
                else:
                    pdf_hash = compute_file_hash(pdf_path)
            cache_path = os.path.join(self.tei_cache_dir, f"{pdf_hash}.tei.xml")
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        return ET.fromstring(f.read())
                except Exception:
                    logger.debug("Cached TEI for %s unreadable, reprocessing", pdf_hash, exc_info=True)
            
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
                    temp_pdf = os.path.join(in_dir, base)
                    shutil.copy2(pdf_path, temp_pdf)

                    # Process with sentence segmentation and coordinates
                    self.grobid_client.process(
                        service="processFulltextDocument",
                        input_path=in_dir,
                        output=out_dir,
                        n=1,
                        tei_coordinates=True,
                        segment_sentences=True
                    )

                    expected = os.path.splitext(base)[0] + ".tei.xml"
                    tei_path = os.path.join(out_dir, expected)
                    if not os.path.exists(tei_path):
                        candidates = [
                            os.path.join(out_dir, f) 
                            for f in os.listdir(out_dir) 
                            if f.endswith(".tei.xml")
                        ]
                        tei_path = candidates[0] if candidates else None

                    if tei_path and os.path.exists(tei_path):
                        with open(tei_path, "rb") as f:
                            content = f.read()
                        # Persist to cache for reuse
                        try:
                            with open(cache_path, "wb") as out_f:
                                out_f.write(content)
                        except Exception:
                            logger.debug("Unable to write TEI cache %s", cache_path, exc_info=True)
                        return ET.fromstring(content)
                    else:
                        logger.warning(f"GROBID client did not produce TEI for {pdf_path}")
                        return None
                finally:
                    shutil.rmtree(in_dir, ignore_errors=True)
                    shutil.rmtree(out_dir, ignore_errors=True)
        except ConnectionError:
            raise
        except Exception as e:  # noqa: BLE001 - GROBID failure: skip this PDF
            logger.error(f"Error parsing PDF with GROBID: {e}")
            return None

    def _collect_chunks_from_elements(self,
                                    p_elements: list[ET.Element],
                                    section_type: str,
                                    chunks: list[ExtractedChunk],
                                    full_text_parts: list[str],
                                    chunk_index: int,
                                    pending_short_para: str,
                                    pending_coords: list[tuple[str, str]],
                                    ns: dict[str, str]) -> tuple[int, str, list[tuple[str, str]]]:
        """Turn a list of <p> elements into chunks, merging paragraphs too short to stand alone.
        
        Args:
            p_elements: List of <p> elements to process.
            section_type: Section type for categorization.
            chunks: List to append ExtractedChunk objects to.
            full_text_parts: List to append full text parts for context reconstruction.
            chunk_index: Current chunk index for ordering.
            pending_short_para: Buffer for short paragraph text that may need to be merged.
            pending_coords: Buffer for coordinates of sentences in the pending short paragraph.
            ns: Namespace dictionary for XML parsing.
                
        Returns:
            Updated chunk index, pending short paragraph text, and pending coordinates.
        """
        for p_elem in p_elements:
            p_text, sentences_coords, page_number = self._process_paragraph_element(p_elem, ns)

            if not p_text:
                continue

            if pending_short_para:
                p_text = f"{pending_short_para} {p_text}"
                sentences_coords = pending_coords + sentences_coords
                pending_short_para = ""
                pending_coords = []

            if len(p_text.split()) < self.MIN_PARAGRAPH_WORDS:
                pending_short_para = p_text
                pending_coords = sentences_coords
                continue

            chunks.append(
                ExtractedChunk(
                    text=p_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    section=section_type,
                    sentences=sentences_coords,
                )
            )
            full_text_parts.append(p_text)
            chunk_index += 1

        return chunk_index, pending_short_para, pending_coords
    
    def _extract_chunks_from_tei(self, tei_root: ET.Element) -> tuple[list[ExtractedChunk], str]:
        """Extract chunks from TEI XML structure.
        
        Args:
            tei_root: Root element of TEI XML.
            
        Returns:
            - List of ExtractedChunk objects
            - document_text: Full cleaned text of the PDF for context.
        """
        chunks: list[ExtractedChunk] = []
        full_text_parts = []
        chunk_index = 0
        
        # Define TEI namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # Buffer to merge paragraphs too short to stand alone as a chunk
        pending_short_para = ""
        pending_coords = []
        
        # Extract from abstract (each <p> is a paragraph)
        abstract = tei_root.find('.//tei:abstract', ns)
        if abstract is not None:
            chunk_index, pending_short_para, pending_coords = self._collect_chunks_from_elements(
                abstract.findall('.//tei:p', ns),
                "abstract",
                chunks,
                full_text_parts,
                chunk_index,
                pending_short_para,
                pending_coords,
                ns,
            )
        
        # Extract from body (main content)
        body = tei_root.find('.//tei:body', ns)
        if body is not None:
            for section_div in body.findall('tei:div', ns):
                # Determine section type from head element
                head = section_div.find('tei:head', ns)
                head_text_raw = "".join(head.itertext()).strip() if head is not None else ""
                head_text_lower = head_text_raw.lower()
                section_type = self._determine_section_type(head_text_lower)
                
                chunk_index, pending_short_para, pending_coords = self._collect_chunks_from_elements(
                    section_div.findall('.//tei:p', ns),
                    section_type,
                    chunks,
                    full_text_parts,
                    chunk_index,
                    pending_short_para,
                    pending_coords,
                    ns,
                )
        
        if pending_short_para and chunks:
            last_chunk = chunks[-1]
            last_chunk.text = f"{last_chunk.text} {pending_short_para}"
            last_chunk.sentences.extend(pending_coords)
            full_text_parts.append(pending_short_para)

        document_text = "\n\n".join(full_text_parts)
        title_text = self._extract_title_from_tei(tei_root, ns)
        if title_text:
            title_block = f"Title: {title_text}"
            document_text = f"{title_block}\n\n{document_text}" if document_text else title_block
        
        return chunks, document_text

    def _extract_title_from_tei(self, tei_root: ET.Element, ns: dict[str, str]) -> str:
        """Extract document title from TEI header when available.
        
        Args:
            tei_root: Root element of TEI XML.
            ns: Namespace dictionary for XML parsing.
            
        Returns:
            Extracted title text or empty string if not found.
        """
        title_xpaths = [
            ".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title[@type='main']",
            ".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title",
            ".//tei:teiHeader//tei:titleStmt/tei:title",
            ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct/tei:analytic/tei:title",
            ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct/tei:monogr/tei:title",
            ".//tei:teiHeader//tei:biblStruct/tei:analytic/tei:title",
            ".//tei:teiHeader//tei:biblStruct/tei:monogr/tei:title",
            ".//tei:biblStruct/tei:analytic/tei:title",
            ".//tei:biblStruct/tei:monogr/tei:title",
        ]

        for xpath in title_xpaths:
            for title_elem in tei_root.findall(xpath, ns):
                title_text = " ".join("".join(title_elem.itertext()).split())
                if title_text:
                    return title_text

        return ""

    def _process_paragraph_element(self, p_elem, ns):
        """Extract text and sentence coordinates from a <p> element, handling nested sentences.
        
        Args:
            p_elem: XML element representing a paragraph.
            ns: Namespace dictionary for XML parsing.
        """
        sentences_with_coords = []
        page_number = 0
        
        for s in p_elem.findall('.//tei:s', ns):
            sentence_text = " ".join("".join(s.itertext()).split())
            coords = s.get('coords', '')
            
            if sentence_text:
                sentences_with_coords.append((sentence_text, coords))
                if page_number == 0 and coords:
                    try:
                        page_number = int(coords.split(',')[0]) - 1
                    except ValueError: 
                        pass
                    
        paragraph_text = ' '.join([sent for sent, _ in sentences_with_coords])
        return paragraph_text, sentences_with_coords, page_number

    def _determine_section_type(self, head_text):
        """Determine section type based on head text, using simple keyword mapping.
        
        Args:
            head_text: Text content of the section head, normalized to lowercase.
        """
        if not head_text:
            return 'body'

        mapping = {
            'abstract': 'abstract', 'introduction': 'introduction',
            'method': 'methods', 'procedure': 'methods',
            'result': 'results', 'discussion': 'discussion',
            'conclusion': 'conclusion'
        }
        for key, value in mapping.items():
            if key in head_text: return value
        return 'body'
    
    def extract_text_chunks(self, pdf_hash: str, pdf_title: str | None = None) -> tuple[list[ExtractedChunk], str]:
        """Extract chunks from PDF using GROBID.
        
        Args:
            pdf_hash: Hash of the cached PDF file.
            pdf_title: Optional display title for logging.
            
        Returns:
            - List of ExtractedChunk objects.
            - Full document text reconstructed from extracted chunks.
        """
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

        chunks, document_text = self._extract_chunks_from_tei(tei_root)
        return chunks, document_text