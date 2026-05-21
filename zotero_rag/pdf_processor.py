"""PDF processing using GROBID service."""

import os
import logging
import threading
import tempfile
import shutil
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict
import requests
from grobid_client.grobid_client import GrobidClient

from pdf_utils import compute_file_hash

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF parsing and text extraction using GROBID."""
    
    # Reference section patterns to detect bibliography/references
    REFERENCE_PATTERNS = [
        r'^\s*references\s*$',
        r'^\s*bibliography\s*$',
        r'^\s*works\s+cited\s*$',
        r'^\s*literature\s+cited\s*$',
    ]
    
    # Section types to include in chunking (can be customized)
    CONTENT_SECTIONS = {
        'body': True,
        'abstract': True,
        'introduction': True,
        'conclusion': True,
        'results': True,
        'methods': True,
        'discussion': True,
    }
    
    # Serialize calls to GROBID to avoid exhausting its internal pool
    GROBID_LOCK = threading.Lock()
    
    def __init__(self, grobid_url: str = "http://localhost:8070", 
                 grobid_timeout: int = 180, 
                 output_base_dir: str = None):
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
        except Exception:
            return False
        
    @staticmethod
    def _is_hash_name(name: str) -> bool:
        return re.fullmatch(r"[a-fA-F0-9]{64}", name or "") is not None
    
    def remove_cache_item(self, pdf_hash: str) -> bool:
        """Remove cached TEI XML for a given PDF hash."""
        if not pdf_hash:
            logger.warning("PDF hash cannot be empty for cache removal")
            return False

        return self.remove_cache_item_by_hash(pdf_hash)

    def remove_cache_item_by_hash(self, pdf_hash: str) -> bool:
        """Remove cached TEI XML for a given PDF hash."""
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
                raise IOError(
                    f"Error removing cached TEI XML file '{cache_path}': {e}"
                ) from e

        if not removed:
            logger.warning(
                f"Cache item not found for removal (hash: {pdf_hash})"
            )

        return removed

    def clear_index_cache(self, deleted_pdfs: Dict[str, str]) -> None:
        """Remove cached TEI XML files for the deleted PDFs."""
        if not os.path.isdir(self.tei_cache_dir):
            logger.warning(f"Cache directory does not exist for clearing: {self.tei_cache_dir}")
            return
        
        for pdf_hash, title in deleted_pdfs.items():
            if not pdf_hash:
                logger.warning("Skipping cache clearing for entry with empty hash")
                continue
            self.remove_cache_item_by_hash(pdf_hash)

    
    def _parse_pdf(self, pdf_path: str, pdf_hash: Optional[str] = None) -> Optional[ET.Element]:
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
                if self._is_hash_name(base_name):
                    pdf_hash = base_name.lower()
                else:
                    pdf_hash = compute_file_hash(pdf_path)
            cache_path = os.path.join(self.tei_cache_dir, f"{pdf_hash}.tei.xml")
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        return ET.fromstring(f.read())
                except Exception:
                    pass  # fall through to reprocess if cache is unreadable
            
            # Only check GROBID availability if we need to parse (cache miss)
            if not self.is_alive():
                logger.error(f"GROBID not reachable at {self.grobid_url}")
                return None
            
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
                            pass
                        return ET.fromstring(content)
                    else:
                        logger.warning(f"GROBID client did not produce TEI for {pdf_path}")
                        return None
                finally:
                    shutil.rmtree(in_dir, ignore_errors=True)
                    shutil.rmtree(out_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error parsing PDF with GROBID: {e}")
            return None
    
    def _extract_paragraphs_from_tei(self, tei_root: ET.Element
            ) -> Tuple[List[Tuple[str, int, int, str, List[Tuple[str, str]]]], str]:
        """Extract paragraphs from TEI XML structure.
        
        Args:
            tei_root: Root element of TEI XML.
            
        Returns:
            - List of (paragraph_text, page_number, paragraph_index, section_type, sentences)
            - document_text: Full cleaned text of the PDF for context.
        """
        paragraphs = []
        full_text_parts = []
        para_idx = 0
        
        # Define TEI namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # Buffer to handle short paragraphs
        pending_short_para = ""
        pending_coords = []
        
        # Extract from abstract (each <p> is a paragraph)
        abstract = tei_root.find('.//tei:abstract', ns)
        if abstract is not None:
            for p_elem in abstract.findall('.//tei:p', ns):
                p_text, sentences_coords, page_num = self._process_paragraph_element(p_elem, ns)
                
                if p_text:
                    if pending_short_para:
                        p_text = f"{pending_short_para} {p_text}"
                        sentences_coords = pending_coords + sentences_coords
                        pending_short_para = ""
                        pending_coords = []

                    if len(p_text.split()) < 10:
                        pending_short_para = p_text
                        pending_coords = sentences_coords
                    else:
                        paragraphs.append((p_text, page_num, para_idx, 'abstract', sentences_coords))
                        full_text_parts.append(p_text)
                        para_idx += 1
        
        # Extract from body (main content)
        body = tei_root.find('.//tei:body', ns)
        if body is not None:
            for section_div in body.findall('tei:div', ns):
                # Determine section type from head element
                head = section_div.find('tei:head', ns)
                head_text_raw = "".join(head.itertext()).strip() if head is not None else ""
                head_text_lower = head_text_raw.lower()
                section_type = self._determine_section_type(head_text_lower)
                
                for p_elem in section_div.findall('.//tei:p', ns):
                    p_text, sentences_coords, page_num = self._process_paragraph_element(p_elem, ns)
                    
                    if p_text:
                        if pending_short_para:
                            p_text = f"{pending_short_para} {p_text}"
                            sentences_coords = pending_coords + sentences_coords
                            pending_short_para = ""
                            pending_coords = []

                        if len(p_text.split()) < 10:
                            pending_short_para = p_text
                            pending_coords = sentences_coords
                        else:
                            paragraphs.append((p_text, page_num, para_idx, section_type, sentences_coords))
                            full_text_parts.append(p_text)
                            para_idx += 1
        
        if pending_short_para and paragraphs:
            last_para = paragraphs[-1]
            updated_text = f"{last_para[0]} {pending_short_para}"
            updated_coords = last_para[4] + pending_coords
            paragraphs[-1] = (updated_text, last_para[1], last_para[2], last_para[3], updated_coords)
            full_text_parts.append(pending_short_para)

        document_text = "\n\n".join(full_text_parts)
        
        return paragraphs, document_text

    def _process_paragraph_element(self, p_elem, ns):
        """Estrae testo e coordinate da un elemento paragrafo <p>."""
        sentences_with_coords = []
        page_num = 0
        
        for s in p_elem.findall('.//tei:s', ns):
            sentence_text = " ".join("".join(s.itertext()).split())
            coords = s.get('coords', '')
            
            if sentence_text:
                sentences_with_coords.append((sentence_text, coords))
                if page_num == 0 and coords:
                    try:
                        page_num = int(coords.split(',')[0]) - 1
                    except ValueError: 
                        pass
                    
        paragraph_text = ' '.join([sent for sent, _ in sentences_with_coords])
        return paragraph_text, sentences_with_coords, page_num

    def _determine_section_type(self, head_text):
        """Mappa il titolo della sezione a una categoria."""
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
    
    def extract_text_chunks(self, pdf_hash: str, pdf_title: Optional[str] = None,
            ) -> Tuple[List[Tuple[str, int, int, str, List[Tuple[str, str]]]], str]:
        """Extract paragraphs from PDF using GROBID.
        
        Args:
            pdf_hash: Hash of the cached PDF file.
            pdf_title: Optional display title for logging.
            
        Returns:
            - List of (paragraph_text, page_number, paragraph_index, section_type, sentences) tuples.
            - Full document text reconstructed from extracted paragraphs.
        """
        if not pdf_hash:
            raise ValueError("pdf_hash cannot be empty")

        pdf_path = os.path.join(self.pdf_cache_dir, f"{pdf_hash}.pdf")

        tei_root = self._parse_pdf(pdf_path, pdf_hash=pdf_hash)
        if tei_root is None:
            title_info = f" ({pdf_title})" if pdf_title else ""
            logger.warning(f"GROBID parsing failed for {pdf_path}{title_info}; no paragraphs extracted")
            return [], ""

        paragraphs, document_text = self._extract_paragraphs_from_tei(tei_root)
        return paragraphs, document_text