"""PDF folder source - alternative to Zotero database."""

import os
import logging
import re
from typing import List, Dict, Union

from streamlit.runtime.uploaded_file_manager import UploadedFile

from models import PDFIngestItem, UploadSource, CachedPDF

logger = logging.getLogger(__name__)

class PDFCacheHandler:
    """Handles PDFs cache, ensuring unique filenames and proper storage."""
    
    def __init__(self, folder_path: str):
        """Initialize PDF folder source.
        
        Args:
            folder_path: Path to folder containing PDF files.
        """
        if not folder_path:
            raise ValueError("Folder path cannot be empty")

        self.folder_path = os.path.abspath(folder_path)
        os.makedirs(self.folder_path, exist_ok=True)

        if not os.path.isdir(self.folder_path):
            raise ValueError(f"Path is not a directory: {self.folder_path}")

        logger.info(f"Initialized PDFCacheHandler with folder: {self.folder_path}")

    @staticmethod
    def _is_hash_name(name: str) -> bool:
        return re.fullmatch(r"[a-fA-F0-9]{64}", name or "") is not None
    
    def get_items_from_upload(self, uploaded_files: List[UploadedFile]) -> List[PDFIngestItem]: #TODO: dovrebbe essere privato in zoteroRAG
        items: List[PDFIngestItem] = []
        for uploaded_file in uploaded_files:
            if uploaded_file is None:
                logger.warning("Skipping None uploaded file")
                continue

            title = os.path.splitext(uploaded_file.name)[0]
            items.append(PDFIngestItem(title=title, source=UploadSource(uploaded_file)))

        return items

    def ingest_pdf(self, uploaded_pdf: PDFIngestItem) -> CachedPDF:
        if uploaded_pdf is None:
            raise ValueError("uploaded_pdf cannot be None")

        pdf_hash = uploaded_pdf.source.compute_hash()
        candidate_filename = f"{pdf_hash}.pdf"
        candidate_path = os.path.join(self.folder_path, candidate_filename)

        created = False
        if not os.path.exists(candidate_path):
            try:
                uploaded_pdf.source.write_to(candidate_path)
                created = True
            except Exception as e:
                raise IOError(f"Failed to ingest PDF '{uploaded_pdf.title}'") from e

        title = uploaded_pdf.title or "Unknown"
        return CachedPDF(
            pdf_hash=pdf_hash,
            title=title,
            cache_path=candidate_path,
            created=created,
        )

    def remove_pdf(self, pdf_hash: str) -> bool:
        """Remove PDF file from cache by hash.
        
        Args:
            pdf_hash: SHA-256 hash of the PDF to remove.
        Returns:
            True if deletion was successful, False otherwise.
        """
        if not pdf_hash:
            logger.warning("PDF hash cannot be empty for PDF removal")
            return False

        candidate_filename = f"{pdf_hash}.pdf"
        candidate_path = os.path.join(self.folder_path, candidate_filename)

        if os.path.exists(candidate_path):
            try:
                os.remove(candidate_path)
                logger.info(f"Removed PDF file: {candidate_path}")
                return True
            except Exception as e:
                logger.error(f"Error removing PDF file '{candidate_path}': {e}")
                return False
        else:
            logger.warning(f"PDF file not found for removal: {candidate_path}")
            return False
        
    def clear_index_cache(self, deleted_pdfs: Dict[str, str]) -> None:
        """Remove all PDF files from cache that are listed in deleted_pdfs."""
        if not os.path.isdir(self.folder_path):
            logger.warning(f"Cache folder does not exist for clearing: {self.folder_path}")
            return
        
        for pdf_hash, title in deleted_pdfs.items():
            if not pdf_hash:
                logger.warning("Skipping deletion for entry with empty hash")
                continue

            if self.remove_pdf(pdf_hash):
                logger.info(f"Cleared cached PDF for deleted entry: {title} (hash: {pdf_hash})")
            else:
                raise IOError(f"Failed to clear cached PDF for deleted entry: {title} (hash: {pdf_hash})")

    def get_pdf_path(self, pdf_hash: str) -> Union[str, None]:
        """Get full path of a cached PDF by hash."""
        if not pdf_hash:
            logger.warning("PDF hash cannot be empty for getting PDF path")
            return None

        candidate_path = os.path.join(self.folder_path, f"{pdf_hash}.pdf")
        if os.path.isfile(candidate_path):
            return candidate_path

        logger.warning(f"PDF file not found for hash '{pdf_hash}': {candidate_path}")
        return None

    def get_cached_items(self) -> List[str]:
        """Get PDF items from the folder.
            
        Returns:
            List of cached PDF hashes
        """
        items = set()
        
        # Walk through the folder and find all PDFs
        for _root, _dirs, files in os.walk(self.folder_path):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    base_name = os.path.splitext(filename)[0]
                    if self._is_hash_name(base_name):
                        items.add(base_name.lower())
                    else:
                        logger.warning("Ignoring non-hash cached PDF: %s", filename)

        logger.info(f"Found {len(items)} PDF files in {self.folder_path}")
        return list(items)
    
    def list_collections(self) -> List[Dict]:
        """Return empty list for API compatibility with ZoteroDatabase.
        
        Returns:
            Empty list (folders don't have collections).
        """
        return []
