"""PDF folder source - alternative to Zotero database."""

import os
import logging
import hashlib
from typing import List, Dict, Union

from streamlit.runtime.uploaded_file_manager import UploadedFile

from models import PDFIngestItem, UploadSource

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

    def _sanitize_filename(self, name: str) -> str:
        """Converts a string into a safe filename."""
        import re
        if not name:
            return "_All_Library"
        s = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        s = re.sub(r'(?u)[^-\w.]', '', s)
        return s
    
    def _compute_pdf_hash(self, file_path: str, chunk_size: int = 1024 * 1024) -> str:
            """Compute SHA-256 hash of a PDF file."""
            h = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    h.update(chunk)
            return h.hexdigest()
    
    def get_items_from_upload(self, uploaded_files: List[UploadedFile]) -> List[PDFIngestItem]: #TODO: dovrebbe essere privato in zoteroRAG
        items: List[PDFIngestItem] = []
        for uploaded_file in uploaded_files:
            if uploaded_file is None:
                logger.warning("Skipping None uploaded file")
                continue

            title = os.path.splitext(uploaded_file.name)[0]
            items.append(PDFIngestItem(title=title, source=UploadSource(uploaded_file)))

        return items

    def ingest_pdf(self, uploaded_pdf: PDFIngestItem) -> Dict[str, bool]:
        if uploaded_pdf is None:
            raise ValueError("uploaded_pdf cannot be None")

        base_title = self._sanitize_filename(uploaded_pdf.title)
        candidate_filename = f"{base_title}.pdf"
        candidate_path = os.path.join(self.folder_path, candidate_filename)

        if os.path.exists(candidate_path):
            logger.warning(
                "PDF with title '%s' already exist in cache, skipping upload",
                base_title,
            )
            return {
                "title": base_title,
                "duplicate_title": True,
            }

        try:
            uploaded_pdf.source.write_to(candidate_path)
        except Exception as e:
            raise IOError(f"Failed to ingest PDF '{uploaded_pdf.title}'") from e

        return {
            "title": base_title,
            "duplicate_title": False,
        }

    def remove_pdf(self, title: str) -> bool:
        """Remove PDF file from cache by title.
        
        Args:
            title: Title of the PDF to remove (without extension).
        Returns:
            True if deletion was successful, False otherwise.
        """
        if not title:
            logger.warning("Title cannot be empty for PDF removal")
            return False

        sanitized_title = self._sanitize_filename(title)
        candidate_filename = f"{sanitized_title}.pdf"
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
            if not title:
                logger.warning(f"Skipping deletion for PDF with empty title (hash: {pdf_hash})")
                continue

            if self.remove_pdf(title):
                logger.info(f"Cleared cached PDF for deleted entry: {title} (hash: {pdf_hash})")
            else:
                raise IOError(f"Failed to clear cached PDF for deleted entry: {title} (hash: {pdf_hash})")

    def get_pdf_path(self, title: str) -> Union[str, None]:
        """Get full path of a cached PDF by title."""
        if not title:
            logger.warning("Title cannot be empty for getting PDF path")
            return None

        sanitized_title = self._sanitize_filename(title)
        candidate_path = os.path.join(self.folder_path, f"{sanitized_title}.pdf")

        if os.path.isfile(candidate_path):
            return candidate_path

        logger.warning(f"PDF file not found for title '{title}': {candidate_path}")
        return None

    def get_cached_items(self) -> List[str]:
        """Get PDF items from the folder.
            
        Returns:
            List of cached PDFs titles
        """
        items = []
        
        # Walk through the folder and find all PDFs
        for _root, _dirs, files in os.walk(self.folder_path):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    title = self._sanitize_filename(os.path.splitext(filename)[0])
                    items.append(title)
        
        logger.info(f"Found {len(items)} PDF files in {self.folder_path}")
        return items
    
    def list_collections(self) -> List[Dict]:
        """Return empty list for API compatibility with ZoteroDatabase.
        
        Returns:
            Empty list (folders don't have collections).
        """
        return []
