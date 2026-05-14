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
    
    def get_items_from_upload(self, uploaded_files: List[UploadedFile]) -> List[PDFIngestItem]:
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
        candidate_title = base_title
        candidate_filename = f"{candidate_title}.pdf"
        candidate_path = os.path.join(self.folder_path, candidate_filename)

        incoming_hash = uploaded_pdf.source.compute_hash()
        counter = 0
        while os.path.exists(candidate_path):
            existing_hash = self._compute_pdf_hash(candidate_path)

            if existing_hash == incoming_hash:
                logger.info(f"Skipped duplicate PDF upload for title '{candidate_title}'")
                return {
                    "title": candidate_title,
                    "already_indexed": True,
                }

            counter += 1
            candidate_title = f"{base_title}_({counter})"
            candidate_filename = f"{candidate_title}.pdf"
            candidate_path = os.path.join(self.folder_path, candidate_filename)

        try:
            uploaded_pdf.source.write_to(candidate_path)
        except Exception as e:
            raise IOError(f"Failed to ingest PDF '{uploaded_pdf.title}'") from e

        return {
            "title": candidate_title,
            "already_indexed": False,
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
        
    def clear_cache(self) -> None:
        """Remove all PDF files from the cache folder."""
        if not os.path.isdir(self.folder_path):
            logger.warning(f"Cache folder does not exist for clearing: {self.folder_path}")
            return
        
        for root, _dirs, files in os.walk(self.folder_path):
            for filename in files:
                pdf_path = os.path.join(root, filename)
                try:
                    os.remove(pdf_path)
                    logger.info(f"Removed cached PDF file: {pdf_path}")
                except Exception as e:
                    logger.error(f"Error removing cached PDF file '{pdf_path}': {e}")

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

    def get_items(self) -> List[Dict]:
        """Get PDF items from the folder.
        
        Args:
            collection_name: Ignored for folder source (for API compatibility).
            
        Returns:
            List of dictionaries with 'key', 'path', and 'title' keys.
        """
        items = []
        
        # Walk through the folder and find all PDFs
        for root, _dirs, files in os.walk(self.folder_path):
            for filename in files:
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, filename)
                    
                    # Use filename (without extension) as title
                    title = self._sanitize_filename(os.path.splitext(filename)[0])
                    
                    # Use relative path as key (for uniqueness)
                    rel_path = os.path.relpath(pdf_path, self.folder_path)
                    key = rel_path.replace(os.sep, '_')
                    
                    items.append({
                        'key': key,
                        'path': pdf_path,
                        'title': title
                    })
        
        logger.info(f"Found {len(items)} PDF files in {self.folder_path}")
        return items
    
    def list_collections(self) -> List[Dict]:
        """Return empty list for API compatibility with ZoteroDatabase.
        
        Returns:
            Empty list (folders don't have collections).
        """
        return []
