"""PDF folder source - alternative to Zotero database."""

import logging
import os
import re

from models import CachedPDF, PDFIngestItem, UploadSource
from streamlit.runtime.uploaded_file_manager import UploadedFile

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
    
    @staticmethod
    def get_items_from_upload(uploaded_files: list[UploadedFile]) -> list[PDFIngestItem]:
        """Convert list of UploadedFile to list of PDFIngestItem for ingestion.
        
        Args:
            uploaded_files: List of UploadedFile objects from Streamlit file uploader.
            
        Returns:
            List of PDFIngestItem objects ready for ingestion.
        """
        items: list[PDFIngestItem] = []
        for uploaded_file in uploaded_files:
            if uploaded_file is None:
                logger.warning("Skipping None uploaded file")
                continue

            title = os.path.splitext(uploaded_file.name)[0]
            items.append(PDFIngestItem(title=title, source=UploadSource(uploaded_file)))

        return items

    def ingest_pdf(self, uploaded_pdf: PDFIngestItem) -> CachedPDF:
        """Ingest a PDF from an UploadedFile, saving it to the cache folder with a unique hash-based filename.

        Args:
            uploaded_pdf: PDFIngestItem containing the UploadedFile to ingest.

        Returns:
            CachedPDF object with details of the ingested PDF.
        """
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
                raise OSError(f"Failed to ingest PDF '{uploaded_pdf.title}'") from e

        title = uploaded_pdf.title or "Unknown"
        return CachedPDF(
            pdf_hash=pdf_hash,
            title=title,
            cache_path=candidate_path,
            created=created,
        )

    def remove_pdf(self, pdf_hash: str) -> bool:
        """Remove a cached PDF file by its hash.

        Args:
            pdf_hash: Hash of the PDF to remove.


        Returns:
            True if the file was successfully removed, False otherwise.
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
            except OSError as e:
                logger.error(f"Error removing PDF file '{candidate_path}': {e}")
                return False
        else:
            logger.warning(f"PDF file not found for removal: {candidate_path}")
            return False
        
    def clear_index_cache(self, deleted_pdfs: dict[str, str]):
        """Clear cached PDF files for deleted entries.

        Args:
            deleted_pdfs: Dictionary mapping PDF hashes to titles for entries that were deleted.
        """
        if not os.path.isdir(self.folder_path):
            logger.warning(f"Cache folder does not exist for clearing: {self.folder_path}")
            return
        
        for pdf_hash, title in deleted_pdfs.items():
            if not pdf_hash:
                logger.warning("Skipping deletion for entry with empty hash")
                continue

            ok = self.remove_pdf(pdf_hash)
            if not ok:
                raise OSError(f"Failed to clear cached PDF for deleted entry: {title} (hash: {pdf_hash})")

    def get_pdf_path(self, pdf_hash: str) -> str | None:
        """Get the file path of a cached PDF by its hash.

        Args:
            pdf_hash: Hash of the PDF to retrieve.

        Returns:
            File path of the cached PDF if it exists, None otherwise.
        """
        if not pdf_hash:
            logger.warning("PDF hash cannot be empty for getting PDF path")
            return None

        candidate_path = os.path.join(self.folder_path, f"{pdf_hash}.pdf")
        if os.path.isfile(candidate_path):
            return candidate_path

        logger.warning(f"PDF file not found for hash '{pdf_hash}': {candidate_path}")
        return None

    def get_cached_items(self) -> list[str]:
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