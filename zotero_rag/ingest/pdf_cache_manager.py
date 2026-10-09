"""PDF folder source - alternative to Zotero database."""

import logging
import os

from zotero_rag.ingest.pdf_utils import is_pdf_hash
from zotero_rag.models import CachedPDF, PDFIngestItem

logger = logging.getLogger(__name__)

class PDFCacheManager:
    """Owns the local PDF cache, ensuring unique filenames and proper storage."""
    
    def __init__(self, folder_path: str):
        """Initialize PDF folder source.
        
        Args:
            folder_path: Path to folder containing PDF files.
        """
        if not folder_path:
            raise ValueError("Folder path cannot be empty")

        self.folder_path = os.path.abspath(folder_path)
        os.makedirs(self.folder_path, exist_ok=True)

        logger.info(f"Initialized PDFCacheManager with folder: {self.folder_path}")

    def _path(self, pdf_hash: str) -> str:
        """Where the PDF with this hash is cached, whether or not it is."""
        return os.path.join(self.folder_path, f"{pdf_hash}.pdf")

    def ingest_pdf(self, uploaded_pdf: PDFIngestItem) -> CachedPDF:
        """Ingest a PDF from an upload, saving it to the cache folder with a unique hash-based filename.

        Args:
            uploaded_pdf: PDFIngestItem containing the upload to ingest.

        Returns:
            CachedPDF object with details of the ingested PDF.
        """
        if uploaded_pdf is None:
            raise ValueError("uploaded_pdf cannot be None")

        pdf_hash = uploaded_pdf.source.compute_hash()
        candidate_path = self._path(pdf_hash)

        newly_cached = False
        if not os.path.exists(candidate_path):
            try:
                uploaded_pdf.source.write_to(candidate_path)
                newly_cached = True
            except Exception as e:
                raise OSError(f"Failed to ingest PDF '{uploaded_pdf.title}'") from e

        title = uploaded_pdf.title or "Unknown"
        return CachedPDF(
            pdf_hash=pdf_hash,
            title=title,
            cache_path=candidate_path,
            newly_cached=newly_cached,
            reference=uploaded_pdf.reference,
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

        candidate_path = self._path(pdf_hash)

        if not os.path.exists(candidate_path):
            logger.warning(f"PDF file not found for removal: {candidate_path}")
            return False
        try:
            os.remove(candidate_path)
        except OSError as e:
            logger.error(f"Error removing PDF file '{candidate_path}': {e}")
            return False
        logger.info(f"Removed PDF file: {candidate_path}")
        return True
        
    def clear_pdf_cache(self, deleted_pdfs: dict[str, str]):
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

        candidate_path = self._path(pdf_hash)
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
                    if is_pdf_hash(base_name):
                        items.add(base_name.lower())
                    else:
                        logger.warning("Ignoring non-hash cached PDF: %s", filename)

        logger.info(f"Found {len(items)} PDF files in {self.folder_path}")
        return list(items)