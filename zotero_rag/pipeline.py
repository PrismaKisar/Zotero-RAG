"""Main orchestration class for Zotero RAG system."""

import os

# Suppress noisy progress bars that can trigger BrokenPipe in Streamlit
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import logging
import time
import warnings

from contextualizer import ChunkContextualizer
from embedding_manager import DEFAULT_DENSE_MODEL, EmbeddingManager
from highlighter import PDFHighlighter
from models import (
    Answer,
    CachedPDF,
    Chunk,
    IngestResult,
    PDFIngestItem,
    QuestionTrace,
    Reader,
    RerankedChunk,
    RetrievedChunk,
    UpsertResult,
    ingest_items_from_folder,
)
from pdf_cache_manager import PDFCacheManager
from pdf_processor import PDFProcessor
from qa_engine import DEFAULT_QA_MODEL, QAEngine
from qdrant_manager import QdrantManager
from reranker import DEFAULT_RERANKER_MODEL, Reranker
from settings import resolve
from streamlit.runtime.uploaded_file_manager import UploadedFile
from zotero_db import ZoteroDatabase

warnings.filterwarnings('ignore', message='.*position_ids.*')

# Logs live in one directory instead of scattering across the working directory
LOG_DIR = 'logs'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'zotero_rag.log'), mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers only if they haven't been added yet
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class ZoteroRAG:
    """Main orchestration class for the Zotero RAG pipeline."""
    
    def __init__(self, 
                dense_model_name: str = DEFAULT_DENSE_MODEL,
                qa_model: str = DEFAULT_QA_MODEL,
                reranker_model: str = DEFAULT_RERANKER_MODEL,
                grobid_url: str = "http://localhost:8070", 
                grobid_timeout: int = 180,
                qdrant_url: str = "http://localhost:6333",
                ollama_url: str = "http://localhost:11434",
                model_device: str | None = None, 
                encode_batch_size: int | None = None,
                qa_batch_size: int | None = None,
                rerank_batch_size: int | None = None,
                use_chunk_contextualization: bool = True,
                output_base_dir: str = "output",
                qdrant_collection_suffix: str = ""):
        """Initialize the RAG system.
        
        Args:
            dense_model_name: Name of the sentence-transformers dense model for embeddings.
            qa_model: Name of the QA model for answer extraction.
            reranker_model: Name of the cross-encoder model for reranking.
            grobid_url: URL of the GROBID service.
            grobid_timeout: Timeout in seconds for GROBID requests.
            qdrant_url: URL of the Qdrant service.
            ollama_url: URL of the Ollama service.
            model_device: Device to use for models ('cpu', 'cuda', 'mps'). Auto-detect if None.
            encode_batch_size: Batch size for encoding. Module default if None.
            qa_batch_size: Batch size for QA extraction. Module default if None.
            rerank_batch_size: Batch size for reranking. Module default if None.
            use_chunk_contextualization: Whether to contextualize chunks with Ollama before embedding.
            output_base_dir: Base directory for storing outputs.
            qdrant_collection_suffix: Appended to the Qdrant collection names, to keep
                separate corpora out of each other's retrieval pool.
        """
        self.output_base_dir = output_base_dir
        self.pdf_cache_dir = os.path.join(self.output_base_dir, "pdf_cache") or "pdf_cache"

        pdf_cache = PDFCacheManager(
            folder_path=self.pdf_cache_dir
        )

        pdf_processor = PDFProcessor(
            grobid_url=grobid_url,
            grobid_timeout=grobid_timeout,
            output_base_dir=self.output_base_dir
        )

        embedding_manager = EmbeddingManager(
            dense_model_name=dense_model_name,
            device=model_device,
            encode_batch_size=encode_batch_size,
        )
        contextualizer = (ChunkContextualizer(ollama_url=ollama_url)
                          if use_chunk_contextualization else None)

        self._assemble(
            pdf_cache=pdf_cache,
            pdf_processor=pdf_processor,
            embedding_manager=embedding_manager,
            contextualizer=contextualizer,
            qdrant_manager=QdrantManager(
                dense_model_name=dense_model_name,
                qdrant_url=qdrant_url,
                vector_size=embedding_manager.vector_size,
                collection_suffix=qdrant_collection_suffix,
            ),
            reranker=Reranker(
                model_name=reranker_model,
                device=model_device,
                batch_size=rerank_batch_size
            ),
            reader=QAEngine(
                model_name=qa_model,
                device=model_device,
                batch_size=qa_batch_size,
            ),
            highlighter=PDFHighlighter(),
        )

    @classmethod
    def from_components(cls, *, qdrant_manager=None, embedding_manager=None,
                        reranker=None, reader: Reader | None = None, pdf_cache=None,
                        pdf_processor=None, contextualizer=None,
                        highlighter=None) -> "ZoteroRAG":
        """A pipeline over components built elsewhere, loading no model itself.

        The seam for the tests, which pass stand-ins for only the components the
        call under test touches. A component left out is None, and whatever
        reaches for it fails.
        """
        rag = cls.__new__(cls)
        rag._assemble(pdf_cache=pdf_cache, pdf_processor=pdf_processor,
                      embedding_manager=embedding_manager, contextualizer=contextualizer,
                      qdrant_manager=qdrant_manager, reranker=reranker, reader=reader,
                      highlighter=highlighter)
        return rag

    def _assemble(self, *, pdf_cache, pdf_processor, embedding_manager, contextualizer,
                  qdrant_manager, reranker, reader, highlighter) -> None:
        """Wire the components and the per-instance state every constructor needs.

        ``reader`` is public: the benchmarks point it at the reader a config asks
        for (see ``models.Reader``). ``contextualizer`` is None when chunks are
        embedded as extracted.
        """
        self.pdf_cache = pdf_cache
        self.pdf_processor = pdf_processor
        self.embedding_manager = embedding_manager
        self.contextualizer = contextualizer
        self.qdrant_manager = qdrant_manager
        self.reranker = reranker
        self.reader = reader
        self.highlighter = highlighter

        # Color management for multi-query highlighting
        self.query_colors = [
            (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0),
            (0.5, 1.0, 0.5), (1.0, 0.7, 0.8), (0.7, 0.5, 1.0),
        ]
        self.query_color_map = {}
        
        # What the last answer_question call did at each stage, for the
        # benchmarks: see QuestionTrace.
        self.last_trace = QuestionTrace()
    
    def get_query_color(self, query: str) -> tuple[float, float, float]:
        """Get a consistent color for a query string.
        
        Args:
            query: Query string.
            
        Returns:
            RGB color tuple.
        """
        if query not in self.query_color_map:
            color_idx = len(self.query_color_map) % len(self.query_colors)
            self.query_color_map[query] = self.query_colors[color_idx]
        return self.query_color_map[query]
    
    def get_indexed_pdfs(self) -> list[dict[str, str]]:
        """Get list of indexed PDFs.
        
        Returns:
            List of dictionaries with title and pdf_hash.
        """
        try:
            return self.qdrant_manager.list_indexed_pdfs()
        except Exception as e:
            logger.error(f"Error during get_indexed_pdfs: {e!s}")
            raise

    def consistency_check(self, indexed_pdfs: list[dict[str, str]]) -> bool:
        """Check consistency between indexed PDFs and cached PDFs.
        
        Args:
            indexed_pdfs: List of dicts or hashes for PDFs indexed in Qdrant.
        
        Returns:
            True if all indexed PDFs have corresponding cache entries, False otherwise.
        """
        if not indexed_pdfs:
            return True

        cached_hashes = set(self.pdf_cache.get_cached_items())
        indexed_hashes = set()
        for item in indexed_pdfs:
            if isinstance(item, dict):
                pdf_hash = item.get("pdf_hash")
                if pdf_hash:
                    indexed_hashes.add(pdf_hash)
            elif isinstance(item, str):
                indexed_hashes.add(item)

        missing_in_cache = indexed_hashes - cached_hashes

        if missing_in_cache:
            logger.warning(f"Consistency check failed: {len(missing_in_cache)} indexed PDFs missing in cache: {missing_in_cache}")
            return False
        
        return True

    def _ingest_pdfs(self, uploaded_pdfs: list[PDFIngestItem]) -> IngestResult:
        """Ingest PDFs into cache and prepare for indexing.

        Args:
            uploaded_pdfs: List of PDFIngestItem objects for PDFs to ingest.

        Returns:
            IngestResult with summary and error details.
        """
        if not uploaded_pdfs:
            raise ValueError("uploaded_pdfs cannot be empty")

        result = IngestResult()
        by_hash: dict[str, CachedPDF] = {}

        for uploaded_pdf in uploaded_pdfs:
            try:
                cached_pdf = self.pdf_cache.ingest_pdf(uploaded_pdf)
                if cached_pdf.pdf_hash in by_hash:
                    continue
                by_hash[cached_pdf.pdf_hash] = cached_pdf
            except Exception as e:  # noqa: BLE001 - per-PDF: record the failure and continue
                logger.error(f"Error ingesting PDF '{uploaded_pdf.title}': {e!s}")
                result.failed_pdfs.append(
                    {"title": str(uploaded_pdf.title), "error": str(e)}
                )

        result.ingested_pdfs = list(by_hash.values())
        return result

    def ingest_pdfs_from_upload(self, uploaded_pdfs: list[UploadedFile]) -> IngestResult:
        """Ingest PDFs from user uploads.

        Args:
            uploaded_pdfs: List of UploadedFile objects from Streamlit file uploader.
        
        Returns:
            IngestResult with summary and error details.
        """
        if not uploaded_pdfs:
            raise ValueError("uploaded_pdfs cannot be empty")
        
        return self._ingest_pdfs(PDFCacheManager.get_items_from_upload(uploaded_pdfs))
        
    def ingest_pdfs_from_zotero(self, zotero_collection: str | None = None,
                                zotero_data_dir: str | None = None) -> IngestResult:
        """Ingest PDFs from Zotero.

        Args:
            zotero_collection: Name of the Zotero collection to ingest from.
                If None, ingest all PDFs from the library.
            zotero_data_dir: Path to the Zotero data directory. If None, auto-detect.

        Returns:
            IngestResult with summary and error details.
        """ 
        source = ZoteroDatabase(zotero_data_dir)
        uploaded_pdfs = source.get_items(zotero_collection)
        
        return self._ingest_pdfs(uploaded_pdfs)

    def ingest_pdfs_from_folder(self, folder_path: str) -> IngestResult:
        """Ingest every PDF in a folder, each titled by its filename stem.

        Args:
            folder_path: Path to the folder to read PDFs from (not recursive).

        Returns:
            IngestResult with summary and error details.
        """
        items = ingest_items_from_folder(folder_path)
        if not items:
            logger.warning("No PDF found in %s", folder_path)

        return self._ingest_pdfs(items)

    def _rollback_pdfs(self, newly_cached_hashes: set[str], failed_hashes: set[str]) -> bool:
        """Rollback cached PDFs that were newly cached during the current upsert if they failed to index.

        Args:
            newly_cached_hashes: Set of PDF hashes that were newly cached during the current upsert.
            failed_hashes: Set of PDF hashes that failed to index during the current upsert.

        Returns:
            True if any rollbacks were performed, False otherwise.
        """
        to_remove = newly_cached_hashes & failed_hashes
        if not to_remove:
            return False

        for pdf_hash in to_remove:
            try:
                self.pdf_cache.remove_pdf(pdf_hash)
                self.pdf_processor.remove_tei_cache(pdf_hash)
                logger.debug("Rolled back PDF from cache: %s", pdf_hash)
            except Exception as e:  # noqa: BLE001 - rollback is best-effort
                logger.error("Error rolling back PDF '%s': %s", pdf_hash, str(e))

        return True

    def _skip_message(self, item: CachedPDF, result: UpsertResult,
                    newly_cached_hashes: set[str]) -> str | None:
        """Why ``item`` is not indexed, as a progress message, or None to index it.

        Records the skip in ``result``. A duplicate title that was newly cached
        is removed from the cache here, and from ``newly_cached_hashes`` so the
        rollback does not try to remove it a second time.
        """
        title, pdf_hash = item.title, item.pdf_hash
        existing_title_hash = self.qdrant_manager.find_pdf_hash_by_title(
            title,
            filter_by_model=False,
        )
        if existing_title_hash and existing_title_hash != pdf_hash:
            logger.warning("Title already in use, skipping indexing: %s", title)
            result.duplicate_titles.append(title)
            if item.newly_cached:
                removed = self.pdf_cache.remove_pdf(pdf_hash)
                if removed:
                    newly_cached_hashes.discard(pdf_hash)
                    logger.info("Removed cached PDF for duplicate title: %s", title)
                else:
                    logger.warning("Failed to remove cached PDF for duplicate title: %s", title)
            return f"Skipped (title already in use): {title[:80]}"

        try:
            if self.qdrant_manager.is_pdf_indexed(pdf_hash):
                indexed_title = self.qdrant_manager.register_pdf_title(pdf_hash, title)
                result.already_indexed.append({
                    "input_title": title,
                    "indexed_title": indexed_title or title,
                    "pdf_hash": pdf_hash,
                })
                if indexed_title and indexed_title != title:
                    return (
                        "Skipped (already indexed, using: "
                        f"{indexed_title[:80]}): {title[:80]}"
                    )
                return f"Skipped (already indexed): {title[:80]}"
        except Exception as e:  # noqa: BLE001 - index check failure: fall through to reindexing
            logger.warning(
                "Qdrant index check failed for '%s': %s", title, str(e),
            )
        return None

    def _extract_chunks(self, target_pdfs: list[CachedPDF], result: UpsertResult,
                        newly_cached_hashes: set[str],
                        progress_callback=None) -> tuple[list[Chunk], dict]:
        """Stage 1: the chunks of every PDF that needs indexing.

        A PDF that fails is recorded in ``result`` and the others go on.

        Returns:
            ``(all_chunks, per_pdf_context)``: the latter maps each processed
            pdf_hash to its title, full document text, and the positions of its
            chunks in ``all_chunks``.
        """
        all_chunks = []
        per_pdf_context = {}
        for idx, item in enumerate(target_pdfs):
            title = item.title
            pdf_hash = item.pdf_hash
            if progress_callback:
                progress_callback(
                    'pdf',
                    idx,
                    len(target_pdfs),
                    f"Analysing PDF {idx + 1}/{len(target_pdfs)}: {title[:80]}",
                )

            try:
                skip = self._skip_message(item, result, newly_cached_hashes)
                if skip:
                    if progress_callback:
                        progress_callback('pdf', idx + 1, len(target_pdfs), skip)
                    continue

                canonical_title = self.qdrant_manager.register_pdf_title(pdf_hash, title)
                if canonical_title:
                    title = canonical_title

                extracted_chunks, document_text = self.pdf_processor.extract_text_chunks(
                    pdf_hash,
                    pdf_title=title,
                )

                if pdf_hash not in per_pdf_context:
                    per_pdf_context[pdf_hash] = {
                        "title": title,
                        "document_text": document_text,
                        "chunk_indices": [],
                    }

                for extracted in extracted_chunks:
                    # Filter by section type if needed
                    if not self.pdf_processor.CONTENT_SECTIONS.get(extracted.section, True):
                        continue

                    chunk = Chunk(
                        text=extracted.text,
                        page_number=extracted.page_number,
                        chunk_index=extracted.chunk_index,
                        title=title,
                        pdf_hash=pdf_hash,
                        section=extracted.section,
                        sentence_count=len(extracted.sentences),
                        sentences=extracted.sentences
                    )
                    all_chunks.append(chunk)
                    per_pdf_context[pdf_hash]["chunk_indices"].append(len(all_chunks) - 1)
            except Exception as e:  # noqa: BLE001 - per-PDF: record the failure and continue
                logger.error("Failed to process PDF '%s': %s", title, str(e))
                result.failed_pdfs.append({"title": title, "error": str(e), "pdf_hash": pdf_hash})

            if progress_callback:
                progress_callback(
                    'pdf',
                    idx + 1,
                    len(target_pdfs),
                    f"Analysed PDF {idx + 1}/{len(target_pdfs)}: {title[:80]}",
                )
        return all_chunks, per_pdf_context

    def _contextualize(self, all_texts: list[str], per_pdf_context: dict,
                       progress_callback=None) -> None:
        """Stage 2: replace each PDF's texts in ``all_texts`` with contextualized ones.

        Best-effort, per PDF: a PDF with no document text or a mismatched
        result keeps its original texts, and an unexpected error stops here
        leaving the PDFs already done contextualized.
        """
        try:
            pdf_hashes_with_chunks = [
                pdf_hash
                for pdf_hash, context_info in per_pdf_context.items()
                if context_info.get("chunk_indices")
            ]

            for pdf_idx, pdf_hash in enumerate(pdf_hashes_with_chunks):
                context_info = per_pdf_context[pdf_hash]
                pdf_title = context_info.get("title", pdf_hash)
                if progress_callback:
                    progress_callback(
                        "contextualization",
                        pdf_idx,
                        len(pdf_hashes_with_chunks),
                        f"Contextualizing PDF {pdf_idx + 1}/{len(pdf_hashes_with_chunks)}: {pdf_title[:80]}",
                    )
                chunk_indices = context_info.get("chunk_indices", [])

                document_text = context_info.get("document_text", "")
                if not document_text.strip():
                    logger.warning("Skipping contextualization for %s due to empty document text", pdf_title)
                    continue

                base_chunks = [all_texts[i] for i in chunk_indices]
                contextualized_chunks = self.contextualizer.contextualize(
                    document_text=document_text,
                    all_texts=base_chunks,
                )

                if len(contextualized_chunks) != len(chunk_indices):
                    logger.warning(
                        "Contextualization size mismatch for %s (%s vs %s), using original chunks",
                        pdf_title,
                        len(contextualized_chunks),
                        len(chunk_indices),
                    )
                    continue

                for offset, chunk_abs_idx in enumerate(chunk_indices):
                    all_texts[chunk_abs_idx] = contextualized_chunks[offset]

                if progress_callback:
                    progress_callback(
                        "contextualization",
                        pdf_idx + 1,
                        len(pdf_hashes_with_chunks),
                        f"Contextualized PDF {pdf_idx + 1}/{len(pdf_hashes_with_chunks)}: {pdf_title[:80]}",
                    )
        except Exception as e:  # noqa: BLE001 - contextualization optional: keep the original chunks
            logger.warning("Contextual chunk generation failed, falling back to original chunks: %s", str(e))

    def upsert_pdfs(self, target_pdfs: list[CachedPDF], progress_callback=None) -> UpsertResult:
        """Process PDFs, extract chunks, and upsert into Qdrant index.
        
        Args:
            target_pdfs: List of cached PDFs to process. Must not be empty.
            progress_callback: Optional function(current_stage, current_idx, total, message) for progress updates.
        Returns:
            UpsertResult with indexing summary and warning/error details.
        """
        if target_pdfs is None or target_pdfs == []:
            raise ValueError("No new PDFs provided for upsert. Skipping indexing stage.")

        result = UpsertResult()
        newly_cached_hashes: set[str] = set()
        try:
            self.qdrant_manager.ping()

            newly_cached_hashes = {
                item.pdf_hash
                for item in target_pdfs
                if item.newly_cached
            }
            all_chunks, per_pdf_context = self._extract_chunks(
                target_pdfs, result, newly_cached_hashes, progress_callback)
            result.processed_pdfs = len(per_pdf_context)

            if not all_chunks:
                if (not result.failed_pdfs
                    and not result.already_indexed
                    and not result.duplicate_titles):
                    raise ValueError("No text could be extracted from the selected PDFs.")
                return result

            all_texts = [p.text for p in all_chunks]
            if self.contextualizer is not None:
                self._contextualize(all_texts, per_pdf_context, progress_callback)
            else:
                logger.info("Chunk contextualization disabled: using original chunks for embedding.")

            # Stage 3: embed and index
            hybrid_embeddings = self.embedding_manager.encode_chunks(progress_callback, all_texts)

            indexed_chunks, title_overrides = self.qdrant_manager.upsert_chunks(
                all_chunks,
                dense_embeddings=hybrid_embeddings["dense"],
                sparse_embeddings=hybrid_embeddings["sparse"],
                contextual_texts=all_texts, #FIXME: debug
                progress_callback=progress_callback,
            )
            result.indexed_chunks = indexed_chunks
            result.title_overrides = title_overrides
        except Exception as e:
            logger.error(f"Error during upsert_chunks: {e!s}")
            raise
        finally:
            failed_hashes = {
                item.get("pdf_hash")
                for item in result.failed_pdfs
                if item.get("pdf_hash")
            }
            _ = self._rollback_pdfs(newly_cached_hashes, failed_hashes)

        return result
    
    def delete_pdf_by_title(self, pdf_title: str) -> bool:
        """Delete all chunks from a specific PDF in the index.
        
        Args:
            pdf_title: Title of the PDF to delete.
        
        Returns:
            Number of deleted PDFs.
        """
        try:
            pdf_hash = self.qdrant_manager.find_pdf_hash_by_title(
                pdf_title.strip(),
                filter_by_model=True,
            )
            if not pdf_hash:
                logger.warning("No chunks found to delete for PDF: %s", pdf_title)
                return False

            delete_result = self.qdrant_manager.delete_pdf_from_index(pdf_hash)
            deleted = delete_result.get("deleted", False)
            if deleted:
                if not delete_result.get("had_other_models", False):
                    self.pdf_processor.remove_tei_cache(pdf_hash)
                    self.pdf_cache.remove_pdf(pdf_hash)
                logger.info("Successfully deleted chunks for PDF: %s", pdf_title)
                return True

            logger.warning("No chunks found to delete for PDF: %s", pdf_title)
            return False
        except Exception as e:  # noqa: BLE001 - failure reported to the caller as False
            logger.error(f"Error during delete_pdf_from_index: {e!s}")
            return False

    def clear_index(self) -> bool:
        """Clear the entire Qdrant collection, removing all indexed chunks.
        
        Returns:
            True if the collection was cleared successfully, False otherwise.
        """
        try:
            deleted_pdfs = self.qdrant_manager.clear_chunk_collection()
            self.pdf_cache.clear_pdf_cache(deleted_pdfs)
            self.pdf_processor.clear_tei_cache(deleted_pdfs)
            logger.info("Successfully cleared the Qdrant collection.")
            return True
        except Exception as e:  # noqa: BLE001 - failure reported to the caller as False
            logger.error(f"Error during clear_chunk_collection: {e!s}")
            return False

    def _neighbour_candidates(self, retrieved: list, seen: set, span: int) -> list:
        """Chunks adjacent to the retrieved ones, for evidence spread over paragraphs.

        Retrieval scores whole paragraphs independently, so evidence that runs
        across two consecutive paragraphs is only found when both happen to
        score well on their own. Scattered-evidence questions are retrieved far
        worse than single-chunk ones, and nothing else in the pipeline addresses
        that. Pulling in each hit's neighbours does, for the case where the
        continuation is the next paragraph rather than a distant one.

        A neighbour inherits the score of the hit that pulled it in: it was
        never scored against the query, and inventing a score for it would put
        a fabricated number into the ordering the retrieval metrics read. The
        reranker rescores everything anyway, so the inherited value only feeds
        the rerank input. It does NOT decide rank: the caller ranks every
        neighbour below every hit, because a chunk carrying a borrowed 0.9
        would otherwise outrank a hit that genuinely scored 0.4 and push it out
        of the top ten the retrieval metrics read.

        Args:
            retrieved: (chunk, score) pairs already found, in any order.
            seen: chunk ids already present; mutated so neighbours stay unique.
            span: how many paragraphs either side; 0 disables the whole thing.

        Returns:
            The (chunk, score) pairs to append, empty when ``span`` is 0.
        """
        if span <= 0 or not retrieved:
            return []

        wanted = {}
        for chunk, score in retrieved:
            for offset in range(-span, span + 1):
                index = chunk.chunk_index + offset
                # ponytail: no upper bound per document is known here, so a
                # neighbour past the last paragraph simply misses in Qdrant.
                if offset == 0 or index < 0:
                    continue
                neighbour_id = (chunk.pdf_hash, index)
                if neighbour_id not in seen and neighbour_id not in wanted:
                    wanted[neighbour_id] = score

        if not wanted:
            return []

        ids = list(wanted)
        chunks = self.qdrant_manager.fetch_chunks(ids)
        found = []
        for chunk in chunks:
            chunk_id = (chunk.pdf_hash, chunk.chunk_index)
            seen.add(chunk_id)
            found.append((chunk, wanted[chunk_id]))

        logger.debug(f"Neighbour expansion (span {span}): "
                     f"{len(found)} added to {len(retrieved)} retrieved")
        return found

    def answer_question(self,
                    question: str,
                    overrides: dict | None = None,
                    progress_callback=None,
                    rerank_callback=None,
                    highlight_color: tuple[float, float, float] | None = None,
                    pdf_hashes: list[str] | None = None) -> list[Answer]:
        """Answer a question using the full RAG pipeline.

        Pipeline stages:
        1. Qdrant Retrieval (Cosine Similarity)
        2. CrossEncoder Reranking (Threshold Filtering)
        3. QA Extraction (with Context Overlap/Sliding Window)

        Args:
            question: The question to answer.
            overrides: Changes to the fixed settings (see ``settings``), for the
                benchmarks that vary them one at a time. The product passes none.
            progress_callback: Function(current, total, message) for QA progress.
            rerank_callback: Function(current, total, message) for rerank progress.
            highlight_color: RGB tuple (0-1) for highlighting. If None, use query-based color.
            pdf_hashes: Restrict retrieval to these documents. None searches the
                whole library, which is the product's default; scoping it is what
                the benchmark needs to separate picking the right paper from
                picking the right chunk inside it.

        Returns:
            List of Answer objects, deduplicated and sorted by score.
        """
        config = resolve(overrides)

        # A fresh trace per call: a stage absent from it is one this call never
        # reached, which is the honest record when retrieval comes back empty.
        trace = QuestionTrace()
        self.last_trace = trace
        stage_started = time.perf_counter()

        def mark(stage: str) -> None:
            nonlocal stage_started
            now = time.perf_counter()
            trace.stage_times[stage] = now - stage_started
            stage_started = now

        # Stage 1: Retrieve candidate chunks (Qdrant). One query: expanding it
        # with paraphrases was measured, never added recall, and was removed.
        try:
            [hits] = self.qdrant_manager.search_batch(
                [self.embedding_manager.encode_query(question)],
                result_limit=config['result_limit'],
                mode=config['retrieval_mode'],
                pdf_hashes=pdf_hashes,
            )
            hits = list(hits)
            seen_chunks = {(chunk.pdf_hash, chunk.chunk_index) for chunk, _ in hits}

            # Best first, ties broken by chunk id. RRF gives equal scores to
            # chunks at the same rank in different branches and Qdrant returns
            # equals in no stable order: one config ranked 117 of 120 questions
            # differently across campaigns, mostly by adjacent swaps. Every
            # consumer re-sorts by its own score with a stable sort, so this is
            # the tie order they all inherit.
            hits.sort(key=lambda c: (-c[1], c[0].pdf_hash, c[0].chunk_index))
            neighbours = self._neighbour_candidates(
                hits, seen_chunks, config.get('retrieval_neighbours', 0))
            candidates = hits + neighbours

            logger.debug(f"Question: {question}")
            logger.debug(f"Retrieved {len(candidates)} chunks")

            # The retrieval order: every neighbour after every hit, best first
            # inside each group. A neighbour inherits its parent's score, so on
            # the score alone a chunk nothing scored could outrank a hit that
            # scored lower and take a position retrieval never assigned it.
            trace.retrieved = (
                [RetrievedChunk(chunk, score) for chunk, score in hits]
                + [RetrievedChunk(chunk, score, is_neighbour=True)
                   for chunk, score in sorted(neighbours, key=lambda c: -c[1])])

            if not candidates:
                return []
        finally:
            mark("retrieval")

        # Stage 2: Rerank and Filter (CrossEncoder)
        if config['rerank_enabled']:
            reranked = self.reranker.rerank(
                question,
                candidates,
                config['rerank_gate'],
                config['rerank_margin'],
                progress_callback=rerank_callback,
                # Read with a default rather than added to every preset: this is
                # a phase-two intervention, and a preset field would ship it as a
                # default before it has been accepted as one.
                order_by_retrieval=config.get('rerank_order_by_retrieval', False)
            )
        else:
            # ponytail: bypass keeps the retrieval order so the ablation can
            # attribute the reranker's contribution; rerank_score mirrors
            # retrieval_score because nothing rescored the candidates.
            reranked = [RerankedChunk(chunk=r.chunk, retrieval_score=r.retrieval_score,
                                      rerank_score=r.retrieval_score)
                        for r in trace.retrieved]

        trace.reranked = reranked  # already sorted by rerank_score desc
        mark("rerank")

        if not reranked:
            return []
        
        # Stage 3: Extract answers (QA Model)
        # Use provided color or get a query-based color
        if highlight_color is None:
            color = self.get_query_color(question)
        else:
            color = highlight_color
        
        answers = self.reader.extract_answers(
            question,
            reranked,
            config,
            color=color,
            progress_callback=progress_callback,
        )
        mark("read")

        return self._attach_pdf_paths(answers)
    
    def _attach_pdf_paths(self, answers: list[Answer]) -> list[Answer]:
        """Resolve cached PDF paths for extracted answers.
        
        Args:
            answers: List of Answer objects with title but no pdf_path.
        
        Returns:
            List of Answer objects with pdf_path set where available."""
        
        for ans in answers:
            pdf_path = self.pdf_cache.get_pdf_path(ans.pdf_hash)
            if pdf_path:
                ans.pdf_path = pdf_path
            else:
                ans.pdf_path = None

        return answers
    
    def highlight_pdf(self, answers_for_pdf: list[Answer], output_path: str) -> str:
        """Highlight PDF using TEI sentence coordinates.
        
        Args:
            answers_for_pdf: List of Answer objects from the same PDF.
            output_path: Path where the highlighted PDF should be saved.
            
        Returns:
            Path to the highlighted PDF, or None if highlighting failed.
        """
        return self.highlighter.highlight_pdf(answers_for_pdf, output_path)