"""Main orchestration class for Zotero RAG system."""

import os

# Suppress noisy progress bars that can trigger BrokenPipe in Streamlit
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# nltk >=3.9.2 refuses to import any dependency whose path lies under the CWD
# (CWE-427 hardening, nltk/inisec.py). uv puts .venv inside the project root, so
# running anything from the repo root makes *every* installed package look like a
# CWD module and nltk dies on `import regex`. The guard's own escape hatch; safe
# here because the repo root holds no top-level .py files to shadow anything with.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

import logging
import warnings

import nltk
from embedding_manager import EmbeddingManager
from highlighter import PDFHighlighter
from models import (
    Answer,
    CachedPDF,
    Chunk,
    IngestResult,
    PDFIngestItem,
    RerankedChunk,
    UpsertResult,
    ingest_items_from_folder,
)
from pdf_cache_manager import PDFCacheManager
from pdf_processor import PDFProcessor
from qa_engine import QAEngine
from qdrant_manager import QdrantManager
from question_presets import resolve
from reranker import Reranker
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

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ZoteroRAG:
    """Main orchestration class for the Zotero RAG pipeline."""
    
    def __init__(self, 
                dense_model_name: str = "BAAI/bge-base-en-v1.5", 
                qa_model: str = "deepset/deberta-v3-large-squad2",
                reranker_model: str = "BAAI/bge-reranker-base",
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
            dense_model_name: Name of the FastEmbed dense model for embeddings.
            qa_model: Name of the QA model for answer extraction.
            reranker_model: Name of the cross-encoder model for reranking.
            grobid_url: URL of the GROBID service.
            grobid_timeout: Timeout in seconds for GROBID requests.
            qdrant_url: URL of the Qdrant service.
            ollama_url: URL of the Ollama service.
            model_device: Device to use for models ('cpu', 'cuda'). Auto-detect if None.
            encode_batch_size: Batch size for encoding. If None, auto-detect (targets 75% memory).
            qa_batch_size: Batch size for QA extraction. If None, auto-detect (targets 75% memory).
            rerank_batch_size: Batch size for reranking. If None, auto-detect (targets 75% memory).
            use_chunk_contextualization: Whether to contextualize chunks with Ollama before embedding.
            output_base_dir: Base directory for storing outputs.
            qdrant_collection_suffix: Appended to the Qdrant collection names, to keep
                separate corpora out of each other's retrieval pool.
        """
        self.use_chunk_contextualization = use_chunk_contextualization
        self.output_base_dir = output_base_dir
        self.pdf_cache_dir = os.path.join(self.output_base_dir, "pdf_cache") or "pdf_cache"
        
        self.pdf_cache = PDFCacheManager(
            folder_path=self.pdf_cache_dir
        )
        
        self.pdf_processor = PDFProcessor(
            grobid_url=grobid_url,
            grobid_timeout=grobid_timeout,
            output_base_dir=self.output_base_dir
        )
        
        self.embedding_manager = EmbeddingManager(
            dense_model_name=dense_model_name,
            ollama_url=ollama_url,
            device=model_device,
            encode_batch_size=encode_batch_size,
            use_chunk_contextualization=use_chunk_contextualization
        )

        self.qdrant_manager = QdrantManager(
            dense_model_name=dense_model_name,
            qdrant_url=qdrant_url,
            vector_size=self.embedding_manager.vector_size,
            collection_suffix=qdrant_collection_suffix,
        )
        
        self.reranker = Reranker(
            model_name=reranker_model,
            device=model_device,
            batch_size=rerank_batch_size
        )
        
        self.qa_engine = QAEngine(
            model_name=qa_model,
            device=model_device,
            batch_size=qa_batch_size,
        )
        
        self.highlighter = PDFHighlighter()
        
        # Color management for multi-query highlighting
        self.query_colors = [
            (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0),
            (0.5, 1.0, 0.5), (1.0, 0.7, 0.8), (0.7, 0.5, 1.0),
        ]
        self.query_color_map = {}
        
        # For debugging/inspection
        self.last_candidates = []
        self.last_reranked = []
    
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
            self.qdrant_manager.open_connection()
            return self.qdrant_manager.list_indexed_pdfs()
        except ConnectionError as e:
            logger.error(f"Error during get_indexed_pdfs: {e!s}")
            raise
        except Exception as e:
            logger.error(f"Error during get_indexed_pdfs: {e!s}")
            raise
        finally:
            self.qdrant_manager.close_connection()

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

        all_chunks = []
        per_pdf_context = {}
        result = UpsertResult()
        newly_cached_hashes: set[str] = set()
        try:
            self.qdrant_manager.open_connection()
            
            # Stage 1: Process PDFs and extract chunks
            newly_cached_hashes = {
                item.pdf_hash
                for item in target_pdfs
                if item.newly_cached
            }

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
                        if progress_callback:
                            progress_callback(
                                'pdf',
                                idx + 1,
                                len(target_pdfs),
                                f"Skipped (title already in use): {title[:80]}",
                            )
                        continue

                    try:
                        if self.qdrant_manager.is_pdf_indexed(pdf_hash):
                            indexed_title = self.qdrant_manager.register_pdf_title(pdf_hash, title)
                            result.already_indexed.append({
                                "input_title": title,
                                "indexed_title": indexed_title or title,
                                "pdf_hash": pdf_hash,
                            })
                            if progress_callback:
                                message = f"Skipped (already indexed): {title[:80]}"
                                if indexed_title and indexed_title != title:
                                    message = (
                                        "Skipped (already indexed, using: "
                                        f"{indexed_title[:80]}): {title[:80]}"
                                    )
                                progress_callback(
                                    'pdf',
                                    idx + 1,
                                    len(target_pdfs),
                                    message,
                                )
                            continue
                    except Exception as e:  # noqa: BLE001 - index check failure: fall through to reindexing
                        logger.warning(
                            "Qdrant index check failed for '%s': %s", title, str(e),
                        )

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

                        sentence_count = len(extracted.sentences)
                        chunk = Chunk(
                            text=extracted.text,
                            page_number=extracted.page_number,
                            chunk_index=extracted.chunk_index,
                            title=title,
                            pdf_hash=pdf_hash,
                            section=extracted.section,
                            sentence_count=sentence_count,
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

            result.processed_pdfs = len(per_pdf_context)

            if not all_chunks:
                if (not result.failed_pdfs
                    and not result.already_indexed
                    and not result.duplicate_titles):
                    raise ValueError("No text could be extracted from the selected PDFs.")
                return result

            # Stage 2: Build index
            all_texts = [p.text for p in all_chunks]
            if self.use_chunk_contextualization:
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
                        if not chunk_indices:
                            continue

                        document_text = context_info.get("document_text", "")
                        if not document_text.strip():
                            logger.warning("Skipping contextualization for %s due to empty document text", pdf_title)
                            continue

                        base_chunks = [all_texts[i] for i in chunk_indices]
                        contextualized_chunks = self.embedding_manager.generate_contextual_chunks(
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
            else:
                logger.info("Chunk contextualization disabled: using original chunks for embedding.")

            contextual_texts = list(all_texts) #FIXME: debug
            hybrid_embeddings = self.embedding_manager.encode_chunks(progress_callback, all_texts)

            indexed_chunks, title_overrides = self.qdrant_manager.upsert_chunks(
                all_chunks,
                dense_embeddings=hybrid_embeddings["dense"],
                sparse_embeddings=hybrid_embeddings["sparse"],
                contextual_texts=contextual_texts, #FIXME: debug
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
            self.qdrant_manager.close_connection()

        return result
    
    def delete_pdf_by_title(self, pdf_title: str) -> bool:
        """Delete all chunks from a specific PDF in the index.
        
        Args:
            pdf_title: Title of the PDF to delete.
        
        Returns:
            Number of deleted PDFs.
        """
        try:
            self.qdrant_manager.open_connection()

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
        finally:
            self.qdrant_manager.close_connection()

    def clear_index(self) -> bool:
        """Clear the entire Qdrant collection, removing all indexed chunks.
        
        Returns:
            True if the collection was cleared successfully, False otherwise.
        """
        try:
            self.qdrant_manager.open_connection()
            deleted_pdfs = self.qdrant_manager.clear_chunk_collection()
            self.pdf_cache.clear_pdf_cache(deleted_pdfs)
            self.pdf_processor.clear_tei_cache(deleted_pdfs)
            logger.info("Successfully cleared the Qdrant collection.")
            return True
        except Exception as e:  # noqa: BLE001 - failure reported to the caller as False
            logger.error(f"Error during clear_chunk_collection: {e!s}")
            return False
        finally:
            self.qdrant_manager.close_connection()

    def answer_question(self,
                    question: str,
                    question_type: str = 'general',
                    overrides: dict | None = None,
                    progress_callback=None,
                    rerank_callback=None,
                    num_paraphrases: int = 2,
                    highlight_color: tuple[float, float, float] | None = None,
                    question_variations: list[str] | None = None) -> list[Answer]:
        """Answer a question using the full RAG pipeline.

        Pipeline stages:
        1. Qdrant Retrieval (Cosine Similarity)
        2. CrossEncoder Reranking (Threshold Filtering)
        3. QA Extraction (with Context Overlap/Sliding Window)

        Args:
            question: The question to answer.
            question_type: Type of question (factoid, explanation, methodology, etc.).
            overrides: User overrides applied on top of the question-type preset.
                The merged config drives every stage: retrieval_threshold,
                rerank_threshold and qa_score_threshold are applied literally.
            progress_callback: Function(current, total, message) for QA progress.
            rerank_callback: Function(current, total, message) for rerank progress.
            num_paraphrases: Number of question paraphrases to generate (0 = disabled).
            highlight_color: RGB tuple (0-1) for highlighting. If None, use query-based color.
            question_variations: Pre-generated question variations to use. If None, generate them.

        Returns:
            List of Answer objects, deduplicated and sorted by score.
        """
        config = resolve(question_type, overrides)

        # Stage 0: Expand question if enabled and variations not provided
        if question_variations is None:
            question_variations = [question]  # Always include original
            if self.qa_engine.enable_question_expansion and num_paraphrases > 0:
                question_variations = self.qa_engine.expand_question(question, num_variations=num_paraphrases)
                logger.info(f"Question expansion: {len(question_variations)} variations generated")
            elif num_paraphrases == 0:
                logger.info("Question paraphrasing disabled by user")
        else:
            logger.info(f"Using {len(question_variations)} pre-selected question variations")
        
        # Stage 1: Retrieve candidate chunks (Qdrant)
        # Search with all question variations and merge results
        try:
            all_candidates = []
            seen_chunks = set()
            self.qdrant_manager.open_connection()
            
            query_embeddings = []
            for q_var in question_variations:
                query_embeddings.append(self.embedding_manager.encode_query(q_var))

            batch_results = self.qdrant_manager.search_batch(
                query_embeddings,
                config['retrieval_threshold'],
                result_limit=config['result_limit'],
                mode=config['retrieval_mode'],
            )

            for i, q_var in enumerate(question_variations):
                var_candidates = batch_results[i] if i < len(batch_results) else []
                logger.debug(f"Variation {i}: '{q_var}' -> {len(var_candidates)} candidates")

                # Add unseen candidates
                for chunk, score in var_candidates:
                    chunk_id = (chunk.pdf_hash, chunk.chunk_index)  # Unique identifier
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        all_candidates.append((chunk, score))
            
            # Sort by retrieval score
            all_candidates.sort(key=lambda x: x[1])
            candidates = all_candidates
            
            logger.debug(f"Question: {question}")
            logger.debug(f"Retrieved {len(candidates)} unique chunks from {len(question_variations)} variations")
            
            if not candidates:
                self.last_candidates = []
                self.last_reranked = []
                return []
            
            # Store for debugging
            self.last_candidates = [
                {
                    'chunk': c[0],
                    'retrieval_score': c[1],
                    'kept': True
                }
                for c in candidates
            ]
        finally:
            self.qdrant_manager.close_connection()
        
        # Stage 2: Rerank and Filter (CrossEncoder)
        if config['rerank_enabled']:
            reranked = self.reranker.rerank(
                question,
                candidates,
                config['rerank_threshold'],
                progress_callback=rerank_callback,
                query_variations=question_variations
            )
        else:
            # ponytail: bypass keeps the retrieval order (best first) so the
            # ablation can attribute the reranker's contribution; rerank_score
            # mirrors retrieval_score because nothing rescored the candidates.
            reranked = [RerankedChunk(chunk=chunk, retrieval_score=score,
                                          rerank_score=score)
                        for chunk, score in sorted(candidates, key=lambda c: c[1], reverse=True)]
        
        # Update debug info with rerank results
        reranked_texts = {c.chunk.text for c in reranked}
        for c in self.last_candidates:
            c['kept'] = c['chunk'].text in reranked_texts
        self.last_reranked = reranked  # already sorted by rerank_score desc

        if not reranked:
            return []
        
        # Stage 3: Extract answers (QA Model)
        # Use provided color or get a query-based color
        if highlight_color is None:
            color = self.get_query_color(question)
        else:
            color = highlight_color
        
        answers = self.qa_engine.extract_answers(
            question,
            reranked,
            config,
            color=color,
            progress_callback=progress_callback,
            question_variations=question_variations
        )
        
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