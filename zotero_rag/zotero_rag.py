"""Main orchestration class for Zotero RAG system."""

import os
# Suppress noisy progress bars that can trigger BrokenPipe in Streamlit
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import logging
from typing import List, Dict, Tuple, Optional
import warnings
import nltk
from streamlit.runtime.uploaded_file_manager import UploadedFile

from models import Paragraph, Answer, PDFIngestItem, IngestResult, UpsertResult, CachedPDF
from zotero_db import ZoteroDatabase
from pdf_cache_handler import PDFCacheHandler
from pdf_processor import PDFProcessor
from embedding_manager import EmbeddingManager
from qdrant_manager import QdrantManager
from reranker import Reranker
from qa_engine import QAEngine
from highlighter import PDFHighlighter

warnings.filterwarnings('ignore', message='.*position_ids.*')

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler('zotero_rag.log', mode='a')
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
                 qa_model: str = "deepset/roberta-base-squad2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 grobid_url: str = "http://localhost:8070", 
                 grobid_timeout: int = 180,
                 qdrant_url: str = "http://localhost:6333",
                 ollama_url: str = "http://localhost:11434",
                 model_device: str = None, 
                 encode_batch_size: int = None,
                 qa_batch_size: int = None,
                 rerank_batch_size: int = None,
                 use_chunk_contextualization: bool = True,
                 output_base_dir: str = "output"):
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
        """
        self.use_chunk_contextualization = use_chunk_contextualization
        self.output_base_dir = output_base_dir
        self.pdf_cache_dir = os.path.join(self.output_base_dir, "pdf_cache") or "pdf_cache"
        
        self.pdf_cache = PDFCacheHandler(
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
        )

        self.qdrant_manager = QdrantManager(
            dense_model_name=dense_model_name,
            qdrant_url=qdrant_url,
            vector_size=self.embedding_manager.vector_size,
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
    
    @staticmethod
    def list_collections(zotero_data_dir: str = None) -> List[Dict]:
        """Load collections from the Zotero database.
        
        Args:
            zotero_data_dir: Path to Zotero data directory. Auto-detect if None.
            
        Returns:
            List of dictionaries with 'id', 'name', and 'parent_id' keys.
        """
        db = ZoteroDatabase(zotero_data_dir)
        return db.list_collections()
    
    @property
    def paragraphs(self):
        """Access paragraphs from the Qdrant Manager (backward compatibility)."""
        return self.qdrant_manager.paragraphs
    
    @property
    def client(self):
        """Access the Qdrant client directly if needed (backward compatibility)."""
        return self.qdrant_manager.client
    
    def get_query_color(self, query: str) -> Tuple[float, float, float]:
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
    
    def get_indexed_pdfs(self) -> List[Dict[str, str]]:
        """Get list of indexed PDFs.
        
        Returns:
            List of dictionaries with title and pdf_hash.
        """
        try:
            self.qdrant_manager.open_connection()
            return self.qdrant_manager.list_indexed_pdfs()
        except Exception as e:
            logger.error(f"Error during get_indexed_pdfs: {str(e)}")
            return []
        finally:
            self.qdrant_manager.close_connection()

    def consistency_check(self, indexed_pdfs: List[Dict[str, str]]) -> bool:
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

    def _ingest_pdfs(self, uploaded_pdfs: List[PDFIngestItem]) -> IngestResult:
        """Ingest PDFs into cache and prepare for indexing.

        Args:
            uploaded_pdfs: List of PDFIngestItem objects for PDFs to ingest.

        Returns:
            IngestResult with summary and error details.
        """
        if not uploaded_pdfs:
            raise ValueError("uploaded_pdfs cannot be empty")

        result = IngestResult()
        by_hash: Dict[str, CachedPDF] = {}

        for uploaded_pdf in uploaded_pdfs:
            try:
                cached_pdf = self.pdf_cache.ingest_pdf(uploaded_pdf)
                if cached_pdf.pdf_hash in by_hash:
                    continue
                by_hash[cached_pdf.pdf_hash] = cached_pdf
            except Exception as e:
                logger.error(f"Error ingesting PDF '{uploaded_pdf.title}': {str(e)}")
                result.failed_uploads.append(
                    {"title": str(uploaded_pdf.title), "error": str(e)}
                )

        result.ingested_pdfs = list(by_hash.values())
        return result

    def ingest_pdfs_from_upload(self, uploaded_pdfs: List[UploadedFile]) -> IngestResult:
        """Ingest PDFs from user uploads.

        Args:
            uploaded_pdfs: List of UploadedFile objects from Streamlit file uploader.
        
        Returns:
            IngestResult with summary and error details.
        """
        if not uploaded_pdfs:
            raise ValueError("uploaded_pdfs cannot be empty")
        
        return self._ingest_pdfs(self.pdf_cache.get_items_from_upload(uploaded_pdfs))
        
    def ingest_pdfs_from_zotero(self, collection_name: str) -> IngestResult:
        """Ingest PDFs from a specific Zotero collection.

        Args:
            zotero_data_dir: Path to Zotero data directory. Auto-detect if None.
            collection_name: Name of the Zotero collection to ingest from.

        Returns:
            IngestResult with summary and error details.
        """ 
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        
        source = ZoteroDatabase(None)
        uploaded_pdfs = source.get_items(collection_name)
        
        return self._ingest_pdfs(uploaded_pdfs)

    def _rollback_pdfs(self, created_hashes: set[str], failed_hashes: set[str]) -> bool:
        """Remove cached PDFs created during ingest that failed processing."""
        to_remove = created_hashes & failed_hashes
        if not to_remove:
            return False

        for pdf_hash in to_remove:
            try:
                self.pdf_cache.remove_pdf(pdf_hash)
                logger.debug("Rolled back PDF from cache: %s", pdf_hash)
            except Exception as e:
                logger.error("Error rolling back PDF '%s': %s", pdf_hash, str(e))

        return True

    def upsert_pdfs(self, target_pdfs: List[CachedPDF], progress_callback=None) -> UpsertResult:
        """Process PDFs, extract paragraphs, and upsert into Qdrant index.
        
        Args:
            target_pdfs: List of cached PDFs to process. Must not be empty.
            progress_callback: Function(stage, current, total, message) for progress updates.
                             stage is one of: 'pdf', 'contextualization', 'encoding', 'upserting'.
        Returns:
            UpsertResult with indexing summary and warning/error details.
        """
        if target_pdfs is None or target_pdfs == []:
            raise ValueError("No new PDFs provided for upsert. Skipping indexing stage.")

        all_paragraphs = []
        per_pdf_context = {}
        result = UpsertResult()
        created_hashes: set[str] = set()
        try:
            self.qdrant_manager.open_connection()
            
            # Stage 1: Process PDFs and extract paragraphs
            created_hashes = {
                item.pdf_hash
                for item in target_pdfs
                if item.created
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
                        result.duplicate_title_titles.append(title)
                        if item.created:
                            removed = self.pdf_cache.remove_pdf(pdf_hash)
                            if removed:
                                created_hashes.discard(pdf_hash)
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
                            result.already_indexed_info.append({
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
                    except Exception as e:
                        logger.warning(
                            "Qdrant index check failed for '%s': %s", title, str(e),
                        )

                    paragraph_tuples, document_text = self.pdf_processor.extract_text_chunks(
                        pdf_hash,
                        pdf_title=title,
                    )

                    if pdf_hash not in per_pdf_context:
                        per_pdf_context[pdf_hash] = {
                            "title": title,
                            "document_text": document_text,
                            "paragraph_indices": [],
                        }

                    for text, page_num, para_idx, section, sentences in paragraph_tuples:
                        # Filter by section type if needed
                        if not self.pdf_processor.CONTENT_SECTIONS.get(section, True):
                            continue

                        sentence_count = len(sentences)
                        paragraph = Paragraph(
                            text=text,
                            page_num=page_num,
                            para_idx=para_idx,
                            title=title,
                            pdf_hash=pdf_hash,
                            section=section,
                            sentence_count=sentence_count,
                            sentences=sentences
                        )
                        all_paragraphs.append(paragraph)
                        per_pdf_context[pdf_hash]["paragraph_indices"].append(len(all_paragraphs) - 1)
                except Exception as e:
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

            if not all_paragraphs:
                if (not result.failed_pdfs
                    and not result.already_indexed_info
                    and not result.duplicate_title_titles):
                    raise ValueError("No text could be extracted from the selected PDFs.")
                return result

            # Stage 2: Build index
            all_texts = [p.text for p in all_paragraphs]
            if self.use_chunk_contextualization:
                try:
                    pdf_hashes_with_chunks = [
                        pdf_hash
                        for pdf_hash, context_info in per_pdf_context.items()
                        if context_info.get("paragraph_indices")
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
                        paragraph_indices = context_info.get("paragraph_indices", [])
                        if not paragraph_indices:
                            continue

                        document_text = context_info.get("document_text", "")
                        if not document_text.strip():
                            logger.warning("Skipping contextualization for %s due to empty document text", pdf_title)
                            continue

                        base_chunks = [all_texts[i] for i in paragraph_indices]
                        contextualized_chunks = self.embedding_manager.generate_contextual_chunks(
                            document_text=document_text,
                            all_texts=base_chunks,
                        )

                        if len(contextualized_chunks) != len(paragraph_indices):
                            logger.warning(
                                "Contextualization size mismatch for %s (%s vs %s), using original chunks",
                                pdf_title,
                                len(contextualized_chunks),
                                len(paragraph_indices),
                            )
                            continue

                        for offset, para_abs_idx in enumerate(paragraph_indices):
                            all_texts[para_abs_idx] = contextualized_chunks[offset]

                        if progress_callback:
                            progress_callback(
                                "contextualization",
                                pdf_idx + 1,
                                len(pdf_hashes_with_chunks),
                                f"Contextualized PDF {pdf_idx + 1}/{len(pdf_hashes_with_chunks)}: {pdf_title[:80]}",
                            )
                except Exception as e:
                    logger.warning("Contextual chunk generation failed, falling back to original chunks: %s", str(e))
                finally:
                    self.embedding_manager.flush_ollama_cache()
            else:
                logger.info("Chunk contextualization disabled: using original chunks for embedding.")

            contextual_texts = list(all_texts) #FIXME: debug
            hybrid_embeddings = self.embedding_manager.encode_paragraphs(progress_callback, all_texts)

            indexed_chunks, title_overrides = self.qdrant_manager.upsert_paragraphs(
                all_paragraphs,
                dense_embeddings=hybrid_embeddings["dense"],
                sparse_embeddings=hybrid_embeddings["sparse"],
                contextual_texts=contextual_texts, #FIXME: debug
                progress_callback=progress_callback,
            )
            result.indexed_chunks = indexed_chunks
            result.title_overrides = title_overrides
        except Exception as e:
            logger.error(f"Error during upsert_paragraphs: {str(e)}")
            raise
        finally:
            failed_hashes = {
                item.get("pdf_hash")
                for item in result.failed_pdfs
                if item.get("pdf_hash")
            }
            _ = self._rollback_pdfs(created_hashes, failed_hashes)
            self.qdrant_manager.close_connection()

        return result
    
    def delete_pdf_by_title(self, pdf_title: str) -> bool:
        """Delete all paragraphs from a specific PDF in the index.
        
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
                logger.warning("No paragraphs found to delete for PDF: %s", pdf_title)
                return False

            delete_result = self.qdrant_manager.delete_pdf_from_index(pdf_hash)
            deleted = delete_result.get("deleted", False)
            if deleted:
                if not delete_result.get("had_other_models", False):
                    self.pdf_processor.remove_cache_item(pdf_hash)
                    self.pdf_cache.remove_pdf(pdf_hash)
                logger.info("Successfully deleted paragraphs for PDF: %s", pdf_title)
                return True

            logger.warning("No paragraphs found to delete for PDF: %s", pdf_title)
            return False
        except Exception as e:
            logger.error(f"Error during delete_pdf_from_index: {str(e)}")
            return False
        finally:
            self.qdrant_manager.close_connection()

    def clear_index(self) -> bool:
        """Clear the entire Qdrant collection, removing all indexed paragraphs.
        
        Returns:
            True if the collection was cleared successfully, False otherwise.
        """
        try:
            self.qdrant_manager.open_connection()
            deleted_pdfs = self.qdrant_manager.clear_collection()
            self.pdf_cache.clear_index_cache(deleted_pdfs)
            self.pdf_processor.clear_index_cache(deleted_pdfs)
            logger.info("Successfully cleared the Qdrant collection.")
            return True
        except Exception as e:
            logger.error(f"Error during clear_collection: {str(e)}")
            return False
        finally:
            self.qdrant_manager.close_connection()

    def answer_question(self, 
                       question: str, 
                       retrieval_threshold: float = 0.7, 
                       qa_score_threshold: float = 0.0, 
                       rerank_threshold: float = 0.25, 
                       progress_callback=None, 
                       rerank_callback=None,
                       question_type: str = 'general',
                       custom_config: dict = None,
                       num_paraphrases: int = 2,
                       highlight_color: Tuple[float, float, float] = None,
                       question_variations: List[str] = None) -> List[Answer]:
        """Answer a question using the full RAG pipeline.
        
        Pipeline stages:
        1. Qdrant Retrieval (Cosine Similarity)
        2. CrossEncoder Reranking (Threshold Filtering)
        3. QA Extraction (with Context Overlap/Sliding Window)
        
        Args:
            question: The question to answer.
            retrieval_threshold: Minimum cosine similarity score to keep retrieved paragraphs.
            qa_score_threshold: Minimum QA confidence score to keep answers.
            rerank_threshold: Minimum rerank probability to keep candidates.
            progress_callback: Function(current, total, message) for QA progress.
            rerank_callback: Function(current, total, message) for rerank progress.
            question_type: Type of question (factoid, explanation, methodology, etc.).
            custom_config: Custom configuration dict to override preset config.
            num_paraphrases: Number of question paraphrases to generate (0 = disabled).
            highlight_color: RGB tuple (0-1) for highlighting. If None, use query-based color.
            question_variations: Pre-generated question variations to use. If None, generate them.
            
        Returns:
            List of Answer objects, deduplicated and sorted by score.
        """
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
        
        # Stage 1: Retrieve candidate paragraphs (Qdrant)
        # Search with all question variations and merge results
        try:
            all_candidates = []
            seen_paragraphs = set()
            self.qdrant_manager.open_connection()
            
            for i, q_var in enumerate(question_variations):
                query_embeddings = self.embedding_manager.encode_query(q_var)
                var_candidates = self.qdrant_manager.search(
                    query_embeddings["dense"],
                    query_embeddings["sparse"],
                    retrieval_threshold,
                )
                logger.debug(f"Variation {i}: '{q_var}' -> {len(var_candidates)} candidates")
                
                # Add unseen candidates
                for para, score in var_candidates:
                    para_id = (para.pdf_hash, para.para_idx)  # Unique identifier
                    if para_id not in seen_paragraphs:
                        seen_paragraphs.add(para_id)
                        all_candidates.append((para, score))
            
            # Sort by retrieval score
            all_candidates.sort(key=lambda x: x[1])
            candidates = all_candidates
            
            logger.debug(f"Question: {question}")
            logger.debug(f"Retrieved {len(candidates)} unique paragraphs from {len(question_variations)} variations")
            
            if not candidates:
                self.last_candidates = []
                return []
            
            # Store for debugging
            self.last_candidates = [
                {
                    'paragraph': c[0],
                    'retrieval_score': c[1],
                    'kept': True
                }
                for c in candidates
            ]
        finally:
            self.qdrant_manager.close_connection()
        
        # Stage 2: Rerank and Filter (CrossEncoder)
        reranked = self.reranker.rerank(
            question, 
            candidates, 
            rerank_threshold,
            progress_callback=rerank_callback,
            query_variations=question_variations
        )
        
        # Update debug info with rerank results
        reranked_texts = {c[0].text for c in reranked}
        for c in self.last_candidates:
            c['kept'] = c['paragraph'].text in reranked_texts
        
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
            qa_score_threshold=qa_score_threshold,
            color=color,
            progress_callback=progress_callback,
            question_variations=question_variations,
            question_type=question_type,
            custom_config=custom_config
        )
        
        return self._attach_pdf_paths(answers)
    
    def _attach_pdf_paths(self, answers: List[Answer]) -> List[Answer]:
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
    
    def highlight_pdf(self, answers_for_pdf: List[Answer], output_path: str) -> str:
        """Highlight PDF using TEI sentence coordinates.
        
        Args:
            answers_for_pdf: List of Answer objects from the same PDF.
            output_path: Path where the highlighted PDF should be saved.
            
        Returns:
            Path to the highlighted PDF, or None if highlighting failed.
        """
        return self.highlighter.highlight_pdf(answers_for_pdf, output_path)