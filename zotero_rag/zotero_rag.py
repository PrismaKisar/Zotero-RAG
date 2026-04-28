"""Main orchestration class for Zotero RAG system."""

import os
# Suppress noisy progress bars that can trigger BrokenPipe in Streamlit
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import logging
from typing import List, Dict, Tuple, Union, Any
import warnings
import nltk
from streamlit.runtime.uploaded_file_manager import UploadedFile

from models import Paragraph, Answer
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
    def _sanitize_filename(name: str) -> str:
        """Converts a string into a safe folder/file name."""
        import re
        if not name:
            return "_All_Library"
        s = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        s = re.sub(r'(?u)[^-\w.]', '', s)
        return s
    
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
    
    def get_indexed_pdfs(self) -> List[Dict]:
        """Get list of indexed PDFs.
        
        Returns:
            List of dictionaries with 'title', 'chunk_count'.
        """
        try:
            self.qdrant_manager.initialize_connection()
            return self.qdrant_manager.list_indexed_pdfs()
        except Exception as e:
            logger.error(f"Error during get_indexed_pdfs: {str(e)}")
            return []
        finally:
            self.qdrant_manager.close_connection()

    def ingest_pdf(self, uploaded_pdfs: List[UploadedFile]) -> Dict[str, Union[int, List[str], List[Dict[str, str]]]]:
        """Ingest multiple uploaded PDFs and return their metadata.

        Args:
            uploaded_pdfs: List of UploadedFile objects from Streamlit file uploader.

        Returns:
            Dictionary with ingestion counters and list of ingested PDF titles.
        """
        if not uploaded_pdfs:
            raise ValueError("uploaded_pdfs cannot be empty")
        
        ingested_count = 0
        already_indexed_count = 0
        ingested_titles: List[str] = []
        already_indexed_titles: List[str] = []
        failed_uploads: List[Dict[str, str]] = []
        for uploaded_pdf in uploaded_pdfs:
            try:
                ingest_result = self.pdf_cache.ingest_pdf(uploaded_pdf)
                if ingest_result["already_indexed"]:
                    already_indexed_count += 1
                    already_indexed_titles.append(str(ingest_result["title"]))
                else:
                    ingested_count += 1
                    ingested_titles.append(str(ingest_result["title"]))
            except Exception as e:
                logger.error(f"Error ingesting PDF '{uploaded_pdf.name}': {str(e)}")
                failed_uploads.append({"name": str(uploaded_pdf.name), "error": str(e)})

        return {
            "ingested": ingested_count,
            "already_indexed": already_indexed_count,
            "ingested_titles": ingested_titles,
            "already_indexed_titles": already_indexed_titles,
            "failed_uploads": failed_uploads,
        }
    
    def _rollback_pdfs(self, uploaded_pdf: set[str], indexed_pdf: set[str]) -> bool:
        """Determine which PDFs to remove from cache based on ingestion vs indexing results.
        
        Args:
            uploaded_pdf: Set of PDF titles that were uploaded/ingested.
            indexed_pdf: Set of PDF titles that were actually indexed in Qdrant.

        Returns:
            True if rollback was performed, False otherwise.
        """
        to_remove = uploaded_pdf - indexed_pdf
        if not to_remove:
            return False
        
        for title in to_remove:
            try:
                self.pdf_cache.remove_pdf(title)
                logger.debug(f"Rolled back PDF from cache: {title}")
            except Exception as e:
                logger.error(f"Error rolling back PDF '{title}': {str(e)}")
        
        return True

    def upsert_pdfs(self, target_pdf_titles: List[str], progress_callback=None) -> Dict[str, Any]:
        """Process PDFs, extract paragraphs, and upsert into Qdrant index.
            If a pdf has already been indexed (based on hash), it will be skipped.
        
        Args:
            target_pdf_titles: List of PDF titles to process. Must not be empty.
            progress_callback: Function(stage, current, total, message) for progress updates.
                             stage is one of: 'pdf', 'contextualization', 'encoding', 'upserting'.
        Returns:
            Dictionary with indexing summary and warning/error details.
        """
        if target_pdf_titles is None or target_pdf_titles == []:
            raise ValueError("No new PDF titles provided for upsert. Skipping indexing stage.")

        indexed = 0
        all_paragraphs = []
        per_pdf_context = {}
        skipped_already_indexed_titles: List[str] = []
        failed_pdf_errors: List[Dict[str, str]] = []

        try:
            self.qdrant_manager.initialize_connection()
            
            # Stage 1: Process PDFs and extract paragraphs
            for idx, title in enumerate(target_pdf_titles):
                if progress_callback:
                    progress_callback(
                        'pdf',
                        idx,
                        len(target_pdf_titles),
                        f"Analysing PDF {idx + 1}/{len(target_pdf_titles)}: {title[:80]}",
                    )

                try:
                    current_pdf_hash = self.pdf_processor.compute_pdf_hash(title)

                    if self.qdrant_manager.is_pdf_indexed(current_pdf_hash):
                        logger.info(f"Skipping already indexed PDF: {title}")
                        skipped_already_indexed_titles.append(title)
                        continue

                    paragraph_tuples, document_text = self.pdf_processor.extract_text_chunks(title)

                    if title not in per_pdf_context:
                        per_pdf_context[title] = {
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
                            pdf_hash=current_pdf_hash,
                            section=section,
                            sentence_count=sentence_count,
                            sentences=sentences
                        )
                        all_paragraphs.append(paragraph)
                        per_pdf_context[title]["paragraph_indices"].append(len(all_paragraphs) - 1)
                except Exception as e:
                    logger.error("Failed to process PDF '%s': %s", title, str(e))
                    failed_pdf_errors.append({"title": title, "error": str(e)})

                if progress_callback:
                    progress_callback(
                        'pdf',
                        idx + 1,
                        len(target_pdf_titles),
                        f"Analysed PDF {idx + 1}/{len(target_pdf_titles)}: {title[:80]}",
                    )

            if not all_paragraphs:
                if skipped_already_indexed_titles and not failed_pdf_errors:
                    logger.info("No new paragraphs to index. All PDFs were already indexed.")
                    return {
                        "indexed_chunks": 0,
                        "submitted_pdfs": len(target_pdf_titles),
                        "processed_pdfs": 0,
                        "skipped_already_indexed_pdfs": len(skipped_already_indexed_titles),
                        "skipped_already_indexed_titles": skipped_already_indexed_titles,
                        "failed_pdfs": failed_pdf_errors,
                    }
                raise ValueError("No text could be extracted from the selected PDFs.")

            # Stage 2: Build index
            all_texts = [p.text for p in all_paragraphs]
            if self.use_chunk_contextualization:
                try:
                    pdf_titles_with_chunks = [
                        pdf_title
                        for pdf_title, context_info in per_pdf_context.items()
                        if context_info.get("paragraph_indices")
                    ]

                    for pdf_idx, pdf_title in enumerate(pdf_titles_with_chunks):
                        if progress_callback:
                            progress_callback(
                                "contextualization",
                                pdf_idx,
                                len(pdf_titles_with_chunks),
                                f"Contextualizing PDF {pdf_idx + 1}/{len(pdf_titles_with_chunks)}: {pdf_title[:80]}",
                            )

                        context_info = per_pdf_context[pdf_title]
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
                                len(pdf_titles_with_chunks),
                                f"Contextualized PDF {pdf_idx + 1}/{len(pdf_titles_with_chunks)}: {pdf_title[:80]}",
                            )
                except Exception as e:
                    logger.warning("Contextual chunk generation failed, falling back to original chunks: %s", str(e))
                finally:
                    self.embedding_manager.flush_ollama_cache()
            else:
                logger.info("Chunk contextualization disabled: using original chunks for embedding.")

            hybrid_embeddings = self.embedding_manager.encode_paragraphs(progress_callback, all_texts)

            indexed = self.qdrant_manager.upsert_paragraphs(
                all_paragraphs,
                dense_embeddings=hybrid_embeddings["dense"],
                sparse_embeddings=hybrid_embeddings["sparse"],
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.error(f"Error during upsert_paragraphs: {str(e)}")
            raise
        finally:
            _ = self._rollback_pdfs(set(target_pdf_titles), set(per_pdf_context.keys()))
            self.qdrant_manager.close_connection()

        return {
            "indexed_chunks": indexed,
            "submitted_pdfs": len(target_pdf_titles),
            "processed_pdfs": len(per_pdf_context),
            "skipped_already_indexed_pdfs": len(skipped_already_indexed_titles),
            "skipped_already_indexed_titles": skipped_already_indexed_titles,
            "failed_pdfs": failed_pdf_errors,
        }
    
    def delete_pdf_by_title(self, pdf_title: str) -> bool:
        """Delete all paragraphs from a specific PDF in the index.
        
        Args:
            pdf_title: Title (or filename/path) of the PDF to delete.
        
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            self.qdrant_manager.initialize_connection()

            normalized_title = self._sanitize_filename(pdf_title)

            success = self.qdrant_manager.delete_pdf_from_index(normalized_title)
            if success:
                self.pdf_processor.remove_cache_item(normalized_title)
                self.pdf_cache.remove_pdf(normalized_title)
                logger.info(f"Successfully deleted paragraphs for PDF: {normalized_title}")
            else:
                logger.warning(f"No paragraphs found to delete for PDF: {normalized_title}")
            return success
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
            self.qdrant_manager.initialize_connection()
            self.qdrant_manager.clear_collection()
            self.pdf_cache.clear_cache()
            self.pdf_processor.clear_cache()
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
            self.qdrant_manager.initialize_connection()
            
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