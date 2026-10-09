"""Main orchestration class for Zotero RAG system."""

import os

# Suppress noisy progress bars that can trigger BrokenPipe in the web server
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import logging
import time
import warnings
from typing import BinaryIO

from zotero_rag.answering.qa_engine import QAEngine
from zotero_rag.ingest.pdf_cache_manager import PDFCacheManager
from zotero_rag.ingest.pdf_processor import PDFProcessor
from zotero_rag.ingest.zotero_db import ZoteroDatabase
from zotero_rag.models import (
    YELLOW,
    Answer,
    CachedPDF,
    Chunk,
    IngestResult,
    PDFIngestItem,
    QuestionTrace,
    Reader,
    Reference,
    RerankedChunk,
    RetrievedChunk,
    UpsertResult,
    ingest_items_from_folder,
    ingest_items_from_upload,
)
from zotero_rag.retrieval.embedding_manager import DEFAULT_DENSE_MODEL, EmbeddingManager
from zotero_rag.retrieval.qdrant_manager import QdrantManager, best_first
from zotero_rag.retrieval.reranker import DEFAULT_RERANKER_MODEL, Reranker
from zotero_rag.settings import resolve

warnings.filterwarnings('ignore', message='.*position_ids.*')

# Handlers and levels are the entry point's to set (server.py, run_from_config.py):
# a library module that adds its own writes a second copy of every line.
logger = logging.getLogger(__name__)


class ZoteroRAG:
    """Main orchestration class for the Zotero RAG pipeline."""
    
    def __init__(self, 
                dense_model_name: str = DEFAULT_DENSE_MODEL,
                reranker_model: str = DEFAULT_RERANKER_MODEL,
                grobid_url: str = "http://localhost:8070", 
                grobid_timeout: int = 180,
                qdrant_url: str | None = "http://localhost:6333",
                model_device: str | None = None, 
                encode_batch_size: int | None = None,
                qa_batch_size: int | None = None,
                rerank_batch_size: int | None = None,
                output_base_dir: str = "output",
                qdrant_collection_suffix: str = "",
                reader: Reader | None = None):
        """Initialize the RAG system.
        
        Args:
            dense_model_name: Name of the sentence-transformers dense model for embeddings.
            reranker_model: Name of the cross-encoder model for reranking.
            grobid_url: URL of the GROBID service.
            grobid_timeout: Timeout in seconds for GROBID requests.
            qdrant_url: URL of a Qdrant server, or None for the embedded storage
                in ``output_base_dir``/qdrant, which needs no server.
            model_device: Device to use for models ('cpu', 'cuda', 'mps'). Auto-detect if None.
            encode_batch_size: Batch size for encoding. Module default if None.
            qa_batch_size: Batch size for QA extraction. Module default if None.
            rerank_batch_size: Batch size for reranking. Module default if None.
            output_base_dir: Base directory for storing outputs.
            qdrant_collection_suffix: Appended to the Qdrant collection names, to keep
                separate corpora out of each other's retrieval pool.
            reader: The reader to answer with. None builds the delivered one,
                the extractive reader; a benchmark passes the reader it measures.
        """
        pdf_cache = PDFCacheManager(folder_path=os.path.join(output_base_dir, "pdf_cache"))

        pdf_processor = PDFProcessor(
            grobid_url=grobid_url,
            grobid_timeout=grobid_timeout,
            output_base_dir=output_base_dir
        )

        embedding_manager = EmbeddingManager(
            dense_model_name=dense_model_name,
            device=model_device,
            encode_batch_size=encode_batch_size,
        )

        self._assemble(
            pdf_cache=pdf_cache,
            pdf_processor=pdf_processor,
            embedding_manager=embedding_manager,
            qdrant_manager=QdrantManager(
                dense_model_name=dense_model_name,
                qdrant_url=qdrant_url,
                qdrant_path=os.path.join(output_base_dir, "qdrant"),
                vector_size=embedding_manager.vector_size,
                collection_suffix=qdrant_collection_suffix,
            ),
            reranker=Reranker(
                model_name=reranker_model,
                device=model_device,
                batch_size=rerank_batch_size
            ),
            reader=reader or QAEngine(device=model_device, batch_size=qa_batch_size),
        )

    @classmethod
    def from_components(cls, *, qdrant_manager=None, embedding_manager=None,
                        reranker=None, reader: Reader | None = None, pdf_cache=None,
                        pdf_processor=None) -> "ZoteroRAG":
        """A pipeline over components built elsewhere, loading no model itself.

        The seam for the tests, which pass stand-ins for only the components the
        call under test touches. A component left out is None, and whatever
        reaches for it fails.
        """
        rag = cls.__new__(cls)
        rag._assemble(pdf_cache=pdf_cache, pdf_processor=pdf_processor,
                      embedding_manager=embedding_manager,
                      qdrant_manager=qdrant_manager, reranker=reranker, reader=reader)
        return rag

    def _assemble(self, *, pdf_cache, pdf_processor, embedding_manager,
                  qdrant_manager, reranker, reader) -> None:
        """Wire the components and the per-instance state every constructor needs.

        ``reader`` is public: the benchmarks point it at the reader a config asks
        for (see ``models.Reader``).
        """
        self.pdf_cache = pdf_cache
        self.pdf_processor = pdf_processor
        self.embedding_manager = embedding_manager
        self.qdrant_manager = qdrant_manager
        self.reranker = reranker
        self.reader = reader

        # What the last answer_question call did at each stage, for the
        # benchmarks: see QuestionTrace.
        self.last_trace = QuestionTrace()
    
    def reference(self, pdf_hash: str | None) -> Reference:
        """How to cite the paper an answer came from; an empty one if nothing is known.

        A paper indexed before references were kept gets one now, from its cached
        TEI, and keeps it.
        """
        stored = self.qdrant_manager.reference(pdf_hash) if pdf_hash else None
        if stored is not None or not pdf_hash:
            return stored or Reference([], None)
        try:
            stored = self.pdf_processor.extract_reference(pdf_hash)
        except ValueError:  # neither TEI nor GROBID: no citation rather than no answer
            return Reference([], None)
        self.qdrant_manager.set_reference(pdf_hash, stored)
        return stored

    def get_indexed_pdfs(self) -> list[dict[str, str]]:
        """Get list of indexed PDFs.
        
        Returns:
            List of dictionaries with title and pdf_hash.
        """
        return self.qdrant_manager.list_indexed_pdfs()

    def consistency_check(self, indexed_pdfs: list[dict[str, str]]) -> bool:
        """Check consistency between indexed PDFs and cached PDFs.

        Args:
            indexed_pdfs: The PDFs indexed in Qdrant, as ``list_indexed_pdfs`` returns them.

        Returns:
            True if all indexed PDFs have corresponding cache entries, False otherwise.
        """
        if not indexed_pdfs:
            return True

        indexed_hashes = {item["pdf_hash"] for item in indexed_pdfs if item.get("pdf_hash")}
        missing_in_cache = indexed_hashes - set(self.pdf_cache.get_cached_items())

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
                by_hash.setdefault(cached_pdf.pdf_hash, cached_pdf)
            except Exception as e:  # noqa: BLE001 - per-PDF: record the failure and continue
                logger.error(f"Error ingesting PDF '{uploaded_pdf.title}': {e!s}")
                result.failed_pdfs.append(
                    {"title": str(uploaded_pdf.title), "error": str(e)}
                )

        result.ingested_pdfs = list(by_hash.values())
        return result

    def ingest_pdfs_from_upload(self, uploaded_pdfs: list[BinaryIO]) -> IngestResult:
        """Ingest PDFs from user uploads.

        Args:
            uploaded_pdfs: Binary streams carrying the upload's file name in ``name``.
        
        Returns:
            IngestResult with summary and error details.
        """
        return self._ingest_pdfs(ingest_items_from_upload(uploaded_pdfs))
        
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

    def _remove_cached_files(self, pdf_hash: str) -> None:
        """Drop the paper's cached PDF and TEI."""
        self.pdf_cache.remove_pdf(pdf_hash)
        self.pdf_processor.remove_tei_cache(pdf_hash)

    def _rollback_pdfs(self, newly_cached_hashes: set[str], failed_hashes: set[str]) -> None:
        """Rollback cached PDFs that were newly cached during the current upsert if they failed to index.

        Args:
            newly_cached_hashes: Set of PDF hashes that were newly cached during the current upsert.
            failed_hashes: Set of PDF hashes that failed to index during the current upsert.
        """
        for pdf_hash in newly_cached_hashes & failed_hashes:
            try:
                self._remove_cached_files(pdf_hash)
                logger.debug("Rolled back PDF from cache: %s", pdf_hash)
            except Exception as e:  # noqa: BLE001 - rollback is best-effort
                logger.error("Error rolling back PDF '%s': %s", pdf_hash, str(e))

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
                        progress_callback=None) -> tuple[list[Chunk], set[str]]:
        """Stage 1: the chunks of every PDF that needs indexing.

        A PDF that fails is recorded in ``result`` and the others go on.

        Returns:
            ``(all_chunks, processed)``: the latter the pdf_hash of every PDF
            whose chunks were extracted.
        """
        all_chunks = []
        processed = set()
        total = len(target_pdfs)
        progress = progress_callback or (lambda *_: None)
        for idx, item in enumerate(target_pdfs):
            title = item.title
            pdf_hash = item.pdf_hash
            progress('pdf', idx, total, f"Analysing PDF {idx + 1}/{total}: {title[:80]}")

            try:
                skip = self._skip_message(item, result, newly_cached_hashes)
                if skip:
                    progress('pdf', idx + 1, total, skip)
                    continue

                canonical_title = self.qdrant_manager.register_pdf_title(pdf_hash, title)
                if canonical_title:
                    title = canonical_title

                extracted_chunks = self.pdf_processor.extract_text_chunks(
                    pdf_hash,
                    pdf_title=title,
                )
                # Zotero's metadata is curated by the user; the PDF's header is
                # GROBID's reading of it, the best there is for an upload.
                self.qdrant_manager.set_reference(
                    pdf_hash, item.reference or self.pdf_processor.extract_reference(pdf_hash))

                processed.add(pdf_hash)
                all_chunks.extend(Chunk(**vars(extracted), title=title, pdf_hash=pdf_hash)
                                  for extracted in extracted_chunks)
            except Exception as e:  # noqa: BLE001 - per-PDF: record the failure and continue
                logger.error("Failed to process PDF '%s': %s", title, str(e))
                result.failed_pdfs.append({"title": title, "error": str(e), "pdf_hash": pdf_hash})

            progress('pdf', idx + 1, total, f"Analysed PDF {idx + 1}/{total}: {title[:80]}")
        return all_chunks, processed

    def upsert_pdfs(self, target_pdfs: list[CachedPDF], progress_callback=None) -> UpsertResult:
        """Process PDFs, extract chunks, and upsert into Qdrant index.
        
        Args:
            target_pdfs: List of cached PDFs to process. Must not be empty.
            progress_callback: Optional function(current_stage, current_idx, total, message) for progress updates.
        Returns:
            UpsertResult with indexing summary and warning/error details.
        """
        if not target_pdfs:
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
            all_chunks, processed = self._extract_chunks(
                target_pdfs, result, newly_cached_hashes, progress_callback)
            result.processed_pdfs = len(processed)

            if not all_chunks:
                if (not result.failed_pdfs
                    and not result.already_indexed
                    and not result.duplicate_titles):
                    raise ValueError("No text could be extracted from the selected PDFs.")
                return result

            # Stage 2: embed and index
            hybrid_embeddings = self.embedding_manager.encode_chunks(
                [c.text for c in all_chunks], progress_callback)

            indexed_chunks, title_overrides = self.qdrant_manager.upsert_chunks(
                all_chunks,
                dense_embeddings=hybrid_embeddings["dense"],
                sparse_embeddings=hybrid_embeddings["sparse"],
                progress_callback=progress_callback,
            )
            result.indexed_chunks = indexed_chunks
            result.title_overrides = title_overrides
        finally:
            failed_hashes = {
                item.get("pdf_hash")
                for item in result.failed_pdfs
                if item.get("pdf_hash")
            }
            self._rollback_pdfs(newly_cached_hashes, failed_hashes)

        return result
    
    def delete_pdf_by_title(self, pdf_title: str) -> bool:
        """Delete all chunks from a specific PDF in the index.
        
        Args:
            pdf_title: Title of the PDF to delete.
        
        Returns:
            True if the PDF was deleted, False otherwise.
        """
        try:
            pdf_hash = self.qdrant_manager.find_pdf_hash_by_title(
                pdf_title.strip(),
                filter_by_model=True,
            )
            if not pdf_hash:
                logger.warning("No chunks found to delete for PDF: %s", pdf_title)
                return False

            if not self.qdrant_manager.delete_pdf_from_index(pdf_hash):
                self._remove_cached_files(pdf_hash)
            logger.info("Successfully deleted chunks for PDF: %s", pdf_title)
            return True
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

        found = []
        for chunk in self.qdrant_manager.fetch_chunks(list(wanted)):
            seen.add(chunk.key)
            found.append((chunk, wanted[chunk.key]))

        logger.debug(f"Neighbour expansion (span {span}): "
                     f"{len(found)} added to {len(retrieved)} retrieved")
        return found

    def _retrieve(self, question: str, config: dict,
                  pdf_hashes: list[str] | None) -> tuple[list, list[RetrievedChunk]]:
        """Stage 1: the hybrid hits and their neighbours.

        One query: expanding it with paraphrases was measured, never added
        recall, and was removed.

        Returns:
            The (chunk, score) candidates in the order the reranker takes them,
            and the same chunks in retrieval order for the trace.
        """
        hits = self.qdrant_manager.search(
            self.embedding_manager.encode_query(question),
            result_limit=config['result_limit'],
            mode=config['retrieval_mode'],
            pdf_hashes=pdf_hashes,
        )
        seen_chunks = {chunk.key for chunk, _ in hits}

        # Best first, ties broken by chunk id. RRF gives equal scores to
        # chunks at the same rank in different branches and Qdrant returns
        # equals in no stable order: one config ranked 117 of 120 questions
        # differently across campaigns, mostly by adjacent swaps. Every
        # consumer re-sorts by its own score with a stable sort, so this is
        # the tie order they all inherit.
        hits.sort(key=best_first)
        neighbours = self._neighbour_candidates(
            hits, seen_chunks, config.get('retrieval_neighbours', 0))

        logger.debug(f"Question: {question}")
        logger.debug(f"Retrieved {len(hits) + len(neighbours)} chunks")

        # The retrieval order: every neighbour after every hit, best first
        # inside each group. A neighbour inherits its parent's score, so on
        # the score alone a chunk nothing scored could outrank a hit that
        # scored lower and take a position retrieval never assigned it.
        retrieved = (
            [RetrievedChunk(chunk, score) for chunk, score in hits]
            + [RetrievedChunk(chunk, score, is_neighbour=True)
               for chunk, score in sorted(neighbours, key=lambda c: -c[1])])
        return hits + neighbours, retrieved

    def _rerank(self, question: str, candidates: list, retrieved: list[RetrievedChunk],
                config: dict, progress_callback) -> list[RerankedChunk]:
        """Stage 2: the cross-encoder's gate and margin, or the retrieval order when it is off."""
        if config['rerank_enabled']:
            return self.reranker.rerank(
                question,
                candidates,
                config['rerank_gate'],
                config['rerank_margin'],
                progress_callback=progress_callback,
                # Read with defaults rather than added to settings: both exist for
                # the benchmarks alone, and a setting would ship them as defaults.
                order_by_retrieval=config.get('rerank_order_by_retrieval', False),
                absolute_cut=config.get('rerank_absolute', False),
            )
        # ponytail: bypass keeps the retrieval order so the ablation can
        # attribute the reranker's contribution; rerank_score mirrors
        # retrieval_score because nothing rescored the candidates.
        return [RerankedChunk(chunk=r.chunk, retrieval_score=r.retrieval_score,
                              rerank_score=r.retrieval_score)
                for r in retrieved]

    def answer_question(self,
                    question: str,
                    overrides: dict | None = None,
                    progress_callback=None,
                    rerank_callback=None,
                    highlight_color: tuple[float, float, float] | None = None,
                    pdf_hashes: list[str] | None = None) -> list[Answer]:
        """Answer a question using the full RAG pipeline.

        Pipeline stages:
        1. Hybrid retrieval (Qdrant, fused client-side)
        2. Cross-encoder reranking (gate and margin)
        3. Reading (``self.reader``)

        Args:
            question: The question to answer.
            overrides: Changes to the fixed settings (see ``settings``), for the
                benchmarks that vary them one at a time. The product passes none.
            progress_callback: Function(current, total, message) for QA progress.
            rerank_callback: Function(current, total, message) for rerank progress.
            highlight_color: RGB tuple (0-1) for highlighting. If None, yellow.
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

        try:
            candidates, trace.retrieved = self._retrieve(question, config, pdf_hashes)
        finally:
            mark("retrieval")
        if not candidates:
            return []

        trace.reranked = self._rerank(question, candidates, trace.retrieved, config,
                                      rerank_callback)  # already sorted by rerank_score desc
        mark("rerank")
        if not trace.reranked:
            return []

        # Stage 3: Extract answers (QA Model)
        answers = self.reader.extract_answers(
            question,
            trace.reranked,
            config,
            color=highlight_color or YELLOW,
            progress_callback=progress_callback,
        )
        mark("read")

        for ans in answers:
            ans.pdf_path = self.pdf_cache.get_pdf_path(ans.pdf_hash)
        return answers