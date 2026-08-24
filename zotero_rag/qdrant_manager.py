"""QdrantManager class for managing Qdrant vector database operations."""

import logging
import re
import uuid

import qdrant_client as qc
from models import Chunk
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manage Qdrant vector database for storing and searching chunk vectors."""
    
    def __init__(self, 
                dense_model_name: str = "BAAI/bge-base-en-v1.5", 
                qdrant_url: str = "http://localhost:6333",
                vector_size: int = 768,
                collection_suffix: str = ""):
        """Initialize the Qdrant manager.

        Args:
            dense_model_name: Name used to derive a deterministic collection name.
            qdrant_url: URL of the Qdrant service.
            vector_size: Dimensionality of the dense vectors stored in Qdrant.
            collection_suffix: Appended to both collection names, to keep separate
                corpora (e.g. two benchmark datasets) from sharing a retrieval pool.
        """
        self.dense_model_name = dense_model_name
        self.qdrant_url = qdrant_url
        self.collection_suffix = collection_suffix
        self.chunk_collection = (
            "zotero_rag_" + self._sanitize_model_name(dense_model_name) + collection_suffix)
        self.registry_collection = "zotero_rag_registry" + collection_suffix
        self.vector_size = vector_size
        self.chunks: list[Chunk] = []
        self.client: qc.QdrantClient | None = None
        self.conn_initialized = False
    
    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        """Convert model name to safe filename component.
        
        Args:
            model_name: Original model name string.
        
        Returns:
            Sanitized string suitable for use in collection names.
        """
        model_short = model_name.split('/')[-1]
        return re.sub(r'[^a-zA-Z0-9_-]', '_', model_short)
    
    @staticmethod
    def _generate_point_id(pdf_hash: str, chunk_index: int) -> str:
        """Generate a unique point ID for Qdrant.
        
        Args:
            pdf_hash: Hash of the PDF document.
            chunk_index: Index of the chunk within the PDF (can be None for registry entries).
            
        Returns:
            A deterministic UUID string based on the PDF hash and chunk index.
        """
        if chunk_index is None:
            input_str = f"{pdf_hash}"
        else:
            input_str = f"{pdf_hash}_{chunk_index}"
        NAMESPACE_RAG = uuid.UUID("12345678-1234-5678-1234-567812345678")
        return str(uuid.uuid5(NAMESPACE_RAG, input_str))

    def _start_connection(self):
        """Start connection to Qdrant client and verify connectivity."""
        self.client = qc.QdrantClient(
            url=self.qdrant_url,
        )
        try:
            # Force a round-trip to verify the service is reachable.
            self.client.get_collections()
        except Exception as exc:
            if self.client:
                try:
                    self.client.close()
                except Exception:
                    logger.debug("Error closing unreachable Qdrant client", exc_info=True)
            self.client = None
            message = (
                f"Qdrant service not reachable at {self.qdrant_url}. "
                "Start Qdrant and retry."
            )
            logger.error(message)
            raise ConnectionError(message) from exc

        logger.info("Connected to Qdrant client")

    def _initialize_connection(self):
        """Initialize Qdrant client and ensure collections and indexes are set up."""
        self._start_connection()
        
        if not self.client.collection_exists(self.chunk_collection):
            self.client.create_collection(
                collection_name=self.chunk_collection,
                hnsw_config=qmodels.HnswConfigDiff(
                    ef_construct=100,
                ),
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                    datatype=qmodels.Datatype.FLOAT16,
                    # ponytail: thesis-scale corpus (~1e5 chunks fp16 ~= 150MB) fits in RAM;
                    # on_disk pays mmap latency for nothing. Flip to True past ~1e7 chunks.
                    on_disk=False
                ),
                sparse_vectors_config={
                    "text-sparse": qmodels.SparseVectorParams(
                        modifier=qmodels.Modifier.IDF,
                        index=qmodels.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
            )

            # Create an index on 'pdf_hash' payload field for efficient lookups
            self.client.create_payload_index(
                collection_name=self.chunk_collection,
                field_name="pdf_hash",
                field_schema=qmodels.KeywordIndexParams(
                    type="keyword",
                    # ponytail: search_batch(pdf_hashes=...) *is* a filtered vector
                    # search, so this is no longer free - without the filterable
                    # sub-index Qdrant falls back to exact search on the filtered
                    # set. At benchmark scale (~5k chunks) that is fine; flip to
                    # True and reindex if a scoped search ever gets slow.
                    enable_hnsw=False,
                    on_disk=True
                ),
            )

            logger.info(f"Created Qdrant chunk collection: {self.chunk_collection}")
        else:
            logger.info(f"Qdrant chunk collection already exists: {self.chunk_collection}")

        if not self.client.collection_exists(self.registry_collection):
            self.client.create_collection(
                collection_name=self.registry_collection,
                vectors_config=None,
            )

            # Create an index on 'models' payload field for efficient lookups
            self.client.create_payload_index(
                collection_name=self.registry_collection,
                field_name="models",
                field_schema=qmodels.KeywordIndexParams(
                    type="keyword",
                    enable_hnsw=False,
                ),
            )
            # Create an index on title to allow lookups by name
            self.client.create_payload_index(
                collection_name=self.registry_collection,
                field_name="title",
                field_schema=qmodels.KeywordIndexParams(
                    type="keyword",
                    enable_hnsw=False,
                ),
            )
            logger.info(f"Created Qdrant registry collection: {self.registry_collection}")
        else:
            logger.info(f"Qdrant registry collection already exists: {self.registry_collection}")

        self.conn_initialized = True

    def open_connection(self):
        """Open connection to Qdrant client."""
        if self.conn_initialized is False:
            self._initialize_connection()
        else:
            self._start_connection()
        
    def close_connection(self):
        """Disconnect from Qdrant client."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Disconnected from Qdrant client")

    def is_pdf_indexed(self, pdf_hash: str) -> bool:
        """Check if a pdf with the given pdf hash is already indexed in Qdrant.
        
        Args:
            pdf_hash: Hash of the PDF to check for indexing.
            
        Returns:
            True if the pdf is already indexed, False otherwise.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call open_connection() first.")

        registry_id = QdrantManager._generate_point_id(pdf_hash, None)
        flt = qmodels.Filter(
            must=[
                qmodels.HasIdCondition(has_id=[registry_id]),
                qmodels.FieldCondition(
                    key="models",
                    match=qmodels.MatchValue(value=self.dense_model_name),
                ),
            ]
        )

        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.registry_collection,
                scroll_filter=flt,
                limit=1,
                with_payload=False,
                with_vectors=False,
                offset=next_offset,
            )

            if points:
                return True

            if not next_offset or not points:
                break

        return False
    
    def list_indexed_pdfs(self) -> list[dict[str, str]]:
        """List PDFs that have been indexed in Qdrant.
        
        Returns:
            List of dictionaries with 'title' and 'pdf_hash' keys.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call open_connection() first.")

        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="models",
                    match=qmodels.MatchValue(value=self.dense_model_name),
                )
            ]
        )

        indexed = []
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.registry_collection,
                scroll_filter=flt,
                limit=128,
                with_payload=["title", "pdf_hash"],
                with_vectors=False,
                offset=next_offset,
            )
            for point in points:
                payload = point.payload or {}
                indexed.append({
                    "title": payload.get("title", ""),
                    "pdf_hash": payload.get("pdf_hash", "")
                })

            if not next_offset or not points:
                break

        return indexed
    
    def upsert_chunks(self,
                        chunks: list[Chunk],
                        dense_embeddings: list[list[float]],
                        sparse_embeddings: list[dict[str, list[float]]],
                        contextual_texts: list[str] | None = None, #:FIXME: debug
                        progress_callback=None) -> tuple[int, list[dict[str, str]]]:
        """Upsert chunks into Qdrant chunk collection with hybrid vectors (dense + sparse).

        Args:
            chunks: List of Chunk objects to upsert.
            dense_embeddings: Dense vectors aligned by chunk index.
            sparse_embeddings: Sparse vectors aligned by chunk index.
            progress_callback: Function(stage, current, total, message) for progress updates.

        Returns:
            Tuple of (number of chunks upserted, title override info).
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call open_connection() first.")

        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        if len(dense_embeddings) != len(chunks):
            raise ValueError("Dense embeddings count does not match chunks count.")

        if len(sparse_embeddings) != len(chunks):
            raise ValueError("Sparse embeddings count does not match chunks count.")

        if contextual_texts is None:    #FIXME: debug
            contextual_texts = [p.text for p in chunks]
        elif len(contextual_texts) != len(chunks):
            raise ValueError("Contextual texts count does not match chunks count.")
        
        self.chunks = chunks

        points = []
        pdf_indexed_titles: dict[str, str] = {}
        for i, chunk in enumerate(self.chunks):
            point_id = QdrantManager._generate_point_id(chunk.pdf_hash, chunk.chunk_index)
            if chunk.pdf_hash not in pdf_indexed_titles:
                pdf_indexed_titles[chunk.pdf_hash] = chunk.title
                        
            vector_config = {
                "": dense_embeddings[i],
                "text-sparse": sparse_embeddings[i]
            }

            point = qmodels.PointStruct(
                id=point_id,
                vector=vector_config,
                payload={
                    'text': chunk.text,
                    'contextual_text': contextual_texts[i], #FIXME: debug
                    'page_number': chunk.page_number,
                    'chunk_index': chunk.chunk_index,
                    'title': chunk.title,
                    'pdf_hash': chunk.pdf_hash,
                    'section': chunk.section,
                    'sentence_count': chunk.sentence_count,
                    'sentences': chunk.sentences
                }
            )
            points.append(point)

        # Upsert points in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.chunk_collection,
                points=batch_points
            )
            if progress_callback:
                progress_callback('upserting', min(i + batch_size, len(points)), len(points), 
                                f"Upserted {min(i + batch_size, len(points))}/{len(points)} chunks...")
        logger.info(f"Upserted {len(points)} chunks into Qdrant chunk collection: {self.chunk_collection}")

        # Update the registry collection with indexed PDFs
        title_overrides: list[dict[str, str]] = []
        for pdf_hash, pdf_title in pdf_indexed_titles.items():
            canonical_title = self.register_pdf_title(pdf_hash, pdf_title)
            if canonical_title and canonical_title != pdf_title:
                title_overrides.append({
                    "input_title": pdf_title,
                    "indexed_title": canonical_title,
                    "pdf_hash": pdf_hash,
                })

        return len(points), title_overrides

    def register_pdf_title(self, pdf_hash: str, title: str) -> str | None:
        """Register or update the title for a given PDF hash in the registry collection.

        Args:
            pdf_hash: Hash of the PDF to register.
            title: Title to associate with the PDF hash.

        Returns:
            The canonical title for the PDF hash, or None if it could not be resolved.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call open_connection() first.")

        if not pdf_hash:
            raise ValueError("PDF hash is required to register title.")

        registry_id = QdrantManager._generate_point_id(pdf_hash, None)
        existing = self.client.retrieve(
            collection_name=self.registry_collection,
            ids=[registry_id],
            with_payload=True,
            with_vectors=False,
        )

        normalized_title = (title or "").strip()
        payload = (existing[0].payload or {}) if existing else {}
        current_title = (payload.get("title") or "").strip() or None

        if normalized_title and self._title_in_use_by_other_hash(normalized_title, pdf_hash):
            logger.warning(
                "Title '%s' already in use; skipping registry update for hash %s",
                normalized_title,
                pdf_hash,
            )
            return current_title

        if existing:
            models = payload.get("models") or []
            if isinstance(models, str):
                models = [models]
            if self.dense_model_name not in models:
                models.append(self.dense_model_name)

            primary_title = current_title or normalized_title
            update_payload = {
                "pdf_hash": pdf_hash,
                "models": models,
                "title": primary_title or "",
            }
            self.client.set_payload(
                collection_name=self.registry_collection,
                payload=update_payload,
                points=[registry_id],
            )
            return primary_title or None

        self.client.upsert(
            collection_name=self.registry_collection,
            points=[
                qmodels.PointStruct(
                    id=registry_id,
                    vector={},
                    payload={
                        "pdf_hash": pdf_hash,
                        "title": normalized_title,
                        "models": [self.dense_model_name],
                    },
                )
            ],
        )
        return normalized_title or None

    def _title_in_use_by_other_hash(self, title: str, pdf_hash: str) -> bool:
        """Check if a given title is already associated with a different PDF hash in the registry collection.
        
        Args:
            title: Title to check for usage.
            pdf_hash: PDF hash to exclude from the check (i.e., allow if the title is used by the same hash).
            
        Returns:
            True if the title is in use by a different PDF hash, False otherwise.
        """
        if not title:
            return False

        match = self.find_pdf_hash_by_title(title, filter_by_model=False)
        return bool(match and match != pdf_hash)

    def find_pdf_hash_by_title(self, title: str, filter_by_model: bool = True) -> str | None:
        """Return the pdf_hash for a given title.

        Args:
            title: Title to look up.
            filter_by_model: If True, only consider entries indexed by the current model.

        Returns:
            The pdf_hash associated with the title, or None if not found.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call open_connection() first.")

        if not title:
            return None

        must_conditions = [
            qmodels.FieldCondition(
                key="title",
                match=qmodels.MatchValue(value=title),
            )
        ]
        if filter_by_model:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="models",
                    match=qmodels.MatchValue(value=self.dense_model_name),
                )
            )

        flt = qmodels.Filter(must=must_conditions)

        points, _ = self.client.scroll(
            collection_name=self.registry_collection,
            scroll_filter=flt,
            limit=1,
            with_payload=["pdf_hash"],
            with_vectors=False,
        )

        if not points:
            return None

        payload = points[0].payload or {}
        return payload.get("pdf_hash")

    def delete_pdf_from_index(self, pdf_hash: str) -> dict[str, bool]:
        """Delete all chunks associated with a specific PDF hash from the Qdrant collection.
        
        Args:
            pdf_hash: Hash of the PDF whose chunks should be deleted.

        Returns:
            Dictionary indicating whether the entry was deleted and if it had other models.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call open_connection() first.")
        
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="pdf_hash",
                    match=qmodels.MatchValue(value=pdf_hash),
                )
            ]
        )

        delete_result = self.client.delete(
            collection_name=self.chunk_collection,
            points_selector=qmodels.FilterSelector(filter=flt)
        )

        if delete_result:
            res = self.delete_registry_entry(pdf_hash)
            if res["deleted"]:
                return res
        raise ValueError(f"Failed to delete PDF with hash '{pdf_hash}' from index.")
    
    def delete_registry_entry(self, pdf_hash: str) -> dict[str, bool]:
        """Delete the registry entry for a specific PDF hash from the registry collection.
        
        Args:
            pdf_hash: Hash of the PDF whose registry entry should be deleted.
        Returns:
            Dictionary indicating whether the entry was deleted and if it had other models.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call open_connection() first.")
        
        registry_id = QdrantManager._generate_point_id(pdf_hash, None)
        existing = self.client.retrieve(
            collection_name=self.registry_collection,
            ids=[registry_id],
            with_payload=True,
            with_vectors=False,
        )

        if existing:
            payload = existing[0].payload or {}
            models = payload.get("models") or []
            if isinstance(models, str):
                models = [models]

            models = [m for m in models if m != self.dense_model_name]

            if not models:

                self.client.delete(
                    collection_name=self.registry_collection,
                    points_selector=qmodels.PointIdsList(points=[registry_id]),
                )
                return {"deleted": True, "had_other_models": False}
            else:
                self.client.set_payload(
                    collection_name=self.registry_collection,
                    payload={"models": models},
                    points=[registry_id],
                )
            return {"deleted": True, "had_other_models": True}
        return {"deleted": False, "had_other_models": False}

    def clear_chunk_collection(self) -> dict[str, str]:
        """Clear all data from the Qdrant chunk collection.
        
        Returns:
            Dictionary of deleted PDF hashes and their titles.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call open_connection() first.")
        
        indexed_pdfs = self.list_indexed_pdfs()
        deleted_entries: dict[str, str] = {}
        for item in indexed_pdfs:
            pdf_hash = item.get("pdf_hash")
            if pdf_hash:
                res = self.delete_pdf_from_index(pdf_hash)
                if res["deleted"] and not res["had_other_models"]:
                    deleted_entries[pdf_hash] = item.get("title", "")

        self.client.delete_collection(self.chunk_collection)
        self.conn_initialized = False
        logger.info(f"Cleared Qdrant chunk collection: {self.chunk_collection}")
        return deleted_entries

    @staticmethod
    def _pdf_hash_filter(pdf_hashes: list[str] | None):
        """Restrict a search to the given documents, or None for the whole corpus."""
        if not pdf_hashes:
            return None
        return qmodels.Filter(must=[qmodels.FieldCondition(
            key="pdf_hash", match=qmodels.MatchAny(any=list(pdf_hashes)))])

    @staticmethod
    def _build_query_request(dense_query, sparse_query, threshold: float,
                            result_limit: int, mode: str,
                            pdf_hashes: list[str] | None = None) -> qmodels.QueryRequest:
        """One Qdrant query request for ``mode`` ('hybrid', 'dense' or 'sparse').

        'hybrid' RRF-fuses both prefetches; the single-vector modes query that
        vector directly, which is what makes retrieval_mode an ablation knob
        that genuinely reorders results rather than just filtering them.

        ``pdf_hashes`` scopes the search to those documents. In hybrid mode the
        filter goes on each prefetch, not only on the fused query: filtering
        after fusion would let both prefetches spend their whole ``limit`` on
        out-of-scope documents and fuse down to nothing.
        """
        flt = QdrantManager._pdf_hash_filter(pdf_hashes)
        if mode == "dense":
            return qmodels.QueryRequest(
                query=dense_query, using="", score_threshold=threshold,
                limit=result_limit, with_payload=True, filter=flt)
        if mode == "sparse":
            return qmodels.QueryRequest(
                query=sparse_query, using="text-sparse",
                limit=result_limit, with_payload=True, filter=flt)
        if mode != "hybrid":
            raise ValueError(f"unknown retrieval_mode: {mode!r}")
        return qmodels.QueryRequest(
            prefetch=[
                qmodels.Prefetch(
                    query=dense_query,
                    using="",
                    score_threshold=threshold,
                    limit=result_limit,
                    filter=flt,
                ),
                qmodels.Prefetch(
                    query=sparse_query,
                    using="text-sparse",
                    limit=result_limit,
                    filter=flt,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=result_limit,
            with_payload=True,
            filter=flt,
        )

    @staticmethod
    def _chunk_from_payload(payload: dict) -> Chunk:
        return Chunk(
            text=payload.get('text', ''),
            page_number=payload.get('page_number', -1),
            chunk_index=payload.get('chunk_index', -1),
            title=payload.get('title', ''),
            pdf_hash=payload.get('pdf_hash', ''),
            section=payload.get('section', ''),
            sentence_count=payload.get('sentence_count', 0),
            sentences=payload.get('sentences', [])
        )

    def fetch_chunks(self, ids: list[tuple[str, int]]) -> list[Chunk]:
        """Retrieve indexed chunks by (pdf_hash, chunk_index), skipping misses.

        Used by the oracle-context evaluation, which needs the gold chunks
        exactly as the reader would have received them from retrieval.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call open_connection() first.")
        if not ids:
            return []
        point_ids = [QdrantManager._generate_point_id(pdf_hash, idx) for pdf_hash, idx in ids]
        points = self.client.retrieve(
            collection_name=self.chunk_collection, ids=point_ids, with_payload=True)
        return [QdrantManager._chunk_from_payload(p.payload or {}) for p in points]

    def search_batch(self,
                    query_embeddings: list[dict[str, list[float]]],
                    threshold: float = 0.45,
                    result_limit: int = 30,
                    mode: str = "hybrid",
                    pdf_hashes: list[str] | None = None) -> list[list[Chunk]]:
        """Batch search for relevant chunks using Hybrid Search (Dense + Sparse BM25).

        Args:
            query_embeddings: List of query embedding dicts with 'dense' and 'sparse' keys.
            threshold: Score threshold for dense prefetch filtering.
            result_limit: Max results per query (the ``result_limit`` preset field).
            mode: 'hybrid', 'dense' or 'sparse' (the ``retrieval_mode`` preset field).
            pdf_hashes: Restrict the search to these documents (None = whole corpus).

        Returns:
            List of lists of (Chunk, score) tuples, aligned to query order.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call open_connection() first.")

        if not query_embeddings:
            return []

        requests = []
        for query_embedding in query_embeddings:
            dense_query = query_embedding.get("dense")
            sparse_query = query_embedding.get("sparse")
            if dense_query is None or sparse_query is None:
                raise ValueError("Each query embedding must include 'dense' and 'sparse' keys.")
            requests.append(QdrantManager._build_query_request(
                dense_query, sparse_query, threshold, result_limit, mode, pdf_hashes))

        search_results = self.client.query_batch_points(
            collection_name=self.chunk_collection,
            requests=requests,
        )

        if hasattr(search_results, "result"):
            search_results = search_results.result

        if not isinstance(search_results, list):
            search_results = [search_results]

        relevant_batches = []
        for response in search_results:
            relevant_chunks = []
            for result in response.points:
                chunk = QdrantManager._chunk_from_payload(result.payload or {})
                relevant_chunks.append((chunk, result.score))
            relevant_batches.append(relevant_chunks)

        return relevant_batches