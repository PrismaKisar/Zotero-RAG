"""QdrantManager class for managing Qdrant vector database operations."""

import functools
import logging
import re
import uuid

import httpx
import qdrant_client as qc
from models import Chunk
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException

logger = logging.getLogger(__name__)

# Qdrant's own RRF constant, kept so client-side fusion ranks exactly as it did.
RRF_K = 2


def _reports_unreachable(method):
    """Raise a Qdrant that went away mid-session as the ConnectionError a failed
    connect raises, so the app's "Qdrant is not running" handling still applies
    to a client that stays open."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except ResponseHandlingException as exc:
            if isinstance(exc.source, httpx.TransportError):
                raise self._unreachable() from exc
            raise
    return wrapper


class QdrantManager:
    """Manage Qdrant vector database for storing and searching chunk vectors."""
    
    def __init__(self, 
                dense_model_name: str,
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
        self._client: qc.QdrantClient | None = None
        self._collections_ready = False
    
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

    def _unreachable(self) -> ConnectionError:
        message = (
            f"Qdrant service not reachable at {self.qdrant_url}. "
            "Start Qdrant and retry."
        )
        logger.error(message)
        return ConnectionError(message)

    def _connect(self) -> qc.QdrantClient:
        """A client that has completed one round-trip, or ConnectionError."""
        client = qc.QdrantClient(url=self.qdrant_url)
        try:
            # Force a round-trip to verify the service is reachable.
            client.get_collections()
        except Exception as exc:
            try:
                client.close()
            except Exception:
                logger.debug("Error closing unreachable Qdrant client", exc_info=True)
            raise self._unreachable() from exc
        logger.info("Connected to Qdrant client")
        return client

    @property
    def client(self) -> qc.QdrantClient:
        """The connected client, with both collections in place.

        Connects on first use and stays connected for the manager's lifetime,
        so callers never sequence an open and a close. A failed connection is
        not cached: the next use tries again.
        """
        if self._client is None:
            self._client = self._connect()
        if not self._collections_ready:
            self._ensure_collections()
        return self._client

    @_reports_unreachable
    def ping(self):
        """Raise ConnectionError now if Qdrant is unreachable.

        For a batch that records per-item failures and moves on: without it, a
        Qdrant that is down is recorded as every item failing instead of as
        the one error it is.
        """
        self.client.get_collections()

    def _ensure_collections(self):
        """Create the chunk and registry collections, and their indexes, if missing."""
        client = self._client
        if not client.collection_exists(self.chunk_collection):
            client.create_collection(
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
            client.create_payload_index(
                collection_name=self.chunk_collection,
                field_name="pdf_hash",
                field_schema=qmodels.KeywordIndexParams(
                    type="keyword",
                    # ponytail: search(pdf_hashes=...) *is* a filtered vector
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

        if not client.collection_exists(self.registry_collection):
            client.create_collection(
                collection_name=self.registry_collection,
                vectors_config=None,
            )

            # Create an index on 'models' payload field for efficient lookups
            client.create_payload_index(
                collection_name=self.registry_collection,
                field_name="models",
                field_schema=qmodels.KeywordIndexParams(
                    type="keyword",
                    enable_hnsw=False,
                ),
            )
            # Create an index on title to allow lookups by name
            client.create_payload_index(
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

        self._collections_ready = True

    @_reports_unreachable
    def is_pdf_indexed(self, pdf_hash: str) -> bool:
        """Check if a pdf with the given pdf hash is already indexed in Qdrant.
        
        Args:
            pdf_hash: Hash of the PDF to check for indexing.
            
        Returns:
            True if the pdf is already indexed, False otherwise.
        """
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

        points, _ = self.client.scroll(
            collection_name=self.registry_collection,
            scroll_filter=flt,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)
    
    @_reports_unreachable
    def list_indexed_pdfs(self) -> list[dict[str, str]]:
        """List PDFs that have been indexed in Qdrant.
        
        Returns:
            List of dictionaries with 'title' and 'pdf_hash' keys.
        """
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
    
    @_reports_unreachable
    def upsert_chunks(self,
                        chunks: list[Chunk],
                        dense_embeddings: list[list[float]],
                        sparse_embeddings: list[dict[str, list[float]]],
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
        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        if len(dense_embeddings) != len(chunks):
            raise ValueError("Dense embeddings count does not match chunks count.")

        if len(sparse_embeddings) != len(chunks):
            raise ValueError("Sparse embeddings count does not match chunks count.")

        points = []
        pdf_indexed_titles: dict[str, str] = {}
        for i, chunk in enumerate(chunks):
            point_id = QdrantManager._generate_point_id(chunk.pdf_hash, chunk.chunk_index)
            pdf_indexed_titles.setdefault(chunk.pdf_hash, chunk.title)

            vector_config = {
                "": dense_embeddings[i],
                "text-sparse": sparse_embeddings[i]
            }

            point = qmodels.PointStruct(
                id=point_id,
                vector=vector_config,
                payload={
                    'text': chunk.text,
                    'page_number': chunk.page_number,
                    'chunk_index': chunk.chunk_index,
                    'title': chunk.title,
                    'pdf_hash': chunk.pdf_hash,
                    'section': chunk.section,
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

    @_reports_unreachable
    def register_pdf_title(self, pdf_hash: str, title: str) -> str | None:
        """Register or update the title for a given PDF hash in the registry collection.

        Args:
            pdf_hash: Hash of the PDF to register.
            title: Title to associate with the PDF hash.

        Returns:
            The canonical title for the PDF hash, or None if it could not be resolved.
        """
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

        owner = self.find_pdf_hash_by_title(normalized_title, filter_by_model=False)
        if owner and owner != pdf_hash:
            logger.warning(
                "Title '%s' already in use; skipping registry update for hash %s",
                normalized_title,
                pdf_hash,
            )
            return current_title

        if existing:
            models = payload.get("models") or []
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

    @_reports_unreachable
    def find_pdf_hash_by_title(self, title: str, filter_by_model: bool = True) -> str | None:
        """Return the pdf_hash for a given title.

        Args:
            title: Title to look up.
            filter_by_model: If True, only consider entries indexed by the current model.

        Returns:
            The pdf_hash associated with the title, or None if not found.
        """
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

    @_reports_unreachable
    def delete_pdf_from_index(self, pdf_hash: str) -> bool:
        """Delete a PDF's chunks and drop this model from its registry entry.

        Args:
            pdf_hash: Hash of the PDF whose chunks should be deleted.

        Returns:
            True if another model still indexes the PDF, so its cached files must stay.

        Raises:
            ValueError: If the PDF has no registry entry.
        """
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="pdf_hash",
                    match=qmodels.MatchValue(value=pdf_hash),
                )
            ]
        )
        self.client.delete(
            collection_name=self.chunk_collection,
            points_selector=qmodels.FilterSelector(filter=flt)
        )

        registry_id = QdrantManager._generate_point_id(pdf_hash, None)
        existing = self.client.retrieve(
            collection_name=self.registry_collection,
            ids=[registry_id],
            with_payload=True,
            with_vectors=False,
        )
        if not existing:
            raise ValueError(f"Failed to delete PDF with hash '{pdf_hash}' from index.")

        models = [m for m in (existing[0].payload or {}).get("models") or []
                  if m != self.dense_model_name]
        if models:
            self.client.set_payload(
                collection_name=self.registry_collection,
                payload={"models": models},
                points=[registry_id],
            )
        else:
            self.client.delete(
                collection_name=self.registry_collection,
                points_selector=qmodels.PointIdsList(points=[registry_id]),
            )
        return bool(models)

    @_reports_unreachable
    def clear_chunk_collection(self) -> dict[str, str]:
        """Clear all data from the Qdrant chunk collection.
        
        Returns:
            Dictionary of deleted PDF hashes and their titles.
        """
        indexed_pdfs = self.list_indexed_pdfs()
        deleted_entries: dict[str, str] = {}
        for item in indexed_pdfs:
            pdf_hash = item.get("pdf_hash")
            if pdf_hash and not self.delete_pdf_from_index(pdf_hash):
                deleted_entries[pdf_hash] = item.get("title", "")

        self.client.delete_collection(self.chunk_collection)
        self._collections_ready = False
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
    def _build_query_requests(dense_query, sparse_query, result_limit: int, mode: str,
                              pdf_hashes: list[str] | None = None) -> list[qmodels.QueryRequest]:
        """The Qdrant requests for ``mode``: one branch, or two to fuse for 'hybrid'.

        Hybrid is fused here rather than by Qdrant (see ``rrf_fuse``), so it is
        two plain requests, one per vector. Each branch asks for twice
        ``result_limit``: the extra depth is what lets ``rank_branch`` decide
        ties at the cut by chunk id instead of taking whichever equal-scored
        points Qdrant returned first.

        ``pdf_hashes`` scopes every branch to those documents, so neither branch
        can spend its depth on out-of-scope chunks.

        No request carries a ``score_threshold``: ``result_limit`` is the only
        cut. A cosine floor used to sit on the dense side and was removed - see
        settings.py for the measurement behind that.

        The dense side is searched exactly, not through HNSW. At benchmark
        scale this changes nothing - below ``indexing_threshold_kb`` (10 MB of
        vectors by default, ~6.7k chunks here) Qdrant builds no HNSW graph and
        scans anyway - but a real library can outgrow that, and with no
        threshold the retrieval stage is its ordering and nothing else. The
        cost is linear in the chunk count: milliseconds up to ~1e5 chunks. The
        sparse side needs no flag, its inverted index is exact already.
        """
        if mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"unknown retrieval_mode: {mode!r}")
        flt = QdrantManager._pdf_hash_filter(pdf_hashes)
        # ponytail: twice the limit settles ties unless more than result_limit
        # points share the score at the cut, which a BM25 tail could; deepen then.
        depth = 2 * result_limit
        requests = []
        if mode in ("dense", "hybrid"):
            requests.append(qmodels.QueryRequest(
                query=dense_query, using="", limit=depth,
                params=qmodels.SearchParams(exact=True), with_payload=True, filter=flt))
        if mode in ("sparse", "hybrid"):
            requests.append(qmodels.QueryRequest(
                query=sparse_query, using="text-sparse", limit=depth,
                with_payload=True, filter=flt))
        return requests

    @staticmethod
    def rank_branch(hits: list[tuple[Chunk, float]], limit: int) -> list[tuple[Chunk, float]]:
        """The top ``limit`` of one branch, best first, ties broken by chunk id."""
        return sorted(hits, key=lambda h: (-h[1], h[0].pdf_hash, h[0].chunk_index))[:limit]

    @staticmethod
    def rrf_fuse(branches: list[list[tuple[Chunk, float]]], limit: int,
                 k: int = RRF_K) -> list[tuple[Chunk, float]]:
        """Reciprocal-rank fusion of ranked branches, as Qdrant computes it.

        A chunk scores the sum over branches of ``1 / (k + rank)``, rank from
        zero, which is Qdrant's RRF with its default k=2 (checked against the
        stored scores: 1.0 for a chunk first in both branches). Done here so
        the fused ties are broken by chunk id: Qdrant returns equal fused scores
        in no stable order, and that decided who entered the top ``limit`` on
        ~15% of questions between otherwise identical runs.
        """
        fused: dict[tuple[str, int], list] = {}
        for branch in branches:
            for rank, (chunk, _) in enumerate(branch):
                entry = fused.setdefault((chunk.pdf_hash, chunk.chunk_index), [chunk, 0.0])
                entry[1] += 1.0 / (k + rank)
        return QdrantManager.rank_branch([(c, s) for c, s in fused.values()], limit)

    @staticmethod
    def _chunk_from_payload(payload: dict) -> Chunk:
        return Chunk(
            text=payload.get('text', ''),
            page_number=payload.get('page_number', -1),
            chunk_index=payload.get('chunk_index', -1),
            title=payload.get('title', ''),
            pdf_hash=payload.get('pdf_hash', ''),
            section=payload.get('section', ''),
            sentences=payload.get('sentences', [])
        )

    @_reports_unreachable
    def fetch_chunks(self, ids: list[tuple[str, int]]) -> list[Chunk]:
        """Retrieve indexed chunks by (pdf_hash, chunk_index), skipping misses.

        Used by the oracle-context evaluation, which needs the gold chunks
        exactly as the reader would have received them from retrieval.
        """
        if not ids:
            return []
        point_ids = [QdrantManager._generate_point_id(pdf_hash, idx) for pdf_hash, idx in ids]
        points = self.client.retrieve(
            collection_name=self.chunk_collection, ids=point_ids, with_payload=True)
        return [QdrantManager._chunk_from_payload(p.payload or {}) for p in points]

    @_reports_unreachable
    def search(self,
               query_embedding: dict[str, list[float]],
               result_limit: int = 30,
               mode: str = "hybrid",
               pdf_hashes: list[str] | None = None) -> list[tuple[Chunk, float]]:
        """Search for relevant chunks using Hybrid Search (Dense + Sparse BM25).

        Args:
            query_embedding: Query embedding dict with 'dense' and 'sparse' keys.
            result_limit: Max results (the ``result_limit`` preset field).
            mode: 'hybrid', 'dense' or 'sparse' (the ``retrieval_mode`` preset field).
            pdf_hashes: Restrict the search to these documents (None = whole corpus).

        Returns:
            (Chunk, score) pairs, best first.
        """
        requests = QdrantManager._build_query_requests(
            query_embedding["dense"], query_embedding["sparse"], result_limit, mode, pdf_hashes)
        responses = self.client.query_batch_points(
            collection_name=self.chunk_collection, requests=requests)
        branches = [QdrantManager.rank_branch(
            [(QdrantManager._chunk_from_payload(p.payload or {}), p.score)
             for p in response.points], result_limit)
            for response in responses]
        if len(branches) == 1:
            return branches[0]
        return QdrantManager.rrf_fuse(branches, result_limit)