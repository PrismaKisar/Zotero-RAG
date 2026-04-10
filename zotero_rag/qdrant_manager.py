"""QdrantManager class for managing Qdrant vector database operations, including collection management, encoding, and upserting paragraphs."""

import re
import uuid
import logging
from typing import List, Optional, Dict
import numpy as np
import torch
import qdrant_client as qc
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from models import Paragraph

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manage Qdrant vector database for storing and searching paragraph embeddings."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", 
                 qdrant_url: str = "http://localhost:6333",
                 device: str = None,
                 encode_batch_size: int = 8):
        """Initialize the Qdrant manager.
        
        Args:
            model_name: Name of the sentence transformer model.
            device: Device to use for encoding ('cpu', 'cuda', 'mps'). Auto-detect if None.
            encode_batch_size: Batch size for encoding.
        """
        self.model_name = model_name
        self.qdrant_url = qdrant_url
        self.qdrant_collection = "zoteroRAG_" + self._sanitize_model_name(model_name)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.encode_batch_size = encode_batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.paragraphs: List[Paragraph] = []
        self.client: Optional[qc.QdrantClient] = None
    
    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        """Convert model name to safe filename component."""
        model_short = model_name.split('/')[-1]
        return re.sub(r'[^a-zA-Z0-9_-]', '_', model_short)
    
    @staticmethod
    def generate_point_id(file_hash: str, paragraph_index) -> str:
        """Generate a unique point ID for Qdrant."""
        input_str = f"{file_hash}_{paragraph_index}"
        NAMESPACE_RAG = uuid.UUID("12345678-1234-5678-1234-567812345678")
        return str(uuid.uuid5(NAMESPACE_RAG, input_str))
 
    def initialize_connection(self):
        """Connect to Qdrant client and ensure collection exists."""
        self.client = qc.QdrantClient(
            url=self.qdrant_url,
        )

        if not self.client:
            raise ValueError("Qdrant client is not connected. Call initialize_connection() first.")
        
        logger.info("Connected to Qdrant client")
        
        vector_size = self.model.get_sentence_embedding_dimension()

        if not self.client.collection_exists(self.qdrant_collection):
            self.client.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                    datatype=qmodels.Datatype.FLOAT16 #TODO: valutare la quantizzazione utilizzando uint8
                ),
                sparse_vectors_config={
                    "text-sparse": qmodels.SparseVectorParams(
                        index=qmodels.SparseIndexParams(
                            on_disk=True,
                        )
                    )
                }
            )

            # Create an index on the 'pdf_hash' payload field for efficient lookups
            self.client.create_payload_index(
                collection_name=self.qdrant_collection,
                field_name="pdf_hash",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

            logger.info(f"Created Qdrant collection: {self.qdrant_collection}")
        else:
            logger.info(f"Qdrant collection already exists: {self.qdrant_collection}")

    def close_connection(self):
        """Disconnect from Qdrant client."""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Disconnected from Qdrant client")

    def _find_safe_batch_size(self, sample_texts: List[str], 
                              start_size: int = 2, 
                              max_size: int = 128,
                              target_memory_fraction: float = 0.75) -> int:
        """Find safe batch size targeting specific memory usage.
        
        Args:
            sample_texts: Sample of texts to test encoding with.
            start_size: Initial batch size to try.
            max_size: Maximum batch size to test.
            target_memory_fraction: Target fraction of memory to use (0.0-1.0).
            
        Returns:
            Safe batch size targeting the memory fraction.
        """
        if not sample_texts:
            return start_size
        
        # Sample a small set to test with
        test_sample = sample_texts[:min(100, len(sample_texts))]
        
        current_size = start_size
        last_safe_size = start_size
        
        while current_size <= max_size:
            try:
                with torch.no_grad():
                    _ = self.model.encode(
                        test_sample,
                        batch_size=current_size,
                        device=self.device,
                        show_progress_bar=False
                    )
                last_safe_size = current_size
                # Scale up more aggressively to find limit
                current_size = int(current_size * 1.5)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in 
                      ["out of memory", "buffer size", "mps", "cuda", "memory"]):
                    # Hit OOM, scale back to target fraction
                    return max(start_size, int(last_safe_size * target_memory_fraction))
                else:
                    return last_safe_size
            except Exception:
                return last_safe_size
        
        # Hit max size without OOM, use target fraction of max
        return max(start_size, int(last_safe_size * target_memory_fraction))
    
    def is_pdf_indexed(self, pdf_hash: str) -> bool:
        """Check if a pdf with the given pdf file hash is already indexed in Qdrant.
        
        Args:
            file_hash: Hash of the pdf file to check.
            
        Returns:
            True if the pdf is already indexed, False otherwise.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call initialize_connection() first.")
        
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="pdf_hash",
                    match=qmodels.MatchValue(value=pdf_hash),
                )
            ]
        )

        # Search for any point with a payload containing the file hash
        points, _ = self.client.scroll(
            collection_name=self.qdrant_collection,
            scroll_filter=flt,
            limit=1,
            with_payload=False,
            with_vectors=False
        )
        return len(points) > 0
    
    #TODO: se si vuole fare bisogna pensarci bene al modo più efficiente / sistemare il magic number 1000
    def list_indexed_pdfs(self) -> List[Dict]:
        """List PDFs that have been indexed in Qdrant.
        
        Returns:
            List of dictionaries with 'pdf_path', 'title', 'pdf_hash' keys.
        """
        raise NotImplementedError("Listing indexed PDFs is not implemented yet.")
        # if not self.client:
        #     raise ValueError("Qdrant client is not connected. Call initialize_connection() first.")
        
        # results = self.client.scroll(
        #     collection_name=self.qdrant_collection,
        #     limit=1000,
        #     with_payload=["title", "pdf_path", "pdf_hash"],
        #     with_vectors=False,
        #     group_by="pdf_hash"
        # )
        
        # indexed_pdfs = [
        #     {
        #         "title": group.hits[0].payload.get("title", "Unknown_Title"),
        #         "pdf_path": group.hits[0].payload.get("pdf_path", "Unknown_Path"),
        #         "hash": group.id,
        #     }
        #     for group in results[0].groups
        # ]
    
        #return indexed_pdfs
    
    def encode_paragraphs(self, progress_callback, all_texts) -> np.ndarray:
        """Encode paragraphs into embeddings with dynamic batch size and progress updates.
        
            Args:
                progress_callback: Function(stage, current, total, message) for progress updates.
                all_texts: List of paragraph texts to encode.

            Returns:
                Dict containing:
                    'dense': np.ndarray of dense embeddings
                    'sparse': List of dicts {'indices': [...], 'values': [...]}
        """
        if not self.model:
            raise ValueError("Model is not loaded. Cannot encode paragraphs.")

        if self.encode_batch_size is None or self.encode_batch_size == 0:
            # Auto-detect safe batch size
            if progress_callback:
                progress_callback('encoding', 0, len(all_texts), "Auto-detecting safe batch size...")
            effective_batch_size = self._find_safe_batch_size(all_texts, start_size=2, max_size=128)
            logger.info(f"Auto-detected encoding batch size: {effective_batch_size}")
        else:
            effective_batch_size = self.encode_batch_size
        
        if progress_callback:
            progress_callback('encoding', 0, len(all_texts), 
                            f"Encoding with batch size {effective_batch_size}...")
        
        # Manually batch and encode to show progress
        dense_embeddings_list = []
        sparse_embeddings_list = []

        for i in range(0, len(all_texts), effective_batch_size):
            batch = all_texts[i:i + effective_batch_size]
            try:
                # 1. GENERAZIONE VETTORI DENSI
                with torch.no_grad():
                    batch_dense = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        batch_size=effective_batch_size,
                        device=self.device
                    )
                
                # 2. GENERAZIONE VETTORI SPARSI
                batch_sparse_gen = self.sparse_model.embed(batch)
                batch_sparse = [
                    {"indices": res.indices.tolist(), "values": res.values.tolist()} 
                    for res in batch_sparse_gen
                ]
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # If we still hit OOM, reduce batch size further
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in 
                      ["out of memory", "buffer size", "mps", "cuda", "memory"]):
                    fallback_size = max(1, effective_batch_size // 2)
                    if progress_callback:
                        progress_callback('encoding', i, len(all_texts), f"OOM: Reducing batch size to {fallback_size}...")
                
                    with torch.no_grad():
                        batch_dense = self.model.encode(batch, show_progress_bar=False, batch_size=fallback_size, device=self.device)
                
                    batch_sparse_gen = self.sparse_model.embed(batch)
                    batch_sparse = [{"indices": res.indices.tolist(), "values": res.values.tolist()} for res in batch_sparse_gen]
                else:
                    raise

            dense_embeddings_list.append(batch_dense)
            sparse_embeddings_list.extend(batch_sparse)

            # Update progress after each batch
            processed = min(i + effective_batch_size, len(all_texts))
            if progress_callback:
                progress_callback('encoding', processed, len(all_texts), 
                                f"Encoded {processed}/{len(all_texts)} chunks...")
        
        return {
            "dense": np.vstack(dense_embeddings_list),
            "sparse": sparse_embeddings_list
        }

    def upsert_paragraphs(self, paragraphs: List[Paragraph], 
                        progress_callback=None) -> int:
        """Upsert paragraphs into Qdrant collection with hybrid vectors (dense + sparse).

        Args:
            paragraphs: List of Paragraph objects to upsert.
            progress_callback: Function(stage, current, total, message) for progress updates.

        Returns:
            Number of paragraphs upserted.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call initialize_connection() first.")

        if not paragraphs:
            raise ValueError("No paragraphs provided for indexing.")
        
        self.paragraphs = paragraphs
        all_texts = [p.text for p in paragraphs]

        hybrid_embeddings = self.encode_paragraphs(progress_callback, all_texts)
        
        dense_embeddings = hybrid_embeddings["dense"]
        sparse_embeddings = hybrid_embeddings["sparse"]

        points = []
        for i, para in enumerate(self.paragraphs):
            point_id = self.generate_point_id(para.pdf_path, para.para_idx)
            
            vector_config = {
                "": dense_embeddings[i].tolist(),
                "text-sparse": sparse_embeddings[i]
            }

            point = qmodels.PointStruct(
                id=point_id,
                vector=vector_config,
                payload={
                    'text': para.text,
                    'pdf_path': para.pdf_path,
                    'page_num': para.page_num,
                    'para_idx': para.para_idx,
                    'item_key': para.item_key,
                    'pdf_hash': para.pdf_hash,
                    'title': para.title,
                    'section': para.section,
                    'sentence_count': para.sentence_count,
                    'sentences': para.sentences
                }
            )
            points.append(point)

        # Upsert points in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.qdrant_collection,
                points=batch_points
            )
            if progress_callback:
                progress_callback('upserting', min(i + batch_size, len(points)), len(points), 
                                f"Upserted {min(i + batch_size, len(points))}/{len(points)} paragraphs...")
        logger.info(f"Upserted {len(points)} paragraphs into Qdrant collection: {self.qdrant_collection}")
        return len(points)
    
    def delete_pdf_from_index(self, pdf_hash: str):
        """Delete all paragraphs associated with a specific PDF hash from the Qdrant collection.
        
        Args:
            pdf_hash: Hash of the PDF whose paragraphs should be deleted.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call initialize_connection() first.")
        
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="pdf_hash",
                    match=qmodels.MatchValue(value=pdf_hash),
                )
            ]
        )

        bool = self.client.delete(
            collection_name=self.qdrant_collection,
            points_selector=qmodels.FilterSelector(filter=flt)
        )

        return bool

    def clear_collection(self):
        """Clear all data from the Qdrant collection."""
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call initialize_connection() first.")
        
        self.client.delete_collection(self.qdrant_collection)
        logger.info(f"Cleared Qdrant collection: {self.qdrant_collection}")

    def search(self, query: str, threshold: float = 0.7) -> List[tuple]:
        """Search for relevant paragraphs using Hybrid Search (Dense + Sparse BM25).
            
        Args:
            query: The search query string.
            limit: Number of results to return.

        Returns:
            List of tuples (Paragraph, score) for relevant paragraphs.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call initialize_connection() first.")
        
        query_dense_embedding = self.model.encode(query, device=self.device).tolist()
        query_sparse_gen = self.sparse_model.embed([query])
        query_sparse_obj = next(query_sparse_gen)
        query_sparse_dict = {
            "indices": query_sparse_obj.indices.tolist(),
            "values": query_sparse_obj.values.tolist()
        }

        search_result = self.client.query_points(
            collection_name=self.qdrant_collection,
            prefetch=[
                qmodels.Prefetch(
                    query=query_dense_embedding,
                    using="",
                    score_threshold=threshold,
                    limit=20
                ),
                qmodels.Prefetch(
                    query=query_sparse_dict,
                    using="text-sparse", #TODO: valutare se mettere un threshold anche quí, si potrebbe mettere 0.01 per eliminare 
                    limit=20             #      i risultati che non matchano nemmeno con BM25... capiamo
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=20,
            with_payload=True
        )
        
        relevant_paragraphs = []
        for result in search_result.points:
            payload = result.payload

            para = Paragraph(
                text=payload.get('text', ''),
                pdf_path=payload.get('pdf_path', ''),
                page_num=payload.get('page_num', -1),
                para_idx=payload.get('para_idx', -1),
                item_key=payload.get('item_key', ''),
                pdf_hash=payload.get('pdf_hash', ''),
                title=payload.get('title', ''),
                section=payload.get('section', ''),
                sentence_count=payload.get('sentence_count', 0),
                sentences=payload.get('sentences', [])
            )
            relevant_paragraphs.append((para, result.score))

        return relevant_paragraphs