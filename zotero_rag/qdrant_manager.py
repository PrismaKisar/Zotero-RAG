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
                Numpy array of embeddings.
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
        embeddings_list = []
        for i in range(0, len(all_texts), effective_batch_size):
            batch = all_texts[i:i + effective_batch_size]
            try:
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        batch_size=effective_batch_size,
                        device=self.device
                    )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # If we still hit OOM, reduce batch size further
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in 
                      ["out of memory", "buffer size", "mps", "cuda", "memory"]):
                    fallback_size = max(1, effective_batch_size // 2)
                    if progress_callback:
                        progress_callback('encoding', i, len(all_texts), 
                                        f"Reducing batch size to {fallback_size}...")
                    with torch.no_grad():
                        batch_embeddings = self.model.encode(
                            batch,
                            show_progress_bar=False,
                            batch_size=fallback_size,
                            device=self.device
                        )
                else:
                    raise
            
            embeddings_list.append(batch_embeddings)
            
            # Update progress after each batch
            processed = min(i + effective_batch_size, len(all_texts))
            if progress_callback:
                progress_callback('encoding', processed, len(all_texts), 
                                f"Encoded {processed}/{len(all_texts)} chunks...")
        
        return np.vstack(embeddings_list)

    def upsert_paragraphs(self, paragraphs: List[Paragraph], 
                        progress_callback=None) -> int:
        """Upsert paragraphs into Qdrant collection.

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

        embeddings = self.encode_paragraphs(progress_callback, all_texts)
        points = []
        for (para, embedding) in zip(self.paragraphs, embeddings):
            point_id = self.generate_point_id(para.pdf_path, para.para_idx)
            point = qmodels.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
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
        """Search for relevant paragraphs based on a query string.
        
        Args:
            query: The search query string.
            threshold: Distance threshold for filtering results (lower is more similar).
        Returns:
            List of tuples (Paragraph, distance) for relevant paragraphs.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call initialize_connection() first.")
        
        query_embedding = self.model.encode(query, device=self.device)
        
        search_result = self.client.query_points(
            collection_name=self.qdrant_collection,
            query=query_embedding.tolist(),
            limit=20,
            score_threshold=threshold,
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