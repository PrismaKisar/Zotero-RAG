"""QdrantManager class for managing Qdrant vector database operations."""

import re
import uuid
import logging
from typing import List, Optional, Dict
import qdrant_client as qc
from qdrant_client.http import models as qmodels

from models import Paragraph

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manage Qdrant vector database for storing and searching paragraph vectors."""
    
    def __init__(self, dense_model_name: str = "BAAI/bge-base-en-v1.5", 
                 qdrant_url: str = "http://localhost:6333",
                 vector_size: int = 768):
        """Initialize the Qdrant manager.
        
        Args:
            dense_model_name: Name used to derive a deterministic collection name.
            vector_size: Dimensionality of the dense vectors stored in Qdrant.
        """
        self.dense_model_name = dense_model_name
        self.qdrant_url = qdrant_url
        self.qdrant_collection = "zoteroRAG_" + self._sanitize_model_name(dense_model_name)
        self.vector_size = vector_size
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
        
        if not self.client.collection_exists(self.qdrant_collection):
            self.client.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
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
    
    def upsert_paragraphs(self,
                        paragraphs: List[Paragraph],
                        dense_embeddings: List[List[float]],
                        sparse_embeddings: List[Dict[str, List[float]]],
                        progress_callback=None) -> int:
        """Upsert paragraphs into Qdrant collection with hybrid vectors (dense + sparse).

        Args:
            paragraphs: List of Paragraph objects to upsert.
            dense_embeddings: Dense vectors aligned by paragraph index.
            sparse_embeddings: Sparse vectors aligned by paragraph index.
            progress_callback: Function(stage, current, total, message) for progress updates.

        Returns:
            Number of paragraphs upserted.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call initialize_connection() first.")

        if not paragraphs:
            raise ValueError("No paragraphs provided for indexing.")

        if len(dense_embeddings) != len(paragraphs):
            raise ValueError("Dense embeddings count does not match paragraphs count.")

        if len(sparse_embeddings) != len(paragraphs):
            raise ValueError("Sparse embeddings count does not match paragraphs count.")
        
        self.paragraphs = paragraphs

        points = []
        for i, para in enumerate(self.paragraphs):
            point_id = self.generate_point_id(para.pdf_path, para.para_idx)
            
            vector_config = {
                "": dense_embeddings[i],
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

    def search(self,
               query_dense_embedding: List[float],
               query_sparse_embedding: Dict[str, List[float]],
               threshold: float = 0.7) -> List[tuple]:
        """Search for relevant paragraphs using Hybrid Search (Dense + Sparse BM25).
            
        Args:
            query_dense_embedding: Dense query embedding.
            query_sparse_embedding: Sparse query embedding with indices/values keys.
            limit: Number of results to return.

        Returns:
            List of tuples (Paragraph, score) for relevant paragraphs.
        """
        if not self.client:
            raise ValueError("Qdrant client is not connected. Call initialize_connection() first.")

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
                    query=query_sparse_embedding,
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