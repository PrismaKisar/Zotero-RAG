"""Embedding manager for dense and sparse vector generation."""

import numpy as np
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from zotero_rag.device import resolve_device

DEFAULT_ENCODE_BATCH_SIZE = 32
# The default, not a fixed setting: the user may index with another model, and
# the Qdrant collection is named after it, so a second model builds a second
# index instead of corrupting the first.
DEFAULT_DENSE_MODEL = "google/embeddinggemma-300m"
# The reranker reads 512 tokens, and the dense comparison (benchmark/studies/retrieval_models.py)
# measured every model at this window, so the index embeds what the reranker sees.
MAX_TOKENS = 512


def _sparse(embedding) -> dict[str, list]:
    """A FastEmbed sparse embedding in the shape Qdrant takes."""
    return {"indices": embedding.indices.tolist(), "values": embedding.values.tolist()}


class EmbeddingManager:
    """Generate dense and sparse embeddings for indexing and retrieval."""

    def __init__(self,
                dense_model_name: str = DEFAULT_DENSE_MODEL,
                device: str | None = None,
                encode_batch_size: int | None = None):
        """Initialize embedding models and runtime options.

        Args:
            dense_model_name: sentence-transformers dense model name.
            device: Device for dense model ('cpu', 'cuda', 'mps'). Auto-detect if None.
            encode_batch_size: Batch size for chunk encoding. Defaults to
                ``DEFAULT_ENCODE_BATCH_SIZE`` when None or 0.
        """
        self.dense_model_name = dense_model_name
        self.sparse_model_name = "Qdrant/bm25"

        self.device = resolve_device(device)
        self.encode_batch_size = encode_batch_size or DEFAULT_ENCODE_BATCH_SIZE

        # Dense runs on torch, so it gets the accelerator: on an M4, EmbeddingGemma on
        # MPS encodes QASA at 29 chunks/s against 16 for the old ONNX bge on CPU. The
        # model's query/document prompts are applied by encode_query/encode_document.
        # BM25 is a tokenizer plus IDF, so FastEmbed's CPU path is all it needs.
        self.dense_model = SentenceTransformer(self.dense_model_name, device=self.device)
        self.dense_model.max_seq_length = MAX_TOKENS
        self.sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name,
                                                cuda=self.device == "cuda")

    @property
    def vector_size(self) -> int:
        """Return the dense embedding dimensionality."""
        return self.dense_model.get_embedding_dimension()

    def encode_chunks(self, all_texts: list[str],
                      progress_callback=None) -> dict[str, list[float]]:
        """Encode chunks into dense+sparse vectors with progress updates.
        
        Args:
            all_texts: List of chunk texts to encode.
            progress_callback: Function to call with progress updates (stage, current, total, message).
        
        Returns:
            Dictionary with 'dense' and 'sparse' keys containing the respective embeddings.
        """
        if not all_texts:
            return {"dense": [], "sparse": []}

        progress = progress_callback or (lambda *_: None)
        progress("encoding", 0, len(all_texts),
                 f"Encoding with batch size {self.encode_batch_size}...")

        dense_embeddings_list = []
        sparse_embeddings_list = []

        for i in range(0, len(all_texts), self.encode_batch_size):
            batch = all_texts[i : i + self.encode_batch_size]
            dense_embeddings_list.append(
                self.dense_model.encode_document(batch, batch_size=len(batch)))
            sparse_embeddings_list.extend(_sparse(res) for res in self.sparse_model.embed(batch))

            processed = i + len(batch)
            progress("encoding", processed, len(all_texts),
                     f"Encoded {processed}/{len(all_texts)} chunks...")

        return {
            "dense": np.vstack(dense_embeddings_list).tolist(),
            "sparse": sparse_embeddings_list,
        }

    def encode_query(self, query: str) -> dict[str, list[float]]:
        """Encode a single query into dense+sparse vectors.
        
        Args:
            query: The query string to encode.
        
        Returns:
            Dictionary with 'dense' and 'sparse' keys containing the respective embeddings.
        """
        return {
            "dense": self.dense_model.encode_query(query).tolist(),
            "sparse": _sparse(next(iter(self.sparse_model.embed([query])))),
        }
