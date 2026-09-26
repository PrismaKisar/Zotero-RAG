"""Embedding manager for dense and sparse vector generation."""

import numpy as np
from device import resolve_device
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

DEFAULT_ENCODE_BATCH_SIZE = 32
# The default, not a fixed setting: the user may index with another model, and
# the Qdrant collection is named after it, so a second model builds a second
# index instead of corrupting the first.
DEFAULT_DENSE_MODEL = "google/embeddinggemma-300m"
# The reranker reads 512 tokens, and the dense comparison (benchmark/retrieval_models.py)
# measured every model at this window, so the index embeds what the reranker sees.
MAX_TOKENS = 512


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

    def encode_chunks(self, progress_callback, all_texts: list[str]) -> dict[str, list[float]]:
        """Encode chunks into dense+sparse vectors with progress updates.
        
        Args:
            progress_callback: Function to call with progress updates (stage, current, total, message).
            all_texts: List of chunk texts to encode.
        
        Returns:
            Dictionary with 'dense' and 'sparse' keys containing the respective embeddings.
        """
        if not all_texts:
            return {"dense": [], "sparse": []}

        effective_batch_size = self.encode_batch_size

        if progress_callback:
            progress_callback(
                "encoding",
                0,
                len(all_texts),
                f"Encoding with batch size {effective_batch_size}...",
            )

        dense_embeddings_list = []
        sparse_embeddings_list = []

        for i in range(0, len(all_texts), effective_batch_size):
            batch = all_texts[i : i + effective_batch_size]
            try:
                batch_dense = self.dense_model.encode_document(batch, batch_size=len(batch))
                batch_sparse_gen = self.sparse_model.embed(batch)
                batch_sparse = [
                    {"indices": res.indices.tolist(), "values": res.values.tolist()}
                    for res in batch_sparse_gen
                ]
            except (RuntimeError, MemoryError) as exc:
                error_str = str(exc).lower()
                if any(
                    phrase in error_str
                    for phrase in ["out of memory", "buffer size", "cuda", "memory", "onnxruntime"]
                ):
                    fallback_size = max(1, effective_batch_size // 2)
                    if progress_callback:
                        progress_callback(
                            "encoding",
                            i,
                            len(all_texts),
                            f"OOM: Reducing batch size to {fallback_size}...",
                        )

                    dense_chunks = []
                    batch_sparse = []
                    for j in range(0, len(batch), fallback_size):
                        fallback_batch = batch[j : j + fallback_size]

                        dense_chunks.append(self.dense_model.encode_document(
                            fallback_batch, batch_size=len(fallback_batch)))

                        sub_sparse_gen = self.sparse_model.embed(fallback_batch)
                        batch_sparse.extend(
                            {"indices": res.indices.tolist(), "values": res.values.tolist()}
                            for res in sub_sparse_gen
                        )

                    batch_dense = np.vstack(dense_chunks)
                else:
                    raise

            dense_embeddings_list.append(batch_dense)
            sparse_embeddings_list.extend(batch_sparse)

            processed = min(i + len(batch), len(all_texts))
            if progress_callback:
                progress_callback(
                    "encoding",
                    processed,
                    len(all_texts),
                    f"Encoded {processed}/{len(all_texts)} chunks...",
                )

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
        query_dense_embedding = self.dense_model.encode_query(query).tolist()

        query_sparse_obj = next(iter(self.sparse_model.embed([query])))
        query_sparse_embedding = {
            "indices": query_sparse_obj.indices.tolist(),
            "values": query_sparse_obj.values.tolist(),
        }

        return {
            "dense": query_dense_embedding,
            "sparse": query_sparse_embedding,
        }
