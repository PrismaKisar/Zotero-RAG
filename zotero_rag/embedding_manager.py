"""Embedding manager for dense and sparse vector generation."""

import logging
from typing import Dict, List

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
import ollama
import torch

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Generate dense and sparse embeddings for indexing and retrieval."""

    def __init__(self,
                dense_model_name: str = "BAAI/bge-base-en-v1.5",
                device: str = None,
                encode_batch_size: int = 8):
        """Initialize embedding models and runtime options.

        Args:
            dense_model_name: FastEmbed dense model name.
            device: Device for dense model ('cpu', 'cuda'). Auto-detect CUDA if None.
            encode_batch_size: Batch size for paragraph encoding. If None or 0, auto-detect.
        """
        self.dense_model_name = dense_model_name
        self.sparse_model_name = "Qdrant/bm25"
        self.context_model_name = "qwen2.5:3b"

        self.device = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
        self.encode_batch_size = encode_batch_size

        use_cuda = self.device == "cuda"
        self.dense_model = TextEmbedding(model_name=self.dense_model_name, cuda=use_cuda)
        self.sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name, cuda=use_cuda)
        self._vector_size = self._resolve_vector_size()

    @property
    def vector_size(self) -> int:
        """Return the dense embedding dimensionality."""
        return self._vector_size

    def _resolve_vector_size(self) -> int:
        """Resolve dense vector size from supported-model metadata, then fallback to probing."""
        try:
            for metadata in TextEmbedding.list_supported_models():
                metadata_name = (
                    metadata.get("model")
                    or metadata.get("model_name")
                    or metadata.get("name")
                )
                if metadata_name == self.dense_model_name:
                    dimension = metadata.get("dim") or metadata.get("dimensions") or metadata.get("size")
                    if isinstance(dimension, (int, float)):
                        return int(dimension)
        except Exception as exc:
            logger.debug("Unable to read FastEmbed model metadata for '%s': %s", self.dense_model_name, exc)

        sample_vector = next(iter(self.dense_model.embed(["dimension probe"])), None)
        if sample_vector is None:
            raise ValueError(f"Unable to determine vector size for model '{self.dense_model_name}'")

        return int(len(sample_vector))

    def _find_safe_batch_size(self,
                            sample_texts: List[str],
                            start_size: int = 2,
                            max_size: int = 128,
                            target_memory_fraction: float = 0.75) -> int:
        """Find a safe encoding batch size targeting a memory usage fraction."""
        if not sample_texts:
            return start_size

        test_sample = sample_texts[: min(100, len(sample_texts))]

        current_size = start_size
        last_safe_size = start_size

        while current_size <= max_size:
            try:
                test_batch = test_sample[: min(current_size, len(test_sample))]
                _ = np.asarray(list(self.dense_model.embed(test_batch)), dtype=np.float32)
                last_safe_size = current_size
                current_size = int(current_size * 1.5)
            except (RuntimeError, MemoryError) as exc:
                error_str = str(exc).lower()
                if any(
                    phrase in error_str
                    for phrase in ["out of memory", "buffer size", "cuda", "memory", "onnxruntime"]
                ):
                    return max(start_size, int(last_safe_size * target_memory_fraction))
                return last_safe_size
            except Exception:
                return last_safe_size

        return max(start_size, int(last_safe_size * target_memory_fraction))

    def generate_contextual_chunks(self, document_text:str, all_texts:List[str]) -> List[str]:
        """Generate contextualized chunks by prompting an LLM to provide succinct context for each chunk.
        
        Args:
            document_text: The full text of the document.
            all_texts: List of text chunks to contextualize.
            
        Returns:
            List of contextualized text chunks, where each chunk is prefixed with a succinct context.
        """
        contextualized_results = []

        prompt = (
            """<document> {document_text} </document> 
                Here is the chunk we want to situate within the whole document <chunk> {chunk_text} </chunk> 
                Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. 
                Answer only with the succinct context and nothing else. """
        )

        for chunk in all_texts:
            formatted_prompt = prompt.format(document_text=document_text, chunk_text=chunk)
            response = ollama.generate(
                model=self.context_model_name,
                prompt=formatted_prompt,
                options={
                    "temperature": 0.1, 
                    "num_predict": 100,
                    "num_ctx": 32768 #FIXME: da capire se serve (in caso i PDF sono tanto tanto lunghi)
                },
                keep_alive=-1
            )
                
            context_prefix = response['response'].strip()
            full_chunk = f"{context_prefix}\n\n{chunk}"

            contextualized_results.append(full_chunk)

        return contextualized_results

    def flush_ollama_cache(self):
        """Flush Ollama's model cache to free up memory."""
        try:
            ollama.generate(model=self.context_model_name, keep_alive=0)
            logger.info("Successfully flushed Ollama cache for model: %s", self.context_model_name)
        except Exception as e:
            logger.warning("Failed to flush Ollama cache: %s", str(e))

    def encode_paragraphs(self, progress_callback, all_texts: List[str]) -> Dict:
        """Encode paragraphs into dense+sparse vectors with progress updates."""
        if not all_texts:
            return {"dense": [], "sparse": []}

        if self.encode_batch_size is None or self.encode_batch_size == 0:
            if progress_callback:
                progress_callback("encoding", 0, len(all_texts), "Auto-detecting safe batch size...")
            effective_batch_size = self._find_safe_batch_size(all_texts, start_size=2, max_size=128)
            logger.info(f"Auto-detected encoding batch size: {effective_batch_size}")
        else:
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
                batch_dense = np.asarray(list(self.dense_model.embed(batch)), dtype=np.float32)
                if batch_dense.ndim == 1:
                    batch_dense = np.expand_dims(batch_dense, axis=0)

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

                        sub_dense = np.asarray(list(self.dense_model.embed(fallback_batch)), dtype=np.float32)
                        if sub_dense.ndim == 1:
                            sub_dense = np.expand_dims(sub_dense, axis=0)
                        dense_chunks.append(sub_dense)

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

    def encode_query(self, query: str) -> Dict[str, List[float]]:
        """Encode a single query into dense+sparse vectors."""
        query_dense_embedding = next(iter(self.dense_model.embed([query]))).tolist()

        query_sparse_obj = next(iter(self.sparse_model.embed([query])))
        query_sparse_embedding = {
            "indices": query_sparse_obj.indices.tolist(),
            "values": query_sparse_obj.values.tolist(),
        }

        return {
            "dense": query_dense_embedding,
            "sparse": query_sparse_embedding,
        }
