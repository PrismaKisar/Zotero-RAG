"""Embedding manager for dense and sparse vector generation."""

import logging

import numpy as np
import ollama
from device import resolve_device
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_ENCODE_BATCH_SIZE = 32
# The reranker reads 512 tokens, and the dense comparison (benchmark/retrieval_models.py)
# measured every model at this window, so the index embeds what the reranker sees.
MAX_TOKENS = 512


class EmbeddingManager:
    """Generate dense and sparse embeddings for indexing and retrieval."""

    def __init__(self,
                dense_model_name: str = "google/embeddinggemma-300m",
                ollama_url: str = "http://localhost:11434",
                device: str | None = None,
                encode_batch_size: int | None = None,
                use_chunk_contextualization: bool = False):
        """Initialize embedding models and runtime options.

        Args:
            dense_model_name: sentence-transformers dense model name.
            device: Device for dense model ('cpu', 'cuda', 'mps'). Auto-detect if None.
            encode_batch_size: Batch size for chunk encoding. Defaults to
                ``DEFAULT_ENCODE_BATCH_SIZE`` when None or 0.
        """
        self.dense_model_name = dense_model_name
        self.sparse_model_name = "Qdrant/bm25"
        self.ollama_url = ollama_url
        self.context_model_name = "llama3.2:3b"

        self.device = resolve_device(device)
        self.encode_batch_size = encode_batch_size or DEFAULT_ENCODE_BATCH_SIZE
        self.use_chunk_contextualization = use_chunk_contextualization

        # Dense runs on torch, so it gets the accelerator: on an M4, EmbeddingGemma on
        # MPS encodes QASA at 29 chunks/s against 16 for the old ONNX bge on CPU. The
        # model's query/document prompts are applied by encode_query/encode_document.
        # BM25 is a tokenizer plus IDF, so FastEmbed's CPU path is all it needs.
        self.dense_model = SentenceTransformer(self.dense_model_name, device=self.device)
        self.dense_model.max_seq_length = MAX_TOKENS
        self.sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name,
                                                cuda=self.device == "cuda")

        if self.use_chunk_contextualization:
            self.ollama_client = ollama.Client(host=self.ollama_url)

    @property
    def vector_size(self) -> int:
        """Return the dense embedding dimensionality."""
        return self.dense_model.get_embedding_dimension()

    def _generate_document_summary(self, document_text: str) -> str:
        """Generates a concise global summary of the given document.

        Args:
            document_text: The full text of the document to summarize.
            
        Returns:
            A concise summary string that captures the main themes and content of the document.
        """
        required_ctx = (len(document_text) // 4) + 1024
        optimal_num_ctx = max(2048, min(required_ctx, 16384))

        prompt = (
            "You are an expert AI assistant preparing metadata for a vector database.\n"
            "Read the following document and write a dense, entity-rich summary in exactly one paragraph.\n"
            "Include the main subject and scope, specific systems, methods, algorithms, datasets, metrics,\n"
            "and key entities such as organizations, authors, products, standards, and technologies.\n"
            "Do NOT use bullets, numbered lists, or markdown formatting.\n\n"
            f"<document>\n{document_text}\n</document>"
        )

        try:
            response = self.ollama_client.generate(
                model=self.context_model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1, 
                    "num_predict": 400,
                    "num_ctx": optimal_num_ctx
                },
                keep_alive=-1
            )
            return response['response'].strip()
        except Exception as e:  # noqa: BLE001 - Ollama optional: degrade to no summary
            logger.warning(f"Document summary generation failed: {e}")
            return ""

    def generate_contextual_chunks(self, 
                                document_text:str, 
                                all_texts:list[str]) -> list[str]:
        """Generate contextualized chunks by prompting an LLM to provide succinct context for each chunk.
        
        Args:
            document_text: The full text of the document.
            all_texts: List of text chunks to contextualize.

        Returns:
            List of contextualized text chunks, where each chunk is prefixed with a succinct context.
        """
        contextualized_results = []

        doc_summary = self._generate_document_summary(document_text)

        max_chunk_chars = max((len(chunk) for chunk in all_texts), default=0)
        prompt_tokens = (len(doc_summary) + max_chunk_chars) // 4
        predict_tokens = 100
        buffer_tokens = 250

        required_ctx = prompt_tokens + predict_tokens + buffer_tokens
        optimal_num_ctx = max(2048, min(required_ctx, 16384))

        # Prompt template for generating contextualized chunks one chunk at a time.
        prompt_template = (
            "<document_summary>\n{doc_summary}\n</document_summary>\n\n"
            "<current_chunk>\n{chunk}\n</current_chunk>\n\n"
            "You are an assistant organizing a vector database. Based on the global summary above, "
            "provide a succinct context (max 30 words) that describes what THIS chunk is about, "
            "so that it can be found via semantic search.\n"
            "Rules:\n"
            "- Name the specific topic, entity, method, or metric discussed in this chunk.\n"
            "- Avoid generic phrases like 'This chunk discusses...' or 'In this section...'.\n"
            "- Output only the context sentence, without labels, bullets, or quotes.\n\n"
            "Context sentence:"
        )

        for i, chunk in enumerate(all_texts):
            formatted_prompt = prompt_template.format(
                doc_summary=doc_summary, 
                chunk=chunk.strip()
            )

            try:
                response = self.ollama_client.generate(
                    model=self.context_model_name,
                    prompt=formatted_prompt,
                    options={
                        "temperature": 0.1, 
                        "num_predict": 100,
                        "num_ctx": optimal_num_ctx
                    },
                    keep_alive=-1
                )

                context_prefix = response["response"].strip().strip('"').strip("'")
                context_prefix = context_prefix.replace("```", "").strip()

                full_chunk = f"{context_prefix}\n\n{chunk}" if context_prefix else chunk
                contextualized_results.append(full_chunk)
            except Exception as e:  # noqa: BLE001 - Ollama optional: keep the raw chunk
                logger.warning(f"Contextual chunk generation failed for chunk index {i}: {e}")
                contextualized_results.append(chunk)

        self._flush_ollama_cache()

        return contextualized_results

    def _flush_ollama_cache(self):
        """Flush Ollama's model cache to free up memory."""
        try:
            self.ollama_client.generate(model=self.context_model_name, keep_alive=0)
            logger.info("Successfully flushed Ollama cache for model: %s", self.context_model_name)
        except Exception as e:  # noqa: BLE001 - cache flush is best-effort
            logger.warning("Failed to flush Ollama cache: %s", str(e))

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
