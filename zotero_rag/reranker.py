"""Cross-encoder reranking model."""

import logging

import numpy as np
import torch
from device import resolve_device
from models import Chunk, RerankedChunk
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANK_BATCH_SIZE = 32


class Reranker:
    """Handles reranking of retrieved candidates using a cross-encoder model."""
    
    def __init__(self, 
                model_name: str = 'BAAI/bge-reranker-base', 
                device: str | None = None,
                batch_size: int | None = None):
        """Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detect if None.
            batch_size: Batch size for reranking. Defaults to ``DEFAULT_RERANK_BATCH_SIZE``.
        """
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size or DEFAULT_RERANK_BATCH_SIZE
        self.model = CrossEncoder(model_name, device=self.device)
        logger.info(f"Reranker initialized with model {model_name} on {self.device}")
    
    def rerank(self,
            query: str,
            candidates: list[tuple[Chunk, float]],
            threshold: float = 0.45,
            progress_callback=None,
            query_variations: list[str] | None = None) -> list[RerankedChunk]:
        """Rerank candidates using cross-encoder scores.
        
        Args:
            query: The query string.
            candidates: List of (Chunk, retrieval_score) tuples.
            threshold: Minimum cross-encoder probability to keep a candidate. The
                model is conservative: see ``question_presets`` for the scale.
            progress_callback: Function(current, total, message) for progress updates.
            query_variations: List of query paraphrases to average scores over.
            
        Returns:
            List of (Chunk, retrieval_score, rerank_score) tuples,
            filtered and sorted by rerank_score descending.
        """
        if not candidates:
            return []
        
        # Use query variations if provided, otherwise just use the original query
        queries_to_use = query_variations if query_variations else [query]
        
        effective_batch_size = self.batch_size

        # Score candidates with each query variation and average
        all_probs_per_variation = []
        
        total_operations = len(queries_to_use) * len(candidates)
        completed_operations = 0
        
        for var_idx, q_var in enumerate(queries_to_use):
            # Prepare pairs for this variation
            var_pairs = [[q_var, p[0].text] for p in candidates]
            
            # Predict scores in batches with progress tracking
            all_scores = []
            num_batches = (len(var_pairs) + effective_batch_size - 1) // effective_batch_size
            
            for batch_idx, i in enumerate(range(0, len(var_pairs), effective_batch_size)):
                batch_pairs = var_pairs[i:i + effective_batch_size]
                
                batch_scores = self.model.predict(batch_pairs, activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
                all_scores.extend(batch_scores)
                
                # Update progress with current variation info
                processed_in_batch = min(len(batch_pairs), len(var_pairs) - i)
                completed_operations += processed_in_batch
                
                if progress_callback:
                    if len(queries_to_use) > 1:
                        variation_info = (
                            f"Scoring merged candidates (variation {var_idx + 1}/{len(queries_to_use)})"
                        )
                    else:
                        variation_info = "Scoring merged candidates"
                    batch_info = f"Batch {batch_idx + 1}/{num_batches}"
                    message = f"{variation_info} - {batch_info}"
                    progress_callback(completed_operations, total_operations, message)
            
            # ``activation_fn`` above already turned the logits into probabilities;
            # a second sigmoid used to squash them into [0.5, 0.73], where the
            # threshold could no longer separate relevant passages from noise.
            all_probs_per_variation.append(np.array(all_scores))
        
        # Calculate max probabilities across all variations
        if len(all_probs_per_variation) > 1:
            probs = np.max(all_probs_per_variation, axis=0)
            logger.info(f"Using max rerank scores across {len(queries_to_use)} query variations")
        else:
            probs = all_probs_per_variation[0]
        
        # Combine probabilities with candidate data
        # Each item: (chunk, retrieval_score, rerank_score)
        scored_candidates = [
            RerankedChunk(
                chunk=p[0],
                retrieval_score=p[1],
                rerank_score=float(prob),
            )
            for p, prob in zip(candidates, probs)
        ]
        
        filtered_candidates = [
            item for item in scored_candidates
            if item.rerank_score >= threshold
        ]
        
        # Sort by rerank score descending
        filtered_candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if progress_callback:
            progress_callback(
                len(candidates),
                len(candidates),
                (
                    "Reranking complete: "
                    f"{len(filtered_candidates)} of {len(candidates)} merged candidates passed threshold."
                ),
            )
        
        logger.debug(f"Reranking: {len(candidates)} -> {len(filtered_candidates)} "
                    f"chunks passed threshold {threshold:.4f}")

        return filtered_candidates
