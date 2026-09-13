"""Cross-encoder reranking model."""

import logging

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
            order_by_retrieval: bool = False) -> list[RerankedChunk]:
        """Rerank candidates using cross-encoder scores.
        
        Args:
            query: The query string.
            candidates: List of (Chunk, retrieval_score) tuples.
            threshold: Minimum cross-encoder probability to keep a candidate. The
                model is conservative: see ``question_presets`` for the scale.
            progress_callback: Function(current, total, message) for progress updates.
            order_by_retrieval: Keep the cross-encoder's threshold but leave the
                survivors in retrieval's order. The two uses of the rerank score
                are separable and the benchmarks disagree about them: as a filter
                it roughly doubles attribution precision, while as a ranking it
                puts gold evidence below where reciprocal-rank fusion had it on
                every row where retrieval found enough to judge. Off by default,
                since it is an intervention and not yet an accepted default.

        Returns:
            List of (Chunk, retrieval_score, rerank_score) tuples, filtered by
            rerank_score and sorted by rerank_score descending, or by
            retrieval_score when ``order_by_retrieval``.
        """
        if not candidates:
            return []
        
        pairs = [[query, candidate[0].text] for candidate in candidates]
        probs = []
        num_batches = (len(pairs) + self.batch_size - 1) // self.batch_size
        for batch_idx, start in enumerate(range(0, len(pairs), self.batch_size)):
            batch = pairs[start:start + self.batch_size]
            # ``activation_fn`` turns the logits into probabilities; a second
            # sigmoid used to squash them into [0.5, 0.73], where the threshold
            # could no longer separate relevant passages from noise.
            probs.extend(self.model.predict(batch, activation_fn=torch.nn.Sigmoid(),
                                            show_progress_bar=False))
            if progress_callback:
                progress_callback(start + len(batch), len(pairs),
                                  f"Scoring candidates - Batch {batch_idx + 1}/{num_batches}")
        
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
        
        # Sort by rerank score descending, or keep retrieval's order if asked
        sort_key = "retrieval_score" if order_by_retrieval else "rerank_score"
        filtered_candidates.sort(key=lambda x: getattr(x, sort_key), reverse=True)
        
        if progress_callback:
            progress_callback(
                len(candidates),
                len(candidates),
                (
                    "Reranking complete: "
                    f"{len(filtered_candidates)} of {len(candidates)} candidates passed threshold."
                ),
            )
        
        logger.debug(f"Reranking: {len(candidates)} -> {len(filtered_candidates)} "
                    f"chunks passed threshold {threshold:.4f}")

        return filtered_candidates
