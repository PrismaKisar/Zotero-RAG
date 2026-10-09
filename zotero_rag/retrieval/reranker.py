"""Cross-encoder reranking model.

The default is gte-reranker-modernbert-base. The former bge-reranker-base
ordered the hybrid top-30 *worse* than retrieval had: -0.070 recall@1 within the
question's own paper on QASA. Measured offline on the same candidates
(benchmark/studies/reranker_diagnosis.py), gte lifts recall@1 over retrieval's order by
+0.147 on QASA and +0.115 on QASPER across the library, matching the larger
bge-reranker-v2-m3 and jina-reranker-v3.5 at under half their latency on MPS.
"""

import logging
import math

import torch
from sentence_transformers import CrossEncoder

from zotero_rag.device import resolve_device
from zotero_rag.models import Chunk, RerankedChunk
from zotero_rag.retrieval.selection import rerank_floor

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
DEFAULT_RERANK_BATCH_SIZE = 32


class Reranker:
    """Handles reranking of retrieved candidates using a cross-encoder model."""
    
    def __init__(self, 
                model_name: str = DEFAULT_RERANKER_MODEL,
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
            gate: float = 0.0,
            margin: float = 1.0,
            progress_callback=None,
            order_by_retrieval: bool = False,
            absolute_cut: bool = False) -> list[RerankedChunk]:
        """Rerank candidates using cross-encoder scores.

        Args:
            query: The query string.
            candidates: List of (Chunk, retrieval_score) tuples.
            gate: Minimum probability the *best* candidate must reach for the
                question to be answered at all. Below it nothing is returned,
                however many middling candidates there are.
            margin: How far below that best candidate a candidate may score and
                still be kept. The cut is relative because gte's probabilities
                crowd at a different height per question, so one absolute cut
                lands mid-crowd on some questions and above it on others; see
                ``settings`` for the values and benchmark/studies/answer_gate.py for the
                search behind them. The defaults keep everything: the benchmarks
                that only want the scores pass no cut.
            progress_callback: Function(current, total, message) for progress updates.
            order_by_retrieval: Keep the cross-encoder's threshold but leave the
                survivors in retrieval's order. The two uses of the rerank score
                are separable and the benchmarks disagree about them: as a filter
                it roughly doubles attribution precision, while as a ranking it
                puts gold evidence below where reciprocal-rank fusion had it on
                every row where retrieval found enough to judge. Off by default,
                since it is an intervention and not yet an accepted default.
            absolute_cut: Keep every candidate at or above ``gate`` instead of
                those within ``margin`` of the best: the rule the margin
                replaced, for the benchmarks only (see ``selection``).

        Returns:
            List of (Chunk, retrieval_score, rerank_score) tuples, empty when no
            candidate clears ``gate``, otherwise those within ``margin`` of the
            best one, sorted by rerank_score descending, or by retrieval_score
            when ``order_by_retrieval``.
        """
        if not candidates:
            return []
        
        probs = self._probabilities(query, [chunk.text for chunk, _ in candidates],
                                    progress_callback)
        scored_candidates = [
            RerankedChunk(chunk=chunk, retrieval_score=retrieval_score, rerank_score=prob)
            for (chunk, retrieval_score), prob in zip(candidates, probs, strict=True)
        ]

        best = max(item.rerank_score for item in scored_candidates)
        # best is already in hand for the log below; the floor only reads the best
        floor = rerank_floor((best,), gate, margin, absolute=absolute_cut)
        filtered_candidates = [] if floor is None else [
            item for item in scored_candidates if item.rerank_score >= floor
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
                    f"{len(filtered_candidates)} of {len(candidates)} candidates kept."
                ),
            )
        
        logger.debug(f"Reranking: {len(candidates)} -> {len(filtered_candidates)} chunks; "
                    f"best {best:.4f} against gate {gate:.2f}, margin {margin:.3f}")

        return filtered_candidates

    def _probabilities(self, query: str, texts: list[str], progress_callback=None) -> list[float]:
        """The cross-encoder's probability that each text answers ``query``, batch by batch."""
        probs = []
        num_batches = math.ceil(len(texts) / self.batch_size)
        for batch_idx, start in enumerate(range(0, len(texts), self.batch_size)):
            batch = [[query, text] for text in texts[start:start + self.batch_size]]
            # ``activation_fn`` turns the logits into probabilities; a second
            # sigmoid used to squash them into [0.5, 0.73], where the threshold
            # could no longer separate relevant passages from noise.
            probs.extend(float(p) for p in self.model.predict(
                batch, activation_fn=torch.nn.Sigmoid(), show_progress_bar=False))
            if progress_callback:
                progress_callback(start + len(batch), len(texts),
                                  f"Scoring candidates - Batch {batch_idx + 1}/{num_batches}")
        return probs
