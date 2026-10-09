"""The index's search, replicated offline and exactly, for the studies that score it.

Every offline study needs the same things: the benchmark corpus as chunk ids and
texts, which of those chunks a golden record marks as evidence, and per question
the hybrid ranking Qdrant would return - dense cosine, BM25 with Qdrant's IDF,
and the reciprocal-rank fusion of the two. Kept here rather than in the study
that first needed them, so a study does not import another study to search.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from zotero_rag.retrieval.qdrant_manager import RRF_K


def load_corpus(chunks_dir: Path) -> tuple[list[tuple[str, int]], list[str]]:
    """Every chunk of every paper, as (paper_id, chunk_index) ids and texts."""
    ids, texts = [], []
    for path in sorted(chunks_dir.glob("*.json")):
        for index, text in enumerate(json.loads(path.read_text())):
            ids.append((path.stem, index))
            texts.append(text)
    return ids, texts


def unit_vectors(vectors) -> np.ndarray:
    """Scale to unit length, so a dot product is the cosine Qdrant computes."""
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def gold_ids(record: dict) -> set[tuple[str, int]]:
    return {(record["paper_id"], hit["chunk_index"])
            for hits in record["aligned_chunks"].values() for hit in hits}


LIMIT = 30         # result_limit: how many candidates the reranker receives


def top(scores: np.ndarray, limit: int = LIMIT) -> list[tuple[int, float]]:
    """Best ``limit`` (index, score) pairs, ties broken by index."""
    order = np.lexsort((np.arange(len(scores)), -scores))[:limit]
    return [(int(i), float(scores[i])) for i in order]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion as the index computes it (rank from zero), ties broken by id."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


def idf_table(docs: list[dict]) -> dict[int, float]:
    """Qdrant's IDF modifier, ln((N - n + 0.5) / (n + 0.5) + 1), per token."""
    counts = defaultdict(int)
    for doc in docs:
        for token in doc["indices"]:
            counts[token] += 1
    n = len(docs)
    return {token: math.log((n - c + 0.5) / (c + 0.5) + 1) for token, c in counts.items()}


def inverted_index(docs: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    postings = defaultdict(list)
    for d, doc in enumerate(docs):
        for token, weight in zip(doc["indices"], doc["values"], strict=True):
            postings[token].append((d, weight))
    return {token: (np.array([d for d, _ in p]), np.array([w for _, w in p]))
            for token, p in postings.items()}


def bm25(query: dict, index: dict, idf: dict, n_docs: int) -> np.ndarray:
    """Sparse dot product with the IDF applied, as Qdrant scores a BM25 vector."""
    scores = np.zeros(n_docs)
    for token, weight in zip(query["indices"], query["values"], strict=True):
        if token in index:
            docs, weights = index[token]
            scores[docs] += idf[token] * weight * weights
    return scores


def retrieve(mode: str, dense: np.ndarray, sparse: np.ndarray,
             limit: int = LIMIT) -> list[tuple[int, float]]:
    """One query's ranked list, the way Qdrant builds it for ``mode``."""
    dense_hits = top(dense, limit)
    # a sparse search only returns chunks sharing a token with the query
    sparse_hits = [(i, s) for i, s in top(sparse, limit) if s > 0]
    if mode == "dense":
        return dense_hits
    if mode == "sparse":
        return sparse_hits
    return rrf([[i for i, _ in dense_hits], [i for i, _ in sparse_hits]])[:limit]


class HybridSearch:
    """The index's hybrid search, replicated offline and exactly over ``chunks_dir``.

    Every offline study needs the same thing: the corpus, its dense and BM25
    encodings, and per question the RRF of the two branches.
    """

    def __init__(self, chunks_dir: Path):
        from zotero_rag.retrieval.embedding_manager import (
            EmbeddingManager,  # heavy import (ONNX)
        )

        self.ids, texts = load_corpus(chunks_dir)
        self.text_of = dict(zip(self.ids, texts, strict=True))
        self.paper_of = np.array([paper for paper, _ in self.ids])
        self.embedder = EmbeddingManager()
        encoded = self.embedder.encode_chunks(texts)
        self.vectors = unit_vectors(encoded["dense"])
        self.index, self.idf = inverted_index(encoded["sparse"]), idf_table(encoded["sparse"])

    def top(self, question: str, paper: str | None = None, limit: int = LIMIT) -> list:
        """The hybrid top-``limit`` chunk ids, over the library or over one paper."""
        query = self.embedder.encode_query(question)
        dense = self.vectors @ unit_vectors(query["dense"])
        sparse = bm25(query["sparse"], self.index, self.idf, len(self.ids))
        keep = np.flatnonzero(self.paper_of == paper) if paper else np.arange(len(self.ids))
        return [self.ids[keep[i]] for i, _ in retrieve("hybrid", dense[keep], sparse[keep], limit)]
