"""Offline test of question paraphrasing, used as well as it can be.

num_paraphrases=2 was the worst row of the ablation (answer F1 -0.040, paired),
and the saved runs show why: retrieval did not move, while the reranker and the
reader both take the best score over the variants and let the noise through.
Before the feature is removed, this checks whether it fails as built or as an
idea: does expansion improve retrieval - the only stage it exists for - when the
paraphrases are faithful and the variant rankings are fused properly?

Two factors, crossed, against the original question alone:

- generation: ``shipped`` is the pipeline's generator as it was before the
  removal, copied here (sampled at temperature 1.5, keeping the candidates
  *least* like the question);
  ``faithful`` is the same T5 model decoded by beam search - deterministic -
  keeping the candidates *most* like the question by embedding cosine;
  ``llm`` asks the generative reader's own model (greedy) for rewrites that
  keep every technical term, which is the test of whether T5 is the problem.
- fusion: ``union`` is the pipeline's merge (the first variant to find a chunk
  sets its score, then best first); ``rrf`` is reciprocal-rank fusion.

The original question is scored at the per-variant depth and again at the depth
the variants reach together, so a gain that is only "more candidates" shows up
as the same gain from a deeper plain search.

Retrieval is replicated offline and exactly for all three modes: dense cosine
with the index's embedding model, BM25 with the index's sparse model and the
IDF Qdrant applies, hybrid as RRF of the two branches. The reranker and the
reader are out of scope on purpose: recall the variants do not add here cannot
be recovered downstream, and the harm they do there is already measured.

Usage:
  python -m benchmark.paraphrase_retrieval --golden-dir output_qasper
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# appended, never prepended: the repo root must keep priority over this directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from benchmark.ablation import load_jsonl
from benchmark.retrieval_metrics import bootstrap_ci, recall_at_k
from benchmark.retrieval_score_distribution import gold_ids, load_corpus, unit_vectors

LIMIT = 30         # result_limit: the depth of every per-variant list
NUM_VARIANTS = 2   # num_paraphrases as it was shipped
DEEP = LIMIT * (1 + NUM_VARIANTS)  # what the variants can reach together
RRF_K = 60         # Qdrant's default constant
KS = (1, 3, 10, LIMIT, DEEP)
MODES = ("dense", "sparse", "hybrid")
PARAPHRASER_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"

REWRITE_PROMPT = (
    "Rewrite the question below in {n} different ways, for searching a scientific paper.\n"
    "Keep every technical term, name and number exactly as written; change only the rest.\n"
    "Output one rewrite per line, nothing else.\n\n"
    "Question: {question}"
)
# Always matches: numbering, a bullet and quotes are all optional.
REWRITE_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*•])?\s*[\"“]?(.*?)[\"”]?\s*$")


def top(scores: np.ndarray, limit: int = LIMIT) -> list[tuple[int, float]]:
    """Best ``limit`` (index, score) pairs, ties broken by index."""
    order = np.lexsort((np.arange(len(scores)), -scores))[:limit]
    return [(int(i), float(scores[i])) for i in order]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion, ties broken by id so the order is reproducible."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


def union_by_score(scored: list[list[tuple[int, float]]]) -> list[int]:
    """The pipeline's merge: the first list to hold a chunk sets its score."""
    seen = {}
    for results in scored:
        for item, score in results:
            seen.setdefault(item, score)
    return sorted(seen, key=lambda item: (-seen[item], item))


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


def jaccard(a: str, b: str) -> float:
    """Word-set overlap, lowercased: the similarity the shipped filter used."""
    words_a, words_b = set(a.lower().split()), set(b.lower().split())
    union = len(words_a | words_b)
    return len(words_a & words_b) / union if union else 1.0


def shipped_variants(question: str, candidates: list[str], n: int = NUM_VARIANTS) -> list[str]:
    """The removed QAEngine.expand_question filter, unchanged: the question plus up to ``n``.

    A candidate must be under 0.75 Jaccard from the question, and is taken
    least similar first if it is under 0.70 from everything taken so far.
    """
    kept = []
    for candidate in candidates:
        candidate = candidate.replace("paraphrase:", "").strip()
        if candidate and candidate.lower() != question.lower():
            similarity = jaccard(question, candidate)
            if similarity < 0.75:
                kept.append((similarity, candidate))
    kept.sort(key=lambda pair: pair[0])
    variations = [question]
    for _, candidate in kept:
        if len(variations) - 1 >= n:
            break
        if all(jaccard(candidate, existing) < 0.70 for existing in variations):
            variations.append(candidate)
    return variations


def faithful_variants(question: str, candidates: list[str], similarity,
                      n: int = NUM_VARIANTS) -> list[str]:
    """The ``n`` distinct candidates closest in meaning to ``question``.

    ``similarity`` maps a list of texts to their cosine with the question. The
    shipped filter keeps the candidates *least* like the question, and on a
    short technical question the only way to be unlike it is to drop the
    technical terms - which are what retrieval matches on.
    """
    unique = distinct(question, candidates)
    if not unique:
        return []
    sims = similarity(unique)
    order = sorted(range(len(unique)), key=lambda i: -sims[i])
    return [unique[i] for i in order[:n]]


def distinct(question: str, candidates: list[str]) -> list[str]:
    """Candidates that differ from the question and each other, up to case and spacing."""
    seen = {" ".join(question.lower().split())}
    unique = []
    for candidate in candidates:
        key = " ".join(candidate.lower().split())
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def parse_rewrites(raw: str, question: str, n: int = NUM_VARIANTS) -> list[str]:
    """The model's rewrites: one per line, numbering and quotes stripped.

    Only lines ending in a question mark count, which drops a preamble like
    "Here are two rewrites:" without a list of phrasings to recognise.
    """
    lines = [REWRITE_LINE.match(line).group(1) for line in raw.splitlines()]
    return distinct(question, [line for line in lines if line.endswith("?")])[:n]


def recalls(ranked: list[int], gold: set[int]) -> dict[int, float]:
    return {k: recall_at_k(ranked, gold, k) for k in KS}


def report(rows: dict, produced: dict) -> str:
    """One table per mode: mean recall per arm, and paired deltas against the plain search."""
    arms = sorted({arm for _, arm in rows} - {"original"})
    header = ["arm", *(f"recall@{k}" for k in KS),
              "Δ@10 vs original", f"Δ@{LIMIT} vs original", f"Δ@{DEEP} vs original@{DEEP}"]
    out = ["# Question paraphrasing, offline retrieval",
           f"{len(rows[(MODES[0], 'original')])} questions. Variants actually produced per "
           "question: " + ", ".join(f"{g} {np.mean(v):.2f}" for g, v in produced.items()) + ".",
           "Δ is the paired mean difference with its 95% bootstrap interval."]
    for mode in MODES:
        base = rows[(mode, "original")]
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines.append("| original | " + " | ".join(
            f"{np.mean([r[k] for r in base]):.3f}" for k in KS) + " | | | |")
        for arm in arms:
            cells = [f"{np.mean([r[k] for r in rows[(mode, arm)]]):.3f}" for k in KS]
            for k in (10, LIMIT, DEEP):
                deltas = [a[k] - b[k] for a, b in zip(rows[(mode, arm)], base, strict=True)]
                low, high = bootstrap_ci(deltas)
                cells.append(f"{np.mean(deltas):+.3f} [{low:+.3f}, {high:+.3f}]")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        out += [f"## {mode}", "\n".join(lines)]
    return "\n\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="output_qasper",
                        help="holds golden_set_aligned.jsonl and chunks/")
    parser.add_argument("--seed", type=int, default=42,
                        help="sampling seed for the shipped generator, which is not deterministic")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="the llm generator asks the generative reader's model")
    args = parser.parse_args()

    import torch
    from embedding_manager import EmbeddingManager  # heavy imports (ONNX, T5)
    from generative_reader import THINKING, GenerativeReader
    from transformers import pipeline

    # First, so a missing Ollama fails before minutes of encoding, not after.
    rewriter = GenerativeReader(ollama_url=args.ollama_url)

    golden_dir = Path(args.golden_dir)
    ids, texts = load_corpus(golden_dir / "chunks")
    position = {chunk_id: i for i, chunk_id in enumerate(ids)}
    embedder = EmbeddingManager()
    encoded = embedder.encode_chunks(None, texts)
    chunk_vectors = unit_vectors(encoded["dense"])
    index, idf = inverted_index(encoded["sparse"]), idf_table(encoded["sparse"])

    paraphraser = pipeline("text2text-generation", model=PARAPHRASER_MODEL, device="cpu")

    cache = {}

    def scores(text):
        if text not in cache:
            query = embedder.encode_query(text)
            cache[text] = (chunk_vectors @ unit_vectors(query["dense"]),
                           bm25(query["sparse"], index, idf, len(ids)))
        return cache[text]

    def shipped(question):
        # the removed generator's call, sampling arguments included
        torch.manual_seed(args.seed)
        samples = paraphraser(
            f"paraphrase: {question}", max_new_tokens=64,
            num_return_sequences=NUM_VARIANTS * 3, num_beams=NUM_VARIANTS * 3,
            temperature=1.5, top_k=50, top_p=0.95, do_sample=True, early_stopping=True)
        return shipped_variants(question, [s["generated_text"] for s in samples])

    def faithful(question):
        beams = paraphraser(
            f"paraphrase: {question}", max_new_tokens=64, do_sample=False,
            num_beams=NUM_VARIANTS * 3, num_return_sequences=NUM_VARIANTS * 3)
        candidates = [b["generated_text"].replace("paraphrase:", "").strip() for b in beams]
        target = unit_vectors(embedder.encode_query(question)["dense"])
        return faithful_variants(question, candidates, lambda ts: np.array(
            [unit_vectors(embedder.encode_query(t)["dense"]) @ target for t in ts]))

    def llm(question):
        response = rewriter.client.generate(
            model=rewriter.model_name, think=THINKING, keep_alive=-1,
            prompt=REWRITE_PROMPT.format(n=NUM_VARIANTS, question=question),
            options={"temperature": 0, "num_predict": 128})
        return parse_rewrites(response["response"], question)

    rows, produced, variants_out = defaultdict(list), defaultdict(list), []
    for record in load_jsonl(golden_dir / "golden_set_aligned.jsonl"):
        gold = {position[g] for g in gold_ids(record) if g in position}
        if not gold:
            continue
        question = record["question"]
        variants = {"shipped": shipped(question),
                    "faithful": [question, *faithful(question)],
                    "llm": [question, *llm(question)]}
        variants_out.append({"question_id": record["question_id"], **variants})
        for generator, texts_ in variants.items():
            produced[generator].append(len(texts_) - 1)
        for mode in MODES:
            plain = [i for i, _ in retrieve(mode, *scores(question))]
            deep = [i for i, _ in retrieve(mode, *scores(question), limit=DEEP)]
            rows[(mode, "original")].append({**recalls(plain, gold),
                                             DEEP: recall_at_k(deep, gold, DEEP)})
            for generator, texts_ in variants.items():
                lists = [retrieve(mode, *scores(t)) for t in texts_]
                rows[(mode, f"{generator}+union")].append(recalls(union_by_score(lists), gold))
                rows[(mode, f"{generator}+rrf")].append(recalls(
                    [i for i, _ in rrf([[i for i, _ in hits] for hits in lists])], gold))

    (golden_dir / "paraphrase_variants.jsonl").write_text(
        "".join(json.dumps(v) + "\n" for v in variants_out))
    out_file = golden_dir / "paraphrase_retrieval.md"
    out_file.write_text(report(rows, produced))
    print(f"{len(variants_out)} questions -> {out_file}")


if __name__ == "__main__":
    main()
