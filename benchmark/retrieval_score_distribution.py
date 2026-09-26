"""Dense cosine score distribution: the measurement that removed retrieval_threshold.

retrieval_threshold used to cut the dense branch by cosine similarity (the
``score_threshold`` of the dense query, or of the dense prefetch in hybrid
mode). The shipped config put it at 0.35-0.45, a value never checked against the
scores, and a threshold below every score a question ever sees is a no-op
however it is swept. This measures where the scores actually lie, the way
rerank_threshold was re-set against the cross-encoder's measured distribution;
what it found got the cut dropped rather than re-tuned (see zotero_rag/settings.py).

Every aligned question is scored against every chunk of the corpus with the
index's own embedding model, exactly: no HNSW, so these are the scores Qdrant
approximates and not a replay of one campaign. Reported:

- percentiles of gold-chunk scores, of the non-gold chunks inside the top
  ``limit`` (what the threshold actually competes against) and of every
  non-gold chunk in the corpus;
- a histogram of the first two, which is where a cut can separate them or not;
- an absolute sweep: per threshold, the share of gold chunks and of top-``limit``
  non-gold candidates that survive, mean survivors per question, the share of
  questions left with no dense candidate, and recall@limit after the cut;
- the same sweep on score / top-1 score, i.e. a per-question relative threshold,
  since an absolute cut means something different on every question.

In hybrid mode such a cut acted on the dense prefetch only and BM25 still filled
its own ``limit``, so the sweep describes the dense branch, not the fused list.

Usage:
  python -m benchmark.retrieval_score_distribution --golden-dir output_qasper \
      --out-file output_qasper/retrieval_score_distribution.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# appended, never prepended: the repo root must keep priority over this directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from benchmark.reports import load_jsonl, markdown_table

PERCENTILES = (5, 25, 50, 75, 95)
ABSOLUTE_THRESHOLDS = np.round(np.arange(0.34, 0.921, 0.02), 2)
RELATIVE_THRESHOLDS = np.round(np.arange(0.80, 1.001, 0.01), 2)
HISTOGRAM_BIN = 0.02


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


def question_scores(scores: np.ndarray, is_gold: np.ndarray, limit: int) -> dict:
    """One question's view: its top-``limit`` scores and every gold score.

    Gold scores are kept wherever the chunk ranks, so a gold chunk the prefetch
    never reaches still counts against the threshold's evidence retention.
    """
    order = np.argsort(-scores, kind="stable")[:limit]
    return {"top": scores[order], "top_gold": is_gold[order],
            "gold": scores[is_gold], "noise": scores[~is_gold]}


def sweep(questions: list[dict], thresholds, relative: bool = False) -> list[dict]:
    """What survives each threshold, over all questions.

    ``relative`` divides every score by the question's top-1 score first, so
    t=1.0 keeps exactly the best candidate and t=0.9 keeps whatever lies within
    10% of it.
    """
    rows = []
    for t in thresholds:
        gold_kept = gold_total = noise_kept = noise_total = survivors = empty = 0
        recall = 0.0
        for q in questions:
            scale = q["top"][0] if relative else 1.0
            kept = q["top"] / scale >= t
            gold_kept += int(np.sum(q["gold"] / scale >= t))
            gold_total += len(q["gold"])
            noise_kept += int(np.sum(kept & ~q["top_gold"]))
            noise_total += int(np.sum(~q["top_gold"]))
            survivors += int(np.sum(kept))
            empty += not kept.any()
            recall += np.sum(kept & q["top_gold"]) / len(q["gold"])
        n = len(questions)
        rows.append({"threshold": float(t), "gold_kept": gold_kept / gold_total,
                     "noise_kept": noise_kept / noise_total,
                     "survivors": survivors / n, "empty": empty / n,
                     "recall": recall / n})
    return rows


def report(questions: list[dict], limit: int) -> str:
    gold = np.concatenate([q["gold"] for q in questions])
    top_noise = np.concatenate([q["top"][~q["top_gold"]] for q in questions])
    noise = np.concatenate([q["noise"] for q in questions])
    top1 = np.array([q["top"][0] for q in questions])
    last = np.array([q["top"][-1] for q in questions])
    gold_in_top = sum(int(q["top_gold"].sum()) for q in questions)

    def stats(name, values):
        return [name, len(values), float(values.min()),
                *(float(v) for v in np.percentile(values, PERCENTILES)),
                float(values.max())]

    percentiles = markdown_table(
        ["scores", "n", "min", *(f"p{p}" for p in PERCENTILES), "max"],
        [stats("gold chunks", gold), stats(f"non-gold in top-{limit}", top_noise),
         stats("non-gold, whole corpus", noise), stats("top-1 per question", top1),
         stats(f"rank-{limit} per question", last)])

    low = np.floor(min(gold.min(), top_noise.min()) / HISTOGRAM_BIN) * HISTOGRAM_BIN
    edges = np.arange(low, top1.max() + HISTOGRAM_BIN, HISTOGRAM_BIN)
    gold_hist = np.histogram(gold, edges)[0] / len(gold)
    noise_hist = np.histogram(top_noise, edges)[0] / len(top_noise)
    histogram = markdown_table(
        ["cosine bin", "gold share", f"top-{limit} non-gold share"],
        [[f"{a:.2f}-{a + HISTOGRAM_BIN:.2f}", float(g), float(n)]
         for a, g, n in zip(edges[:-1], gold_hist, noise_hist, strict=True)])

    sweep_header = ["threshold", "gold kept", f"top-{limit} non-gold kept",
                    "survivors / question", "questions left empty",
                    f"recall@{limit} after cut"]

    def sweep_table(rows):
        return markdown_table(sweep_header, [[r["threshold"], r["gold_kept"], r["noise_kept"],
                                              r["survivors"], r["empty"], r["recall"]]
                                             for r in rows])

    return "\n\n".join([
        "# Dense cosine score distribution (retrieval_threshold)",
        (f"{len(questions)} questions, "
         f"{len(questions[0]['noise']) + len(questions[0]['gold'])} corpus chunks, exact search. "
         f"{gold_in_top} of {len(gold)} gold chunks "
         f"({gold_in_top / len(gold):.1%}) rank inside the top {limit}: no threshold "
         "can bring back the rest."),
        "## Percentiles", percentiles,
        "## Histogram", histogram,
        "## Absolute threshold", sweep_table(sweep(questions, ABSOLUTE_THRESHOLDS)),
        "## Relative threshold (score / top-1 score)",
        sweep_table(sweep(questions, RELATIVE_THRESHOLDS, relative=True)),
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="output_qasper",
                        help="holds golden_set_aligned.jsonl and chunks/")
    parser.add_argument("--limit", type=int, default=30,
                        help="result_limit: how deep the dense branch retrieves")
    parser.add_argument("--out-file", default=None,
                        help="default: <golden-dir>/retrieval_score_distribution.md")
    args = parser.parse_args()

    from embedding_manager import EmbeddingManager  # heavy import (ONNX models)

    golden_dir = Path(args.golden_dir)
    ids, texts = load_corpus(golden_dir / "chunks")
    position = {chunk_id: i for i, chunk_id in enumerate(ids)}
    embedder = EmbeddingManager()

    chunk_vectors = unit_vectors(embedder.encode_chunks(None, texts)["dense"])
    questions = []
    for record in load_jsonl(golden_dir / "golden_set_aligned.jsonl"):
        gold = [position[g] for g in gold_ids(record) if g in position]
        if not gold:
            continue
        is_gold = np.zeros(len(ids), dtype=bool)
        is_gold[gold] = True
        scores = chunk_vectors @ unit_vectors(embedder.encode_query(record["question"])["dense"])
        questions.append(question_scores(scores, is_gold, args.limit))

    out_file = Path(args.out_file or golden_dir / "retrieval_score_distribution.md")
    out_file.write_text(report(questions, args.limit))
    print(f"{len(questions)} questions -> {out_file}")


if __name__ == "__main__":
    main()
