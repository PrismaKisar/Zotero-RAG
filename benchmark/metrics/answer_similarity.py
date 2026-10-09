"""Semantic answer similarity, reported beside token F1 and never instead of it.

Token F1 punishes a correct answer worded differently, and the reader answers in
prose against references that are often noun phrases. This scores the same
predicted answer by the cosine between sentence embeddings, best over the
annotators' references, as token F1 takes its best.

It is a secondary metric, and deterministic on purpose: an LLM judge would not
be. The raw cosine is rescaled against a baseline measured on this very data -
each prediction against the references of a *different* question - because raw
cosines between unrelated sentences sit far above zero, and an unrescaled mean
of 0.6 would read as a reader that is mostly right. Rescaled, 0 is what an
unrelated answer scores and 1 an identical one.

Reads the per-question dumps ablation.py writes, which carry the predicted
answer, so no campaign is re-run. Only sets with reference answers apply: QASA
has none.

Usage:
  python -m benchmark.metrics.answer_similarity --per-question runs/dev/ablation/qasper/per_question \\
      --gold data/dev/qasper/golden_gold.json --out-file runs/dev/ablation/qasper/answer_similarity.md
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.metrics.qasper_evaluator import get_answers_and_evidence
from benchmark.metrics.reports import load_jsonl, markdown_table
from benchmark.metrics.retrieval_metrics import bootstrap_ci, format_delta

# A general-purpose sentence model, deliberately not the retrieval model: the
# metric must not share the embedding space the system ranked with.
SIMILARITY_MODEL = "sentence-transformers/all-mpnet-base-v2"


def best_cosines(predictions: list[str], references: list[list[str]], encode) -> np.ndarray:
    """Per question, the best cosine between its prediction and any reference.

    ``encode`` maps a list of texts to L2-normalised row vectors. An empty
    prediction scores 0.0, as it does under token F1.
    """
    texts = sorted({t for t in predictions if t} | {r for refs in references for r in refs})
    if not texts:
        return np.zeros(len(predictions))
    vectors = dict(zip(texts, encode(texts), strict=True))
    return np.array([
        max((float(vectors[p] @ vectors[r]) for r in refs), default=0.0) if p else 0.0
        for p, refs in zip(predictions, references, strict=True)])


def unrelated_baseline(predictions: list[str], references: list[list[str]], encode) -> float:
    """Mean best cosine of each prediction against the next question's references.

    A shift by one is a derangement whenever there are two or more questions,
    so no prediction is scored against its own references.
    """
    if len(predictions) < 2:
        raise ValueError("the baseline needs at least two questions")
    shifted = references[1:] + references[:1]
    answered = [(p, refs) for p, refs in zip(predictions, shifted, strict=True) if p]
    if not answered:
        return 0.0
    return float(best_cosines(*map(list, zip(*answered, strict=True)), encode).mean())


def rescale(raw: np.ndarray, baseline: float) -> np.ndarray:
    """``(raw - baseline) / (1 - baseline)``: unrelated reads 0, identical 1."""
    return (raw - baseline) / (1.0 - baseline)


def score_config(rows: list[dict], gold: dict, encode, baseline: float) -> dict[str, float]:
    """question_id -> rescaled similarity, for the rows that carry a prediction."""
    rows = [r for r in rows if "predicted_answer" in r["record"]
            and r["record"]["question_id"] in gold]
    predictions = [r["record"]["predicted_answer"] for r in rows]
    references = [[ref["answer"] for ref in gold[r["record"]["question_id"]]] for r in rows]
    scores = rescale(best_cosines(predictions, references, encode), baseline)
    return {r["record"]["question_id"]: float(s) for r, s in zip(rows, scores, strict=True)}


def report(per_question_dir: Path, gold: dict, encode, reference: str = "baseline") -> str:
    """One row per config: token F1 and rescaled similarity, means and paired deltas."""
    configs = {p.stem: load_jsonl(p) for p in sorted(per_question_dir.glob("*.jsonl"))}
    if reference not in configs:
        raise SystemExit(f"no {reference}.jsonl in {per_question_dir}")

    # One baseline for every config, measured on the reference arm, so the
    # rescaling is a fixed affine map and paired deltas stay comparable.
    ref_rows = [r for r in configs[reference] if "predicted_answer" in r["record"]
                and r["record"]["question_id"] in gold]
    if not ref_rows:
        raise SystemExit(f"{reference}.jsonl carries no predicted_answer: "
                         "re-run it with the current ablation.py")
    baseline = unrelated_baseline(
        [r["record"]["predicted_answer"] for r in ref_rows],
        [[ref["answer"] for ref in gold[r["record"]["question_id"]]] for r in ref_rows], encode)

    similarity = {name: score_config(rows, gold, encode, baseline)
                  for name, rows in configs.items()}
    f1 = {name: {r["record"]["question_id"]: r["answer_f1"] for r in rows if "answer_f1" in r}
          for name, rows in configs.items()}

    table = []
    for name in configs:
        sim = similarity[name]
        if not sim:
            continue
        shared = sorted(set(sim) & set(similarity[reference]))
        low, high = bootstrap_ci(list(sim.values()))
        row = [name, len(sim), float(np.mean(list(f1[name].values()) or [0.0])),
               float(np.mean(list(sim.values()))), f"[{low:.3f}, {high:.3f}]"]
        if name == reference:
            row += ["", ""]
        else:
            row += [format_delta([f1[name][q] - f1[reference][q] for q in shared
                                  if q in f1[name] and q in f1[reference]] or [0.0]),
                    format_delta([sim[q] - similarity[reference][q] for q in shared] or [0.0])]
        table.append(row)

    header = ["config", "n", "token F1", "similarity", "similarity CI",
              f"Δ F1 vs {reference}", f"Δ similarity vs {reference}"]
    return (f"Secondary metric: cosine similarity ({SIMILARITY_MODEL}), best over "
            f"references, rescaled against an unrelated-answer baseline of "
            f"{baseline:.3f}. Token F1 stays the primary answer metric.\n\n"
            + markdown_table(header, table) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-question", required=True)
    parser.add_argument("--gold", required=True, help="golden_gold.json of the set")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--reference", default="baseline")
    args = parser.parse_args()

    from sentence_transformers import (
        SentenceTransformer,  # heavy, kept off the pure path
    )

    model = SentenceTransformer(SIMILARITY_MODEL)
    gold = get_answers_and_evidence(json.loads(Path(args.gold).read_text()),
                                    text_evidence_only=True)
    text = report(Path(args.per_question), gold,
                  lambda texts: model.encode(texts, normalize_embeddings=True),
                  args.reference)
    Path(args.out_file).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
