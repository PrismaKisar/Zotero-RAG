"""One-parameter-at-a-time ablation over the PRESETS['general'] baseline.

Per the advisor's methodology (see thesis, chapter on ablation): lock every
parameter except one at the fixed baseline, sweep that one parameter, repeat
per parameter. This is deliberately NOT the adaptive per-question-type preset
in zotero_rag/question_presets.py (that's a separate, out-of-scope experiment).

Runs on the QASPER dev golden set (benchmark_out/), same dataset the frozen
alignment protocol designates for ablation (see benchmark/README.md). Scores:
- Answer F1 via benchmark/qasper_evaluator.py (official QASPER scorer). Its
  Evidence F1 is NOT used: it matches evidence by exact string equality
  against QASPER's LaTeX-derived text, which our GROBID/PDF-extracted
  paragraphs never equal verbatim (it evaluates to 0.0 regardless of config,
  confirmed empirically) - this is exactly the gap align_evidence.py's fuzzy
  overlap criterion exists to close.
- Recall@k/MRR via benchmark/retrieval_metrics.py, against the paragraphs
  aligned_chunks that benchmark/align_evidence.py already matched to a
  pdf_hash/chunk_index pair

Question paraphrasing (num_paraphrases) is fixed at 0 for every run: it is not
one of the ablated parameters, and holding it constant keeps the one-at-a-time
methodology honest (also sidesteps the Ollama dependency, unused otherwise).

Requires: GROBID + Qdrant running, and the corpus already indexed via
benchmark/index_benchmark_pdfs.py (same output_base_dir passed here).

Usage:
  python -m benchmark.ablation --work-dir benchmark_out/grobid \
      --hash-map benchmark_out/pdf_hash_map.json --out benchmark_out/ablation_results.csv
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from question_presets import PRESETS  # noqa: E402

from benchmark.qasper_evaluator import evaluate, get_answers_and_evidence
from benchmark.retrieval_metrics import aggregate

BASELINE = PRESETS["general"]

# ponytail: low/high span the range the pipeline's own question-type presets
# already use (0.35-0.45 for thresholds); section_diversity is boolean so its
# baseline (False) has a single alternative. Not exhaustively tuned.
GRID = {
    "retrieval_threshold": [0.35, 0.55],
    "rerank_threshold": [0.35, 0.55],
    "qa_score_threshold": [0.05, 0.20],
    "min_answer_words": [1, 5],
    "section_diversity": [True],
}

RECALL_K = 10


def build_configs(baseline: dict, grid: dict) -> list[dict]:
    """One baseline config, plus one per (param, value) varied one-at-a-time."""
    configs = [{"param": "baseline", "value": None, "config": dict(baseline)}]
    for param, values in grid.items():
        for value in values:
            config = dict(baseline)
            config[param] = value
            configs.append({"param": param, "value": value, "config": config})
    return configs


def sample_questions(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic subsample of ``n`` questions (or all of them if n >= len)."""
    if n >= len(questions):
        return list(questions)
    return random.Random(seed).sample(questions, n)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def load_gold_chunks(aligned_path: Path, pdf_hash_map: dict) -> dict[str, set[tuple[str, int]]]:
    """question_id -> gold set of (pdf_hash, chunk_index) retrieval targets."""
    gold = {}
    for record in load_jsonl(aligned_path):
        pdf_hash = pdf_hash_map.get(record["paper_id"])
        if pdf_hash is None:
            continue
        ids = {(pdf_hash, hit["chunk_index"])
               for hits in record["aligned_chunks"].values()
               for hit in hits}
        if ids:
            gold[record["question_id"]] = ids
    return gold


def run_config(rag, questions: list[dict], config: dict, gold_chunks: dict[str, set]):
    """Answer every question under ``config``; return (predicted, retrieval_pairs)."""
    predicted = {}
    retrieval_pairs = []
    for q in questions:
        answers = rag.answer_question(q["question"], question_type="general",
                                      overrides=config, num_paraphrases=0)
        predicted[q["question_id"]] = {
            "answer": answers[0].text if answers else "",
            "evidence": [answers[0].context] if answers else [],
        }
        gold_ids = gold_chunks.get(q["question_id"])
        if gold_ids is not None:
            ranked = sorted(rag.last_candidates, key=lambda c: c["retrieval_score"], reverse=True)
            ranked_ids = [(c["paragraph"].pdf_hash, c["paragraph"].para_idx) for c in ranked]
            retrieval_pairs.append((ranked_ids, gold_ids))
    return predicted, retrieval_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="benchmark_out")
    parser.add_argument("--hash-map", default="benchmark_out/pdf_hash_map.json")
    parser.add_argument("--work-dir", default="benchmark_out/grobid",
                        help="output_base_dir ZoteroRAG was indexed with")
    parser.add_argument("--out", default="benchmark_out/ablation_results.csv")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--sample", type=int, default=None,
                        help="subsample this many questions (default: all 218)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from zotero_rag import ZoteroRAG  # heavy import, kept out of build_configs()'s pure path

    golden_dir = Path(args.golden_dir)
    questions = load_jsonl(golden_dir / "golden_set.jsonl")
    if args.sample is not None:
        questions = sample_questions(questions, args.sample, args.seed)
    sampled_ids = {q["question_id"] for q in questions}

    gold_answers = get_answers_and_evidence(
        json.loads((golden_dir / "golden_gold.json").read_text()), text_evidence_only=True)
    gold_answers = {qid: refs for qid, refs in gold_answers.items() if qid in sampled_ids}
    pdf_hash_map = json.loads(Path(args.hash_map).read_text())
    gold_chunks = load_gold_chunks(golden_dir / "golden_set_aligned.jsonl", pdf_hash_map)

    rag = ZoteroRAG(grobid_url=args.grobid_url, qdrant_url=args.qdrant_url,
                    output_base_dir=args.work_dir)
    # ponytail: deberta-v3-large's disentangled attention has no efficient MPS
    # kernel on this hardware (measured ~2x slower than CPU at 512 tokens);
    # the reranker has no such issue, so only the QA model moves to CPU.
    rag.qa_engine.model = rag.qa_engine.model.to("cpu")
    rag.qa_engine.device = "cpu"

    rows = []
    for entry in build_configs(BASELINE, GRID):
        print(f"running: {entry['param']}={entry['value']}")
        predicted, retrieval_pairs = run_config(rag, questions, entry["config"], gold_chunks)
        scores = evaluate(gold_answers, predicted)
        retrieval_scores = aggregate(retrieval_pairs, k=RECALL_K)
        rows.append({
            "param": entry["param"],
            "value": entry["value"],
            "answer_f1": scores["Answer F1"],
            f"recall@{RECALL_K}": retrieval_scores[f"recall@{RECALL_K}"],
            "mrr": retrieval_scores["mrr"],
        })

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
