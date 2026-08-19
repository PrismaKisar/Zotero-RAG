"""One-parameter-at-a-time ablation over the PRESETS['general'] baseline.

Per the advisor's methodology (see thesis, chapter on ablation): lock every
parameter except one at the fixed baseline, sweep that one parameter, repeat
per parameter. This is deliberately NOT the adaptive per-question-type preset
in zotero_rag/question_presets.py (that's a separate, out-of-scope experiment).

Runs on the QASPER dev golden set (benchmark_out_qasper/), same dataset the frozen
alignment protocol designates for ablation (see benchmark/README.md). Scores:

- Answer F1, per question, via qasper_evaluator.token_f1_score (the official
  QASPER scorer's own token F1, max over annotator references).
- Recall@k / precision@k / MRR via benchmark/retrieval_metrics.py, against the
  (pdf_hash, chunk_index) pairs benchmark/align_evidence.py matched.
- Evidence precision/recall/F1 over the chunks the system actually attributed
  (the ones the highlighter would mark). This replaces QASPER's official
  Evidence F1, which matches evidence by exact string equality against
  LaTeX-derived text that our GROBID/PDF-extracted chunks never equal
  verbatim - it evaluates to 0.0 regardless of config, confirmed empirically.

The grid ablates parameters that *reorder* results, not just filter them.
Thresholds alone cannot move recall@k (which scores the ranked list before the
cut), so a threshold-only grid is unfalsifiable by construction; thresholds are
kept in the grid but are now read through precision@k and evidence F1, which do
see the cut.

Two reference rows bracket every configuration:
- ``baseline``: the preset as shipped.
- ``oracle_context``: the reader fed the *gold* chunks directly, bypassing
  retrieval. Its Answer F1 is the reader's ceiling, which is what makes a low
  end-to-end score attributable to retrieval rather than to extraction.

Question paraphrasing (num_paraphrases) is fixed at 0 for every run: it is not
one of the ablated parameters, and holding it constant keeps the one-at-a-time
methodology honest (also sidesteps the Ollama dependency, unused otherwise).

Requires: GROBID + Qdrant running, and the corpus already indexed via
benchmark/index_pdfs.py (same output_base_dir passed here).

Usage:
  python -m benchmark.ablation --work-dir benchmark_out_qasper/grobid \
      --hash-map benchmark_out_qasper/pdf_hash_map.json --out-file benchmark_out_qasper/ablation_results.csv
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

# appended (not inserted): prepending lets zotero_rag.py shadow the zotero_rag package
sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from question_presets import PRESETS

from benchmark.qasper_evaluator import get_answers_and_evidence, token_f1_score
from benchmark.retrieval_metrics import (
    per_question_scores,
    summarize,
)
from benchmark.stratify import stratify, to_markdown

BASELINE = PRESETS["general"]

# ponytail: threshold low/high span the range the pipeline's own question-type
# presets already use (0.35-0.45); the ranking knobs get their full discrete
# range (3 modes, reranker on/off) plus a narrow/wide result_limit. Not
# exhaustively tuned - the point is which stage moves the needle, not the optimum.
GRID = {
    "retrieval_mode": ["dense", "sparse"],
    "rerank_enabled": [False],
    "result_limit": [10, 60],
    "retrieval_threshold": [0.35, 0.55],
    "rerank_threshold": [0.35, 0.55],
    "qa_score_threshold": [0.05, 0.20],
    "min_answer_words": [1, 5],
    "section_diversity": [True],
}

RECALL_K = 10

# Means written to the CSV; the per-question JSONL keeps everything.
CSV_METRICS = [
    "answer_f1", f"recall@{RECALL_K}", f"precision@{RECALL_K}", "mrr",
    "evidence_precision", "evidence_recall", "evidence_f1",
    f"recall@{RECALL_K}_reranked", "mrr_reranked",
]
CI_METRICS = ["answer_f1", f"recall@{RECALL_K}", "evidence_f1"]


def build_fieldnames() -> list[str]:
    fields = ["param", "value", "n_questions"]
    for metric in CSV_METRICS:
        fields.append(metric)
        if metric in CI_METRICS:
            fields += [f"{metric}_ci_low", f"{metric}_ci_high"]
    return fields


def check_schema_compatible(out_path: Path, fieldnames: list[str]) -> None:
    """Raise if ``out_path`` already exists with a different column set.

    Guards against DictWriter silently appending mismatched-width rows onto
    an older-schema CSV when resuming after ablation.py's columns changed.
    """
    if not out_path.exists():
        return
    existing_header = out_path.open(newline="").readline().strip().split(",")
    if existing_header != fieldnames:
        raise SystemExit(
            f"{out_path} has columns {existing_header}, expected {fieldnames}. "
            "Resuming would corrupt the file - migrate or rename it first.")


def load_completed_configs(out_path: Path) -> set[tuple[str, str]]:
    """(param, value) pairs already written to ``out_path``, for resuming a run."""
    if not out_path.exists():
        return set()
    with out_path.open(newline="") as f:
        return {(row["param"], row["value"]) for row in csv.DictReader(f)}


def build_configs(baseline: dict, grid: dict) -> list[dict]:
    """Baseline, oracle-context reference, then one config per (param, value)."""
    configs = [{"param": "baseline", "value": None, "config": dict(baseline)},
               {"param": "oracle_context", "value": None, "config": dict(baseline)}]
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


def answer_f1(predicted_answer: str, references: list[dict]) -> tuple[float, str]:
    """Best token F1 across annotator references, plus that reference's type."""
    if not references:
        return 0.0, "none"
    scored = [(token_f1_score(predicted_answer, ref["answer"]), ref["type"])
              for ref in references]
    return max(scored, key=lambda x: x[0])


def attributed_ids(answers, reranked) -> set[tuple[str, int]]:
    """Chunk ids the system surfaced as evidence for its answers.

    Answer carries no chunk_index, so answers are mapped back to the chunk
    they were extracted from by context text - the same text the highlighter
    resolves coordinates against.
    """
    by_text = {c.chunk.text: (c.chunk.pdf_hash, c.chunk.chunk_index)
               for c in reranked}
    return {by_text[a.context] for a in answers if a.context in by_text}


def score_question(question: dict, answers, rag, gold_chunks: dict,
                   gold_answers: dict) -> dict | None:
    """Per-question metric row, or None if the question has no gold chunks."""
    gold_ids = gold_chunks.get(question["question_id"])
    if gold_ids is None:
        return None

    predicted_answer = answers[0].text if answers else ""
    f1, answer_type = answer_f1(predicted_answer,
                                gold_answers.get(question["question_id"], []))

    ranked = sorted(rag.last_candidates, key=lambda c: c["retrieval_score"], reverse=True)
    ranked_ids = [(c["chunk"].pdf_hash, c["chunk"].chunk_index) for c in ranked]
    reranked_ids = [(c.chunk.pdf_hash, c.chunk.chunk_index) for c in rag.last_reranked]

    row = {"answer_f1": f1}
    row.update(per_question_scores(
        ranked_ids, gold_ids, RECALL_K,
        predicted_ids=attributed_ids(answers, rag.last_reranked)))
    reranked_scores = per_question_scores(reranked_ids, gold_ids, RECALL_K)
    row[f"recall@{RECALL_K}_reranked"] = reranked_scores[f"recall@{RECALL_K}"]
    row["mrr_reranked"] = reranked_scores["mrr"]
    row["record"] = dict(question, answer_type=answer_type)
    return row


def run_config(rag, questions: list[dict], config: dict, gold_chunks: dict,
               gold_answers: dict) -> list[dict]:
    """Answer every question under ``config``; return one metric row per question."""
    rows = []
    for q in questions:
        answers = rag.answer_question(q["question"], question_type="general",
                                      overrides=config, num_paraphrases=0)
        row = score_question(q, answers, rag, gold_chunks, gold_answers)
        if row is not None:
            rows.append(row)
    return rows


def run_oracle(rag, questions: list[dict], config: dict, gold_chunks: dict,
               gold_answers: dict) -> list[dict]:
    """Answer every question from the gold chunks, bypassing retrieval entirely.

    Retrieval metrics are perfect by construction and therefore omitted: the
    only meaningful number here is Answer F1, the reader's ceiling.
    """
    from models import RerankedChunk

    rows = []
    rag.qdrant_manager.open_connection()
    try:
        for q in questions:
            gold_ids = gold_chunks.get(q["question_id"])
            if gold_ids is None:
                continue
            chunks = rag.qdrant_manager.fetch_chunks(sorted(gold_ids))
            if not chunks:
                continue
            candidates = [RerankedChunk(chunk=p, retrieval_score=1.0, rerank_score=1.0)
                          for p in chunks]
            answers = rag.qa_engine.extract_answers(
                q["question"], candidates, config, question_variations=[q["question"]])
            f1, answer_type = answer_f1(answers[0].text if answers else "",
                                        gold_answers.get(q["question_id"], []))
            rows.append({"answer_f1": f1, "record": dict(q, answer_type=answer_type)})
    finally:
        rag.qdrant_manager.close_connection()
    return rows


def csv_row(entry: dict, rows: list[dict]) -> dict:
    """Mean of every CSV metric present in ``rows``, with CIs where configured."""
    summary = summarize([{k: v for k, v in r.items() if k != "record"} for r in rows],
                        with_ci=True)
    out = {"param": entry["param"], "value": entry["value"],
           "n_questions": summary["n_questions"]}
    for metric in CSV_METRICS:
        out[metric] = summary.get(metric, "")
        if metric in CI_METRICS:
            out[f"{metric}_ci_low"] = summary.get(f"{metric}_ci_low", "")
            out[f"{metric}_ci_high"] = summary.get(f"{metric}_ci_high", "")
    return out


def write_per_question(rows: list[dict], path: Path) -> None:
    """Dump per-question scores for traceability and offline re-stratification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def check_corpus_indexed(rag) -> int:
    """Fail fast unless the collection being queried actually holds chunks.

    open_connection() *creates* a missing collection, so a wrong
    --collection-suffix does not raise: the run quietly scores an empty index
    and every metric comes out 0.0, indistinguishable from a real result.
    """
    rag.qdrant_manager.open_connection()
    try:
        name = rag.qdrant_manager.chunk_collection
        if not rag.qdrant_manager.client.collection_exists(name):
            raise SystemExit(f"Qdrant collection {name!r} does not exist - "
                             "index the corpus first, or check --qdrant-collection-suffix.")
        count = rag.qdrant_manager.client.count(name).count
        if not count:
            raise SystemExit(f"Qdrant collection {name!r} is empty.")
        return count
    finally:
        rag.qdrant_manager.close_connection()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="benchmark_out_qasper")
    parser.add_argument("--hash-map", default="benchmark_out_qasper/pdf_hash_map.json")
    parser.add_argument("--work-dir", default="benchmark_out_qasper/grobid",
                        help="output_base_dir ZoteroRAG was indexed with")
    parser.add_argument("--out-file", default="benchmark_out_qasper/ablation_results.csv")
    parser.add_argument("--strata-file", default="benchmark_out_qasper/ablation_by_stratum.md",
                        help="stratified breakdown of the baseline run")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection-suffix", default="_qasper",
                        help="the Qdrant collection this corpus was indexed into")
    parser.add_argument("--sample", type=int, default=None,
                        help="subsample this many questions (default: all 218)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from zotero_rag.zotero_rag import (
        ZoteroRAG,  # heavy import, kept out of build_configs()'s pure path
    )

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
    # Only aligned questions are scorable, so every metric shares one denominator.
    questions = [q for q in questions if q["question_id"] in gold_chunks]
    print(f"scoring {len(questions)} aligned questions")

    rag = ZoteroRAG(grobid_url=args.grobid_url, qdrant_url=args.qdrant_url,
                    output_base_dir=args.work_dir,
                    qdrant_collection_suffix=args.collection_suffix)
    print(f"querying {rag.qdrant_manager.chunk_collection}: "
          f"{check_corpus_indexed(rag)} chunks")
    # ponytail: deberta-v3-large's disentangled attention has no efficient MPS
    # kernel on this hardware (measured ~2x slower than CPU at 512 tokens);
    # the reranker has no such issue, so only the QA model moves to CPU.
    rag.qa_engine.model = rag.qa_engine.model.to("cpu")
    rag.qa_engine.device = "cpu"

    out_path = Path(args.out_file)
    fieldnames = build_fieldnames()
    check_schema_compatible(out_path, fieldnames)
    done = load_completed_configs(out_path)
    if done:
        print(f"resuming: {len(done)} configs already in {out_path}")

    for entry in build_configs(BASELINE, GRID):
        key = (entry["param"], "" if entry["value"] is None else str(entry["value"]))
        if key in done:
            continue
        print(f"running: {entry['param']}={entry['value']}")
        runner = run_oracle if entry["param"] == "oracle_context" else run_config
        rows = runner(rag, questions, entry["config"], gold_chunks, gold_answers)
        if not rows:
            print(f"skipped: {entry['param']}={entry['value']} scored no questions")
            continue

        label = entry["param"] if entry["value"] is None else f"{entry['param']}_{entry['value']}"
        write_per_question(rows, out_path.parent / "per_question" / f"{label}.jsonl")
        if entry["param"] == "baseline":
            Path(args.strata_file).write_text(to_markdown(
                {"QASPER": stratify(rows)},
                [m for m in CSV_METRICS if m in rows[0]]))

        write_header = not out_path.exists()
        with out_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(csv_row(entry, rows))
        print(f"wrote: {entry['param']}={entry['value']} -> {out_path}")


if __name__ == "__main__":
    main()
