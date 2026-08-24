"""One-parameter-at-a-time ablation over the PRESETS['general'] baseline.

Per the advisor's methodology (see thesis, chapter on ablation): lock every
parameter except one at the fixed baseline, sweep that one parameter, repeat
per parameter. This is deliberately NOT the adaptive per-question-type preset
in zotero_rag/question_presets.py (that's a separate, out-of-scope experiment).

Runs on the QASPER dev golden set (output_qasper/), same dataset the frozen
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

Three reference rows bracket every configuration, and together they split the
end-to-end score into the three things that can go wrong:

- ``baseline``: the preset as shipped, retrieving over the whole corpus.
- ``oracle_paper``: identical, except retrieval is scoped to the paper the
  question was asked about. baseline -> oracle_paper is the cost of document
  selection; QASPER is defined as single-paper QA, so searching all 99 papers is
  a harder task than the dataset poses and that surcharge has to be visible
  rather than folded into the retrieval number.
- ``oracle_context``: the reader fed the *gold* chunks directly, bypassing
  retrieval. oracle_paper -> oracle_context is the cost of passage selection,
  and oracle_context itself is the reader's ceiling.

Three axes are applied by the harness rather than by the pipeline config, since
they are not question-preset fields (see HARNESS_PARAMS):

- ``num_paraphrases``: question expansion is a kwarg of answer_question, and it
  costs a T5 generation plus one extra retrieval per variation. Baseline 0.
- ``qa_model``: swaps the extractive reader. deberta-v3-large is 24 layers
  against 12 for the two base-sized alternatives, so this axis separates reader
  capacity from retrieval quality.
- ``reader``: extractive vs generative. QASPER answers are frequently
  abstractive, which no span extractor can reach; the generative arm needs
  Ollama (``ollama pull llama3.2:3b``) and is skipped with a clear error if it
  is not there.

Requires: GROBID + Qdrant running, and the corpus already indexed via
benchmark/index_pdfs.py (same output_base_dir passed here). The ``reader``
axis additionally requires Ollama.

Usage:
  python -m benchmark.ablation --work-dir output_qasper/grobid \
      --hash-map output_qasper/pdf_hash_map.json --out-file output_qasper/ablation_results.csv
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

# appended, never prepended: the repo root must keep priority over this directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from question_presets import PRESETS

from benchmark.qasper_evaluator import (
    get_answers_and_evidence,
    normalize_answer,
    token_f1_score,
)
from benchmark.retrieval_metrics import (
    per_question_scores,
    summarize,
)
from benchmark.stratify import stratify, to_markdown

BASELINE = PRESETS["general"]

# ponytail: threshold low/high bracket the preset defaults, but each pair is set
# on its *own* stage's score scale, measured rather than guessed - a pair outside
# the observed range is a no-op row that reads as a null result. rerank_threshold
# is on the cross-encoder's probability scale, where gold evidence sits around
# p50 0.007. retrieval_threshold is dense cosine, where the top-30 measured
# 0.63-0.82: the former 0.35/0.55 pair never cut a single candidate, so 0.65/0.72
# straddle the observed median instead. qa_score_threshold spans 0.006-0.97 since
# span scoring became a real softmax product. The ranking knobs get their full
# discrete range (3 modes, reranker on/off) plus a narrow/wide result_limit. Not
# exhaustively tuned - the point is which stage moves the needle, not the optimum.
GRID = {
    "retrieval_mode": ["dense", "sparse"],
    "rerank_enabled": [False],
    "result_limit": [10, 60],
    "retrieval_threshold": [0.65, 0.72],
    "rerank_threshold": [0.0002, 0.01],
    "qa_score_threshold": [0.05, 0.20],
    "min_answer_words": [1, 5],
    "section_diversity_enabled": [True],
    "num_paraphrases": [2],
    "qa_model": ["deepset/deberta-v3-base-squad2", "deepset/roberta-base-squad2"],
    "reader": ["generative"],
}

# Axes the harness applies itself: not question-preset fields, so they are split
# out before the merged config reaches answer_question. Absent from BASELINE on
# purpose - each default is read off the pipeline as built, so there is no second
# copy of the model name to drift.
HARNESS_PARAMS = ("num_paraphrases", "qa_model", "reader")

# The protocol names recall@1 the primary retrieval metric, so the sweep can
# not report k=10 alone: at k=10 a config that merely drags gold evidence from
# rank 9 to rank 2 looks identical to one that does nothing, and that is exactly
# the movement a reranker is bought for.
RECALL_KS = (1, 3, 5, 10)
RECALL_K = max(RECALL_KS)  # the cut the attributed-evidence metrics score at

# Means written to the CSV; the per-question JSONL keeps everything.
CSV_METRICS = (
    ["answer_f1", "answer_em"]
    + [f"recall@{k}" for k in RECALL_KS]
    + [f"precision@{k}" for k in RECALL_KS]
    + ["mrr", "evidence_precision", "evidence_recall", "evidence_f1",
       f"recall@{RECALL_K}_reranked", "mrr_reranked"]
)
CI_METRICS = ["answer_f1", "recall@1", f"recall@{RECALL_K}", "evidence_f1"]


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
               {"param": "oracle_paper", "value": None, "config": dict(baseline)},
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


def answer_exact_match(predicted_answer: str, references: list[dict]) -> float:
    """1.0 if the prediction equals any reference after QASPER normalisation.

    Reported next to token F1 because the two disagree in a way that matters
    here: a span reader that returns the right sentence around a one-word answer
    earns most of the F1 and none of the EM, so EM is what separates "found the
    answer" from "found the neighbourhood of the answer".
    """
    if not references:
        return 0.0
    predicted = normalize_answer(predicted_answer)
    return float(any(predicted == normalize_answer(ref["answer"]) for ref in references))


def attributed_ids(answers, reranked) -> set[tuple[str, int]]:
    """Chunk ids the system surfaced as evidence for its answers.

    Answer carries no chunk_index, so answers are mapped back to the chunk
    they were extracted from by context text - the same text the highlighter
    resolves coordinates against.
    """
    by_text = {c.chunk.text: (c.chunk.pdf_hash, c.chunk.chunk_index)
               for c in reranked}
    return {by_text[a.context] for a in answers if a.context in by_text}


TRACE_DEPTH = 30  # deepest k anyone can recompute offline; the sweep cuts at 10


def retrieval_trace(ranked_ids, reranked_ids, predicted_ids, gold_ids) -> dict:
    """The ranked lists themselves, so the sweep never has to be re-run for a metric.

    The first campaign could not report recall@1 - the protocol's own primary
    metric - because only the k=10 *scores* were kept, and a mean cannot be
    un-aggregated. Storing the ids costs a few hundred KB and makes every
    rank-based metric (any k, MAP, nDCG) recomputable from the dump.

    ``gold_paper`` is what separates the two ways retrieval fails: picking the
    wrong paper out of the corpus, or the wrong chunk inside the right one.
    """
    gold_papers = {pdf_hash for pdf_hash, _ in gold_ids}
    return {
        "ranked_ids": [list(i) for i in ranked_ids[:TRACE_DEPTH]],
        "reranked_ids": [list(i) for i in reranked_ids[:TRACE_DEPTH]],
        "predicted_ids": [list(i) for i in sorted(predicted_ids)],
        "gold_ids": [list(i) for i in sorted(gold_ids)],
        "gold_paper_in_top10": any(h in gold_papers for h, _ in ranked_ids[:10]),
    }


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
    predicted_ids = attributed_ids(answers, rag.last_reranked)

    row = {"answer_f1": f1,
           "answer_em": answer_exact_match(predicted_answer,
                                           gold_answers.get(question["question_id"], []))}
    row.update(per_question_scores(ranked_ids, gold_ids, RECALL_K,
                                   predicted_ids=predicted_ids))
    for k in RECALL_KS:
        if k == RECALL_K:
            continue
        scores = per_question_scores(ranked_ids, gold_ids, k)
        row[f"recall@{k}"] = scores[f"recall@{k}"]
        row[f"precision@{k}"] = scores[f"precision@{k}"]
    reranked_scores = per_question_scores(reranked_ids, gold_ids, RECALL_K)
    row[f"recall@{RECALL_K}_reranked"] = reranked_scores[f"recall@{RECALL_K}"]
    row["mrr_reranked"] = reranked_scores["mrr"]
    row["record"] = dict(question, answer_type=answer_type, **retrieval_trace(
        ranked_ids, reranked_ids, predicted_ids, gold_ids))
    return row


def split_harness_params(config: dict) -> tuple[dict, dict]:
    """Separate the harness-applied axes from the pipeline's own config.

    Returns:
        ``(pipeline_config, harness)``; the pipeline half is what answer_question
        merges over the preset, and never carries a key no preset defines.
    """
    harness = {key: config[key] for key in HARNESS_PARAMS if key in config}
    pipeline_config = {key: value for key, value in config.items() if key not in HARNESS_PARAMS}
    return pipeline_config, harness


def reader_key(reader) -> str:
    """Identity of the loaded reader, for skipping redundant reloads."""
    return getattr(reader, "reader_kind", None) or reader.model_name


def wanted_reader(harness: dict, baseline_model: str) -> str:
    """Which reader this config asks for, in ``reader_key`` terms."""
    if harness.get("reader", "extractive") == "generative":
        return "generative"
    return harness.get("qa_model", baseline_model)


def apply_reader(rag, baseline_engine, harness: dict, ollama_url: str) -> None:
    """Point ``rag.qa_engine`` at the reader this config asks for.

    Reloading is skipped when the current reader already matches, so the many
    configs that leave both axes at the baseline share one model load instead of
    paying for one each.
    """
    wanted = wanted_reader(harness, baseline_engine.model_name)
    if reader_key(rag.qa_engine) == wanted:
        return

    import torch
    from generative_reader import GenerativeReader
    from qa_engine import QAEngine

    qa_model = harness.get("qa_model", baseline_engine.model_name)
    if wanted == "generative":
        rag.qa_engine = GenerativeReader(ollama_url=ollama_url)
    elif qa_model == baseline_engine.model_name:
        rag.qa_engine = baseline_engine
    else:
        # The paraphraser belongs to the baseline engine; an alternative reader
        # never needs it, since one-at-a-time never moves both axes together.
        rag.qa_engine = QAEngine(model_name=qa_model, enable_question_expansion=False)

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_config(rag, questions: list[dict], config: dict, gold_chunks: dict,
               gold_answers: dict, scope_to_gold_paper: bool = False) -> list[dict]:
    """Answer every question under ``config``; return one metric row per question.

    With ``scope_to_gold_paper``, retrieval is filtered to the paper the question
    was written about. The full pipeline still runs - the same ranking, the same
    reranker, the same reader - so the gap against the unscoped baseline is the
    cost of having to find the right document among the other 98, isolated from
    everything else. That gap is not visible in the baseline/oracle pair, where a
    failure to rank the right paper and a failure to rank the right paragraph
    land in the same number.
    """
    pipeline_config, harness = split_harness_params(config)
    rows = []
    for q in questions:
        pdf_hashes = None
        if scope_to_gold_paper:
            pdf_hashes = sorted({h for h, _ in gold_chunks[q["question_id"]]})
        answers = rag.answer_question(
            q["question"], question_type="general", overrides=pipeline_config,
            num_paraphrases=harness.get("num_paraphrases", 0),
            pdf_hashes=pdf_hashes)
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

    config, _ = split_harness_params(config)
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
            predicted = answers[0].text if answers else ""
            references = gold_answers.get(q["question_id"], [])
            f1, answer_type = answer_f1(predicted, references)
            rows.append({"answer_f1": f1,
                         "answer_em": answer_exact_match(predicted, references),
                         "record": dict(q, answer_type=answer_type,
                                        gold_ids=[list(i) for i in sorted(gold_ids)])})
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


def config_label(param: str, value) -> str:
    """Filename-safe label for one grid entry.

    Values reach here straight from GRID, and some are model ids that carry a
    slash ("deepset/roberta-base-squad2"). Left raw, that turns the per-question
    filename into a subdirectory - and a value containing ".." would write
    outside the output tree entirely.
    """
    label = param if value is None else f"{param}_{value}"
    return label.replace("/", "_").replace("..", "_")


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
    parser.add_argument("--golden-dir", default="output_qasper")
    parser.add_argument("--hash-map", default="output_qasper/pdf_hash_map.json")
    parser.add_argument("--work-dir", default="output_qasper/grobid",
                        help="output_base_dir ZoteroRAG was indexed with")
    parser.add_argument("--out-file", default="output_qasper/ablation_results.csv")
    parser.add_argument("--strata-file", default="output_qasper/ablation_by_stratum.md",
                        help="stratified breakdown of the baseline run")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="only used by the generative reader axis")
    parser.add_argument("--collection-suffix", default="_qasper",
                        help="the Qdrant collection this corpus was indexed into")
    parser.add_argument("--sample", type=int, default=None,
                        help="subsample this many questions (default: all 218)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from zotero_rag.pipeline import (
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
    # The reader used to be forced onto the CPU here, from a measurement taken
    # when it still ran in float32 on MPS. In float16 that is backwards: over 24
    # chunks at batch 8 the medians are mps/fp16 3.45s, cpu/fp32 4.59s, and
    # cpu/fp16 34.48s - which is what moving the half-precision model to the CPU
    # actually produced. The device now stays wherever QAEngine put it.
    baseline_engine = rag.qa_engine

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
        _, harness = split_harness_params(entry["config"])
        try:
            apply_reader(rag, baseline_engine, harness, args.ollama_url)
        except RuntimeError as exc:
            # Nothing is written, so this config is picked up on the next run
            # once its dependency is there; the rest of the sweep still lands.
            print(f"skipped: {entry['param']}={entry['value']} - {exc}")
            continue
        if entry["param"] == "oracle_context":
            rows = run_oracle(rag, questions, entry["config"], gold_chunks, gold_answers)
        else:
            rows = run_config(rag, questions, entry["config"], gold_chunks, gold_answers,
                              scope_to_gold_paper=entry["param"] == "oracle_paper")
        if not rows:
            print(f"skipped: {entry['param']}={entry['value']} scored no questions")
            continue

        label = config_label(entry["param"], entry["value"])
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
