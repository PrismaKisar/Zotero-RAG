"""One-parameter-at-a-time ablation over the PRESETS['general'] baseline.

Per the advisor's methodology (see thesis, chapter on ablation): lock every
parameter except one at the fixed baseline, sweep that one parameter, repeat
per parameter. This is deliberately NOT the adaptive per-question-type preset
in zotero_rag/question_presets.py (that's a separate, out-of-scope experiment).

Runs on either golden set. QASPER (output_qasper/) is the one the frozen
alignment protocol designates for the ablation proper; QASA (output_qasa/)
carries evidence annotations but no short answers, so it scores retrieval and
attribution only and the answer columns are dropped rather than zeroed - the
switch is whether golden_gold.json is present. The two are never pooled: see
benchmark/compare_datasets.py for how far apart their question populations are.
Scores:

- Answer F1, per question, via qasper_evaluator.token_f1_score (the official
  QASPER scorer's own token F1, max over annotator references). QASPER only.
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

Wall-clock latency is recorded per question, end to end and broken down by
pipeline stage (LATENCY_STAGES). The breakdown is what makes the number
actionable: a slow retriever and a slow reader produce the same total and take
opposite fixes. The reader cannot be read off the oracle rows instead - those
hand it only the gold chunks, roughly a fifteenth of what the pipeline hands it,
and the generative reader's cost is strongly sublinear in context size.

The CSV carries the mean, which is the wrong summary for a skewed quantity; the
per-question JSONL carries every value, so median and tail are recomputable
without re-running anything. Model loading happens between configs, outside the
timer, but the first call of a config still warms caches - which is the other
reason to read the median.

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
- ``oracle_context_generative``: the same ceiling for the generative reader.
  Without it the extractive ceiling reads as a property of the system when it
  is a property of span extraction, which is the distinction the whole reader
  axis exists to draw.

Six axes are applied by the harness rather than by the pipeline config, since
they are not question-preset fields (see HARNESS_PARAMS):

- ``num_paraphrases``: question expansion is a kwarg of answer_question, and it
  costs a T5 generation plus one extra retrieval per variation. Baseline 0.
- ``qa_model``: swaps the extractive reader. deberta-v3-large is 24 layers
  against 12 for the two base-sized alternatives, so this axis separates reader
  capacity from retrieval quality.
- ``reader``: extractive vs generative. QASPER answers are frequently
  abstractive, which no span extractor can reach; the generative arm needs
  Ollama (``ollama pull qwen3.5:2b``) and is skipped with a clear error if it
  is not there.
- ``citation_quote``: makes the generative reader reproduce the sentence it
  used and looks that sentence up in the chunk, so the citation becomes
  checkable rather than asserted. ``lenient`` narrows the highlight when the
  lookup succeeds and falls back to the whole chunk when it does not;
  ``strict`` discards the citation instead. ``quote_match_rate`` reports how
  often the lookup succeeded, and is absent for readers never asked to quote.
- ``answer_style``: how long the generative reader is asked to answer. Exact
  Match is 0.000 under every configuration measured so far, and a two-sentence
  answer cannot match a noun-phrase reference however correct it is, so the
  prompt rule is an axis in its own right rather than a fixed part of the
  reader.
- ``max_context_chunks``: how many top-ranked chunks reach the generative
  prompt. The extractive reader truncates nothing, so the two readers were
  never compared at equal context; this axis is what separates a difference in
  model from a difference in how much each was shown.

The last three imply the generative reader and are ignored by the extractive
one, which has no prompt and no context limit of its own.

Requires: GROBID + Qdrant running, and the corpus already indexed via
benchmark/index_pdfs.py (same output_base_dir passed here). The ``reader``
axis additionally requires Ollama.

Usage:
  python -m benchmark.ablation --work-dir output_qasper/grobid \
      --hash-map output_qasper/pdf_hash_map.json --out-file output_qasper/ablation_results.csv

  python -m benchmark.ablation --golden-dir output_qasa --work-dir output_qasa/grobid \
      --hash-map output_qasa/pdf_hash_map.json --out-file output_qasa/ablation_results.csv \
      --strata-file output_qasa/ablation_by_stratum.md --collection-suffix _qasa
"""

import argparse
import csv
import json
import random
import sys
import time
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
    # Phase-two intervention, not a preset field: keep the cross-encoder's
    # threshold and drop its ordering. Read through the attribution metrics and
    # recall@k, which is where the two uses of that score disagree.
    "rerank_order_by_retrieval": [True],
    "retrieval_neighbours": [1, 2],
    "num_paraphrases": [2],
    "qa_model": ["deepset/deberta-v3-base-squad2", "deepset/roberta-base-squad2"],
    "reader": ["generative"],
    # Phase-two intervention on the generative reader: ask it to reproduce the
    # sentence it used and look that sentence up in the chunk. Implies the
    # generative reader, since there is no citation to verify without one.
    # Only the lenient policy is swept. Strict is the same citations with the
    # unverifiable ones dropped, so running it here would generate a second time
    # and make the two policies differ by sampling as well as by policy; it is
    # measured exactly, from one generation, by benchmark/quote_policies.py.
    "citation_quote": ["lenient"],
    # Phase-two intervention on the prompt rather than on any component: Exact
    # Match is 0.000 everywhere, and a two-sentence answer cannot match a
    # noun-phrase reference however correct it is. Implies the generative
    # reader, which is the only one with a prompt to vary.
    "answer_style": ["one_sentence", "phrase"],
    # How many top-ranked chunks reach the prompt. Implies the generative reader,
    # which is the only one that truncates: the extractive reader scores every
    # candidate the retrieval stage passes it. That asymmetry is the point - the
    # reader comparison was never run at constant context, so a difference
    # attributed to the model could be a difference in how much each was shown,
    # and this axis is what tells the two apart. 8 is the shipped default and is
    # already measured by the reader row, so it is not repeated here.
    "max_context_chunks": [4, 16, 30],
}

# Axes the harness applies itself: not question-preset fields, so they are split
# out before the merged config reaches answer_question. Absent from BASELINE on
# purpose - each default is read off the pipeline as built, so there is no second
# copy of the model name to drift.
HARNESS_PARAMS = ("num_paraphrases", "qa_model", "reader", "citation_quote",
                  "answer_style", "max_context_chunks")

# The generative reader's own defaults, restated here rather than imported:
# generative_reader pulls in the Ollama client, and this module is imported by
# tests that must not need it. wanted_reader() builds the same identity string
# the reader assigns itself, so these three must agree with QUOTE_OFF,
# DEFAULT_ANSWER_STYLE and DEFAULT_CONTEXT_CHUNKS - the reader-axis tests check
# that they do.
QUOTE_DEFAULT = "off"
ANSWER_STYLE_DEFAULT = "two_sentences"
CONTEXT_CHUNKS_DEFAULT = 8

# The protocol names recall@1 the primary retrieval metric, so the sweep can
# not report k=10 alone: at k=10 a config that merely drags gold evidence from
# rank 9 to rank 2 looks identical to one that does nothing, and that is exactly
# the movement a reranker is bought for.
RECALL_KS = (1, 3, 5, 10)
RECALL_K = max(RECALL_KS)  # the cut the attributed-evidence metrics score at

# The metrics that need short reference answers. QASA annotates evidence but no
# short answers, so on that set these are not computed at all rather than
# computed against an empty reference list - which token_f1_score would score as
# a clean 0.0 for every configuration, and a column of zeros reads as a system
# that answers nothing rather than as a metric that was never applicable.
ANSWER_METRICS = ["answer_f1", "answer_em"]

# Means written to the CSV; the per-question JSONL keeps everything.
CSV_METRICS = (
    ANSWER_METRICS
    + [f"recall@{k}" for k in RECALL_KS]
    + [f"precision@{k}" for k in RECALL_KS]
    + ["mrr", "evidence_precision", "evidence_recall", "evidence_f1",
       "highlighted_chars", "highlight_precision", "quote_match_rate",
       f"recall@{RECALL_K}_reranked", "mrr_reranked", "latency_s"]
)
CI_METRICS = ["answer_f1", "recall@1", f"recall@{RECALL_K}", "evidence_f1"]

# Stages ZoteroRAG.answer_question times, in the order it runs them. Recorded as
# a fixed list rather than whatever keys the call happened to set: a stage the
# call never reached scores 0.0, so every row has the same columns and the means
# stay comparable across configs.
LATENCY_STAGES = ("expansion", "retrieval", "rerank", "read")
CSV_METRICS += [f"latency_{stage}_s" for stage in LATENCY_STAGES]


def csv_metrics(score_answers: bool = True) -> list[str]:
    """The CSV's metric columns, minus the answer ones on a set without answers."""
    if score_answers:
        return list(CSV_METRICS)
    return [m for m in CSV_METRICS if m not in ANSWER_METRICS]


def build_fieldnames(metrics: list[str] | None = None) -> list[str]:
    fields = ["param", "value", "n_questions"]
    for metric in CSV_METRICS if metrics is None else metrics:
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


def build_configs(baseline: dict, grid: dict, score_answers: bool = True) -> list[dict]:
    """Baseline, oracle-context reference, then one config per (param, value).

    The oracle-context rows are dropped when ``score_answers`` is False. They
    bypass retrieval and hand the reader the gold chunks, so Answer F1 is the
    only thing they measure; without it they would run the reader over every
    question to produce a row of blanks.
    """
    configs = [{"param": "baseline", "value": None, "config": dict(baseline)},
               {"param": "oracle_paper", "value": None, "config": dict(baseline)}]
    if score_answers:
        configs += [
            {"param": "oracle_context", "value": None, "config": dict(baseline)},
            # The reader ceiling has to be measured once per reader, and the
            # one-at-a-time grid cannot express it: it varies the reader
            # against the *baseline*, which answers "how does this reader do
            # with real retrieval", not "how far can this reader get at all".
            {"param": "oracle_context_generative", "value": None,
             "config": dict(baseline, reader="generative")}]
    for param, values in grid.items():
        for value in values:
            config = dict(baseline)
            config[param] = value
            configs.append({"param": param, "value": value, "config": config})
    return configs


def select_configs(configs: list[dict], only: str | None) -> list[dict]:
    """Keep only the named params, or everything when ``only`` is None.

    Raises on a name no config carries, rather than silently running a shorter
    sweep than asked for - a typo would otherwise look like a finished run.
    """
    if not only:
        return configs
    wanted = {name.strip() for name in only.split(",") if name.strip()}
    unknown = wanted - {c["param"] for c in configs}
    if unknown:
        raise SystemExit(f"--only names no such param: {sorted(unknown)}")
    return [c for c in configs if c["param"] in wanted]


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


def highlight_scores(answers, reranked, gold_ids) -> dict:
    """How much text the highlighter marks, and how much of it lands on gold.

    evidence_precision counts chunk *identifiers*, so a reader marking one
    sentence and a reader marking the whole paragraph score identically on it.
    Highlighting is this system's deliverable and is judged by what appears on
    the page, so the metric that decides it cannot be blind to how much appears.
    These two columns are not: the extractive reader marks the sentences its
    span falls in, the generative one marks whole chunks, and that difference is
    now visible.

    Marks are deduplicated by (chunk, start, end) because two answers extracted
    from the same sentences put one highlight on the page, not two.

    ponytail: gold is known per chunk, not per sentence, so a mark inside a gold
    chunk counts wholly on-gold. This measures ink wasted on the wrong chunks,
    not ink wasted inside the right one; sub-chunk gold would be needed for that.
    """
    by_text = {c.chunk.text: (c.chunk.pdf_hash, c.chunk.chunk_index)
               for c in reranked}
    marks = {(by_text[a.context], a.start_char, a.end_char)
             for a in answers if a.context in by_text}
    total = sum(end - start for _, start, end in marks)
    on_gold = sum(end - start for chunk_id, start, end in marks if chunk_id in gold_ids)
    return {"highlighted_chars": float(total),
            "highlight_precision": on_gold / total if total else 0.0}


def quote_scores(rag) -> dict:
    """Share of this question's citations whose quoted sentence is really there.

    This is the verifiability number: with a generative reader the citation is
    an assertion, and until it is looked up in the chunk nothing distinguishes a
    faithful one from a fabricated one. Absent - not zero - for any reader that
    was never asked to quote, so a column of zeros never reads as a model that
    fabricates everything when it is a metric that did not apply.
    """
    stats = getattr(rag.qa_engine, "last_quote_stats", None)
    if not stats or not stats["cited"]:
        return {}
    return {"quote_match_rate": stats["matched"] / stats["cited"]}


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
                   gold_answers: dict, score_answers: bool = True) -> dict | None:
    """Per-question metric row, or None if the question has no gold chunks.

    With ``score_answers`` False the answer columns are absent from the row
    rather than zero, which is what a set carrying evidence but no reference
    answers supports. Retrieval and attribution are scored exactly as usual.
    """
    gold_ids = gold_chunks.get(question["question_id"])
    if gold_ids is None:
        return None

    predicted_answer = answers[0].text if answers else ""
    references = gold_answers.get(question["question_id"], [])
    f1, answer_type = answer_f1(predicted_answer, references)

    ranked = sorted(rag.last_candidates, key=lambda c: c["retrieval_score"], reverse=True)
    ranked_ids = [(c["chunk"].pdf_hash, c["chunk"].chunk_index) for c in ranked]
    reranked_ids = [(c.chunk.pdf_hash, c.chunk.chunk_index) for c in rag.last_reranked]
    predicted_ids = attributed_ids(answers, rag.last_reranked)

    row = {}
    if score_answers:
        row = {"answer_f1": f1,
               "answer_em": answer_exact_match(predicted_answer, references)}
    row.update(per_question_scores(ranked_ids, gold_ids, RECALL_K,
                                   predicted_ids=predicted_ids))
    for k in RECALL_KS:
        if k == RECALL_K:
            continue
        scores = per_question_scores(ranked_ids, gold_ids, k)
        row[f"recall@{k}"] = scores[f"recall@{k}"]
        row[f"precision@{k}"] = scores[f"precision@{k}"]
    row.update(highlight_scores(answers, rag.last_reranked, gold_ids))
    row.update(quote_scores(rag))
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
    """Which reader this config asks for, in ``reader_key`` terms.

    Any of the three generative-only axes implies the generative reader: the
    extractive one returns a verbatim span already, so there is nothing to
    verify, no prompt to vary, and no context window to size - it scores every
    candidate retrieval passes it.
    """
    quote = harness.get("citation_quote", QUOTE_DEFAULT)
    style = harness.get("answer_style", ANSWER_STYLE_DEFAULT)
    chunks = harness.get("max_context_chunks", CONTEXT_CHUNKS_DEFAULT)
    if (harness.get("reader", "extractive") == "generative"
            or quote != QUOTE_DEFAULT or style != ANSWER_STYLE_DEFAULT
            or chunks != CONTEXT_CHUNKS_DEFAULT):
        return f"generative:{quote}:{style}:{chunks}"
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
    if wanted.startswith("generative"):
        rag.qa_engine = GenerativeReader(
            ollama_url=ollama_url,
            citation_quote=harness.get("citation_quote", QUOTE_DEFAULT),
            answer_style=harness.get("answer_style", ANSWER_STYLE_DEFAULT),
            max_context_chunks=harness.get("max_context_chunks",
                                           CONTEXT_CHUNKS_DEFAULT))
    elif qa_model == baseline_engine.model_name:
        rag.qa_engine = baseline_engine
    else:
        # The paraphraser belongs to the baseline engine; an alternative reader
        # never needs it, since one-at-a-time never moves both axes together.
        rag.qa_engine = QAEngine(model_name=qa_model, enable_question_expansion=False)

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_config(rag, questions: list[dict], config: dict, gold_chunks: dict,
               gold_answers: dict, scope_to_gold_paper: bool = False,
               score_answers: bool = True) -> list[dict]:
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
        started = time.perf_counter()
        answers = rag.answer_question(
            q["question"], question_type="general", overrides=pipeline_config,
            num_paraphrases=harness.get("num_paraphrases", 0),
            pdf_hashes=pdf_hashes)
        elapsed = time.perf_counter() - started
        row = score_question(q, answers, rag, gold_chunks, gold_answers,
                             score_answers=score_answers)
        if row is not None:
            row["latency_s"] = elapsed
            for stage in LATENCY_STAGES:
                row[f"latency_{stage}_s"] = rag.last_stage_times.get(stage, 0.0)
            rows.append(row)
    return rows


def run_oracle(rag, questions: list[dict], config: dict, gold_chunks: dict,
               gold_answers: dict) -> list[dict]:
    """Answer every question from the gold chunks, bypassing retrieval entirely.

    Retrieval metrics are perfect by construction and therefore omitted: the
    only meaningful number here is Answer F1, the reader's ceiling.

    ``latency_s`` here times the reader alone - the gold chunks are already in
    hand, so nothing else is running. Against the same number from ``run_config``
    (which times the whole pipeline) it separates what the reader costs from what
    retrieval costs, without instrumenting the pipeline's internals.
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
            started = time.perf_counter()
            answers = rag.qa_engine.extract_answers(
                q["question"], candidates, config, question_variations=[q["question"]])
            elapsed = time.perf_counter() - started
            predicted = answers[0].text if answers else ""
            references = gold_answers.get(q["question_id"], [])
            f1, answer_type = answer_f1(predicted, references)
            rows.append({"answer_f1": f1,
                         "answer_em": answer_exact_match(predicted, references),
                         "latency_s": elapsed,
                         "record": dict(q, answer_type=answer_type,
                                        gold_ids=[list(i) for i in sorted(gold_ids)])})
    finally:
        rag.qdrant_manager.close_connection()
    return rows


def csv_row(entry: dict, rows: list[dict], metrics: list[str] | None = None) -> dict:
    """Mean of every CSV metric present in ``rows``, with CIs where configured."""
    summary = summarize([{k: v for k, v in r.items() if k != "record"} for r in rows],
                        with_ci=True)
    out = {"param": entry["param"], "value": entry["value"],
           "n_questions": summary["n_questions"]}
    for metric in CSV_METRICS if metrics is None else metrics:
        out[metric] = summary.get(metric, "")
        if metric in CI_METRICS:
            out[f"{metric}_ci_low"] = summary.get(f"{metric}_ci_low", "")
            out[f"{metric}_ci_high"] = summary.get(f"{metric}_ci_high", "")
    return out


def dataset_name(golden_dir: Path) -> str:
    """Heading the stratified table carries, taken from the golden-set directory.

    The two sets are never pooled (see benchmark/stratify.py), so the table has
    to say which one it is; deriving it beats a flag nobody would remember to
    pass, which is how a QASA run would end up labelled QASPER.
    """
    return golden_dir.name.removeprefix("output_").upper() or golden_dir.name


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
    parser.add_argument("--only", default=None,
                        help="comma-separated param names to run (default: the whole grid). "
                             "Written for the latency pass, which needs four rows out of "
                             "twenty-two and would otherwise pay for eighteen it discards.")
    parser.add_argument("--sample", type=int, default=None,
                        help="subsample this many questions (default: the whole golden set)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from zotero_rag.pipeline import (
        ZoteroRAG,  # heavy import, kept out of build_configs()'s pure path
    )

    golden_dir = Path(args.golden_dir)
    questions = load_jsonl(golden_dir / "golden_set.jsonl")
    pdf_hash_map = json.loads(Path(args.hash_map).read_text())
    gold_chunks = load_gold_chunks(golden_dir / "golden_set_aligned.jsonl", pdf_hash_map)
    # Only aligned questions are scorable, so every metric shares one denominator.
    # Sampling happens after this filter, not before: --sample is a budget on
    # questions actually scored, and drawing from the full set instead would make
    # the reported n a function of the alignment rate rather than of the request.
    questions = [q for q in questions if q["question_id"] in gold_chunks]
    if args.sample is not None:
        questions = sample_questions(questions, args.sample, args.seed)
    sampled_ids = {q["question_id"] for q in questions}
    print(f"scoring {len(questions)} aligned questions")

    # QASA annotates evidence but no short answers, so it ships no golden_gold.json
    # and the answer columns are dropped for the whole sweep rather than scored
    # against nothing. Presence of the file is the switch: a set either has
    # reference answers or it does not, and there is no third case to configure.
    gold_path = golden_dir / "golden_gold.json"
    score_answers = gold_path.exists()
    gold_answers = {}
    if score_answers:
        gold_answers = {qid: refs for qid, refs in get_answers_and_evidence(
            json.loads(gold_path.read_text()), text_evidence_only=True).items()
            if qid in sampled_ids}
    else:
        print(f"{gold_path} absent: scoring retrieval and attribution only")

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
    metrics = csv_metrics(score_answers)
    fieldnames = build_fieldnames(metrics)
    check_schema_compatible(out_path, fieldnames)
    done = load_completed_configs(out_path)
    if done:
        print(f"resuming: {len(done)} configs already in {out_path}")

    for entry in select_configs(build_configs(BASELINE, GRID, score_answers), args.only):
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
        # Reset per config, not per reader: apply_reader reuses a loaded reader
        # whose identity already matches, so a counter left alone would carry
        # one config's failures into the next one's report.
        if hasattr(rag.qa_engine, "generation_failures"):
            rag.qa_engine.generation_failures = 0
        if entry["param"].startswith("oracle_context"):
            rows = run_oracle(rag, questions, entry["config"], gold_chunks, gold_answers)
        else:
            rows = run_config(rag, questions, entry["config"], gold_chunks, gold_answers,
                              scope_to_gold_paper=entry["param"] == "oracle_paper",
                              score_answers=score_answers)
        # Loud, because these are scored as unanswered questions: an arm with
        # failures is measuring the service as well as the configuration.
        lost = getattr(rag.qa_engine, "generation_failures", 0)
        if lost:
            print(f"WARNING: {entry['param']}={entry['value']} lost {lost} of "
                  f"{len(questions)} questions to failed generation; "
                  f"they are scored as unanswered")
        if not rows:
            print(f"skipped: {entry['param']}={entry['value']} scored no questions")
            continue

        label = config_label(entry["param"], entry["value"])
        write_per_question(rows, out_path.parent / "per_question" / f"{label}.jsonl")
        if entry["param"] == "baseline":
            Path(args.strata_file).write_text(to_markdown(
                {dataset_name(golden_dir): stratify(rows)},
                [m for m in metrics if m in rows[0]]))

        write_header = not out_path.exists()
        with out_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(csv_row(entry, rows, metrics))
        print(f"wrote: {entry['param']}={entry['value']} -> {out_path}")


if __name__ == "__main__":
    main()
