"""Where the reranker's scores put gold chunks, and what each cut keeps.

rerank_threshold drops every candidate the cross-encoder scores below it,
before the reader sees it. Its 0.001 was read off bge-reranker-base, whose
probabilities stayed low even for real evidence; with the reranker replaced
(benchmark/reranker_diagnosis.py) that scale no longer holds. This scores every
question's hybrid top-LIMIT with the shipped Reranker and reports the
distribution on gold and non-gold chunks, the pooled AUC and a threshold sweep.

A candidate is highlighted only if it clears the rerank threshold *and* the
reader's span score clears qa_score_threshold. When
benchmark/qa_score_distribution.py has scored the same candidates
(qa_scores.jsonl), the report adds the joint sweep of both cuts, which is what
the pipeline would highlight per question.

Usage:
  python -m benchmark.rerank_score_distribution --golden-dir output_qasa
  python -m benchmark.rerank_score_distribution --golden-dir output_qasa --from-scores
"""

import argparse
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

# appended, never prepended: the repo root must keep priority over this directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from benchmark.ablation import load_jsonl
from benchmark.paraphrase_retrieval import LIMIT, HybridSearch
from benchmark.qa_score_distribution import (
    DISTRIBUTION_HEADER,
    STATS_HEADER,
    auc,
    cut_stats,
    distribution_rows,
    stats_cells,
    sweep,
)
from benchmark.retrieval_score_distribution import _table, gold_ids

# gte's probabilities crowd the top: on the hybrid top-30 non-gold chunks have a
# median of 0.80-0.84 and gold ones 0.91-0.93, so every cut that acts is above 0.8.
RERANK_THRESHOLDS = (0.5, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96)
JOINT_RERANK = (0.0, 0.85, 0.88, 0.9, 0.92)
JOINT_QA = (0.0, 0.01, 0.05, 0.1)
# A per-question cut, which no score scale can shift: the k best by rerank score.
TOP_K = (1, 2, 3, 5, 10)


def join_qa(rows: list[dict], qa_rows: list[dict]) -> list[dict]:
    """Each reranked candidate with the reader's span score for it, where one exists."""
    qa = {(r["question_id"], tuple(r["chunk"])): r["with_null"] for r in qa_rows}
    return [{**r, "qa": qa[key]} for r in rows
            if (key := (r["question_id"], tuple(r["chunk"]))) in qa]


def joint_sweep(rows: list[dict], rerank_ts=JOINT_RERANK, qa_ts=JOINT_QA) -> list[dict]:
    """What the two cuts keep together, at every pair of thresholds."""
    return [{"rerank": tr, "qa": tq,
             **cut_stats(rows, lambda r, tr=tr, tq=tq: r["rerank"] >= tr and r["qa"] >= tq)}
            for tr in rerank_ts for tq in qa_ts]


def with_ranks(rows: list[dict]) -> list[dict]:
    """Each row with its rank by rerank score within its question, 1 the best."""
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)
    return [{**row, "rank": rank} for question in by_question.values()
            for rank, row in enumerate(sorted(question, key=lambda r: -r["rerank"]), start=1)]


def top_k_sweep(rows: list[dict], ks=TOP_K, qa_ts=JOINT_QA) -> list[dict]:
    """What keeping the k best per question keeps, with the reader's cut on top."""
    ranked = with_ranks(rows)
    return [{"k": k, "qa": tq,
             **cut_stats(ranked, lambda r, k=k, tq=tq: r["rank"] <= k and r["qa"] >= tq)}
            for k in ks for tq in qa_ts]


def report(name: str, model: str, rows: list[dict], joined: list[dict]) -> str:
    gold = [r["rerank"] for r in rows if r["gold"]]
    other = [r["rerank"] for r in rows if not r["gold"]]
    n_questions = len({r["question_id"] for r in rows})
    out = [f"# Rerank score distribution: {name}",
           (f"`{model}`, {n_questions} questions, hybrid top-{LIMIT} each, {len(gold)} gold and "
            f"{len(other)} non-gold chunks. Pooled AUC gold vs non-gold: {auc(gold, other):.3f}."),
           "## Distribution",
           _table(DISTRIBUTION_HEADER, distribution_rows(rows, ("rerank",))),
           "## Sweep: rerank threshold alone",
           _table(["threshold", *STATS_HEADER],
                  [[r["threshold"], *stats_cells(r)] for r in sweep(rows, "rerank", RERANK_THRESHOLDS)])]
    if joined:
        out += ["## Joint sweep: rerank and QA thresholds together",
                (f"{len(joined)} of {len(rows)} candidates carry a reader score "
                 "(qa_scores.jsonl, score `with_null`). Kept = highlighted."),
                _table(["rerank", "qa", *STATS_HEADER],
                       [[r["rerank"], r["qa"], *stats_cells(r)] for r in joint_sweep(joined)]),
                "## Top-k per question by rerank score, with the QA cut",
                _table(["k", "qa", *STATS_HEADER],
                       [[r["k"], r["qa"], *stats_cells(r)] for r in top_k_sweep(joined)])]
    return "\n\n".join(out) + "\n"


def reranker_scores(reranker, question: str, texts: list[str]) -> list[float]:
    """The shipped Reranker's score for each text, back in input order."""
    # rerank's defaults keep every candidate; the stand-ins carry only what it reads
    stand_ins = [(types.SimpleNamespace(text=text, position=i), 0.0) for i, text in enumerate(texts)]
    scores = [0.0] * len(texts)
    for scored in reranker.rerank(question, stand_ins):
        scores[scored.chunk.position] = scored.rerank_score
    return scores


def score_candidates(golden_dir: Path) -> tuple[str, list[dict]]:
    from reranker import Reranker  # heavy import (the cross-encoder)

    search = HybridSearch(golden_dir / "chunks")
    reranker = Reranker()  # the shipped model and scoring
    rows = []
    for record in load_jsonl(golden_dir / "golden_set_aligned.jsonl"):
        gold = {g for g in gold_ids(record) if g in search.text_of}
        if not gold:
            continue
        hits = search.top(record["question"])
        scores = reranker_scores(reranker, record["question"], [search.text_of[c] for c in hits])
        rows += [{"question_id": record["question_id"], "chunk": list(chunk),
                  "gold": chunk in gold, "rerank": score}
                 for chunk, score in zip(hits, scores, strict=True)]
    return reranker.model_name, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="output_qasa",
                        help="holds golden_set_aligned.jsonl and chunks/")
    parser.add_argument("--from-scores", action="store_true",
                        help="rebuild the report from rerank_scores.jsonl without re-scoring")
    args = parser.parse_args()

    golden_dir = Path(args.golden_dir)
    scores_file = golden_dir / "rerank_scores.jsonl"
    if args.from_scores:
        saved = load_jsonl(scores_file)
        model, rows = saved[0]["model"], saved[1:]
    else:
        model, rows = score_candidates(golden_dir)
        scores_file.write_text("".join(json.dumps(r) + "\n" for r in [{"model": model}, *rows]))
    qa_file = golden_dir / "qa_scores.jsonl"
    joined = join_qa(rows, load_jsonl(qa_file)) if qa_file.exists() else []
    out_file = golden_dir / "rerank_score_distribution.md"
    out_file.write_text(report(golden_dir.name, model, rows, joined))
    print(f"{len(rows)} candidates -> {out_file}")


if __name__ == "__main__":
    main()
