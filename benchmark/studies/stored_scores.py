"""The candidate scores the cut studies read back, and the objective they share.

rerank_score_distribution.py and qa_score_distribution.py score every question's
hybrid top-LIMIT once and store it; negative_scores.py does the same for the
questions the library does not answer. Every study that searches the cuts
(threshold_search.py, answer_gate.py, failure_attribution.py) starts from those
files, grouped per question, so the reading lives here and not in one of them.
"""

from collections import defaultdict
from pathlib import Path

from benchmark.metrics.reports import load_jsonl
from benchmark.studies.offline_retrieval import gold_ids, load_corpus

SILENCE_TARGET = 0.9     # share of off-topic questions that must come back empty

def join_qa(rows: list[dict], qa_rows: list[dict]) -> list[dict]:
    """Each reranked candidate with the reader's span score for it, where one exists."""
    qa = {(r["question_id"], tuple(r["chunk"])): r["with_null"] for r in qa_rows}
    return [{**r, "qa": qa[key]} for r in rows
            if (key := (r["question_id"], tuple(r["chunk"]))) in qa]


def group(rows: list[dict], gold: dict[str, set]) -> list[dict]:
    """Per answerable question: its gold chunks and its (chunk, rerank, qa) candidates."""
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append((tuple(row["chunk"]), row["rerank"], row["qa"]))
    return [{"gold": gold[q], "candidates": c} for q, c in by_question.items() if gold.get(q)]


def group_negatives(rows: list[dict]) -> list[dict]:
    """Per unanswerable question: its source and its (chunk, rerank, qa) candidates."""
    by_question = {}
    for row in rows:
        question = by_question.setdefault(row["question_id"],
                                          {"source": row["source"], "candidates": []})
        question["candidates"].append((tuple(row["chunk"]), row["rerank"], row["qa"]))
    return list(by_question.values())


def load_questions(golden_dir: Path) -> list[dict]:
    ids, _ = load_corpus(golden_dir / "chunks")
    known = set(ids)
    gold = {r["question_id"]: {g for g in gold_ids(r) if g in known}
            for r in load_jsonl(golden_dir / "golden_set_aligned.jsonl")}
    rerank_rows = load_jsonl(golden_dir / "rerank_scores.jsonl")[1:]  # line 0 names the model
    return group(join_qa(rerank_rows, load_jsonl(golden_dir / "qa_scores.jsonl")), gold)


def load_negatives(golden_dir: Path) -> list[dict]:
    """The negative questions scored for ``golden_dir``, or none if they were never scored."""
    path = golden_dir / "negative_scores.jsonl"
    return group_negatives(load_jsonl(path)) if path.exists() else []
