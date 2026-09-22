"""Scores for questions the library does not answer: the thresholds' other side.

The golden sets hold answerable questions only, so tuning rerank_threshold and
qa_score_threshold on them rewards highlighting something for every question,
which is exactly what a question the library cannot answer must not get. This
scores such questions the way the golden-set runs score theirs: every
question's hybrid top-LIMIT, with the shipped reranker's score and the reader's
span score (no-answer mass kept) for each candidate.

Two sources, kept apart in the output:

- ``offtopic``: benchmark/offtopic_questions.txt - general knowledge, other
  disciplines, research-shaped questions about things no NLP/ML paper studies,
  and nonsense.
- ``unanswerable``: QASPER questions that every annotator marked unanswerable,
  about the papers in this corpus. On topic and sensible, but the paper they
  were asked of does not answer them, which makes them the hard case. Another
  paper in the library might, so silence on them is a lower bound of what is right.

Usage:
  python -m benchmark.negative_scores --golden-dir output_qasper \
      --qasper-release qasper-dev-v0.3.json
  python -m benchmark.negative_scores --golden-dir output_qasa
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# appended, never prepended: the repo root must keep priority over this directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from benchmark.paraphrase_retrieval import HybridSearch
from benchmark.qa_score_distribution import reader_scores
from benchmark.rerank_score_distribution import reranker_scores

OFFTOPIC = Path(__file__).resolve().parent / "offtopic_questions.txt"


def read_offtopic(path: Path = OFFTOPIC) -> list[dict]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    questions = [line for line in lines if line and not line.startswith("#")]
    return [{"question_id": f"offtopic#{i}", "question": q, "source": "offtopic"}
            for i, q in enumerate(questions)]


def unanswerable_questions(qasper: dict, papers: set[str]) -> list[dict]:
    """QASPER questions about ``papers`` that every annotator marked unanswerable."""
    return [{"question_id": qa["question_id"], "question": qa["question"],
             "source": "unanswerable"}
            for paper_id in sorted(papers & set(qasper)) for qa in qasper[paper_id]["qas"]
            if qa["answers"] and all(a["answer"].get("unanswerable") for a in qa["answers"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="output_qasper", help="holds chunks/")
    parser.add_argument("--qasper-release", default=None,
                        help="QASPER JSON release, for its unanswerable questions")
    parser.add_argument("--offtopic", type=Path, default=OFFTOPIC)
    args = parser.parse_args()

    from qa_engine import QAEngine  # heavy imports (the reader, the cross-encoder)
    from reranker import Reranker

    golden_dir = Path(args.golden_dir)
    search = HybridSearch(golden_dir / "chunks")
    questions = read_offtopic(args.offtopic)
    if args.qasper_release:
        release = json.loads(Path(args.qasper_release).read_text())
        questions += unanswerable_questions(release, {paper for paper, _ in search.ids})
    reranker, engine = Reranker(), QAEngine()

    rows = []
    for q in questions:
        hits = search.top(q["question"])
        texts = [search.text_of[c] for c in hits]
        rerank = reranker_scores(reranker, q["question"], texts)
        reader = reader_scores(engine, q["question"], texts)
        rows += [{"question_id": q["question_id"], "source": q["source"], "chunk": list(c),
                  "rerank": r, "qa": s["with_null"]}
                 for c, r, s in zip(hits, rerank, reader, strict=True) if s]

    out_file = golden_dir / "negative_scores.jsonl"
    out_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(dict(Counter(q["source"] for q in questions)), f"-> {out_file}")


if __name__ == "__main__":
    main()
