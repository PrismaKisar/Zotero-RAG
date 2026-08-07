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
- ``foreign``: real questions from the same benchmark about papers that neither
  split holds. Same style, domain and vocabulary as the questions the library
  must answer, and not written by us; the off-topic list is easy to stay silent
  on, these are not. A generic one ("which datasets do they use?") may still be
  answered by a paper in the library, so silence is again a lower bound.

Usage:
  python -m benchmark.negative_scores --golden-dir data/dev/qasper \
      --qasper-release data/raw/qasper-dev-v0.3.json \
      --foreign-release data/raw/qasper-train-v0.3.json data/raw/qasper-dev-v0.3.json
  python -m benchmark.negative_scores --golden-dir data/dev/qasa \
      --foreign-release data/raw/testset_answerable_1554_v1.1.json
"""

import argparse
import json
import random
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


def release_questions(release: dict) -> list[dict]:
    """Every question of a QASPER or QASA release, with its paper.

    QASPER is keyed by paper and keeps only questions some annotator answered, so
    a foreign question is one its own paper does answer; QASA ships answerable
    questions only, keyed by record, and ids are made unique the way its golden
    set makes them.
    """
    if all("qas" in paper for paper in release.values()):
        return [{"paper_id": paper_id, "question_id": qa["question_id"], "question": qa["question"]}
                for paper_id, paper in release.items() for qa in paper["qas"]
                if any(not a["answer"].get("unanswerable") for a in qa["answers"])]
    return [{"paper_id": r["paper_id"], "question_id": f"{r['paper_id']}#{r['question_id']}",
             "question": r["question"]} for r in release.values()]


def foreign_questions(questions: list[dict], used: set[str], n: int, seed: int) -> list[dict]:
    """At most ``n`` questions about papers outside ``used``, drawn reproducibly."""
    pool = sorted((q for q in questions if q["paper_id"] not in used),
                  key=lambda q: q["question_id"])
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    return [{"question_id": q["question_id"], "question": q["question"], "source": "foreign"}
            for q in picked]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="data/dev/qasper", help="holds chunks/")
    parser.add_argument("--qasper-release", default=None,
                        help="QASPER JSON release, for its unanswerable questions")
    parser.add_argument("--offtopic", type=Path, default=OFFTOPIC)
    parser.add_argument("--foreign-release", nargs="*", default=[],
                        help="releases of the same benchmark to draw foreign questions from")
    parser.add_argument("--used-sets", nargs="*",
                        default=[f"data/{split}/{corpus}/golden_set.jsonl"
                                 for split in ("dev", "heldout") for corpus in ("qasper", "qasa")],
                        help="golden_set.jsonl files whose papers no foreign question may be about")
    parser.add_argument("--foreign", type=int, default=100, help="how many foreign questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from qa_engine import QAEngine  # heavy imports (the reader, the cross-encoder)
    from reranker import Reranker

    golden_dir = Path(args.golden_dir)
    search = HybridSearch(golden_dir / "chunks")
    questions = read_offtopic(args.offtopic)
    if args.qasper_release:
        release = json.loads(Path(args.qasper_release).read_text())
        questions += unanswerable_questions(release, {paper for paper, _ in search.ids})
    if args.foreign_release:
        pool = [q for path in args.foreign_release
                for q in release_questions(json.loads(Path(path).read_text()))]
        used = {json.loads(line)["paper_id"] for path in args.used_sets
                for line in Path(path).read_text().splitlines() if line.strip()}
        questions += foreign_questions(list({q["question_id"]: q for q in pool}.values()),
                                       used, args.foreign, args.seed)
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
