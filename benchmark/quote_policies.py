"""Both citation-verification policies scored from one generation per question.

The ablation grid can express one configuration per pass, so measuring the
lenient and strict policies through it means generating twice. At temperature
0.1 those are two different texts, and the difference between the two arms then
carries sampling variation on top of the policy's effect - which matters, since
the effect that difference is used to argue for is a few hundredths wide.

Strict is a pure function of lenient given the same generation: it is the same
citations with the unverifiable ones removed. So this runs the reader once, in
lenient mode, and scores the answers twice - once whole, once filtered to the
citations whose quoted sentence was located. The two rows then differ by the
policy and by nothing else, and the paired bootstrap over them is exact.

Writes the same per-question JSONL layout benchmark/ablation.py does, with
lenient as ``baseline.jsonl``, so benchmark/paired_test.py consumes it unchanged.

Usage:
  python -m benchmark.quote_policies --out-dir output_qasper/quote_exact
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from question_presets import PRESETS

from benchmark.ablation import (
    LATENCY_STAGES,
    load_gold_chunks,
    load_jsonl,
    score_question,
    write_per_question,
)
from benchmark.qasper_evaluator import get_answers_and_evidence


def split_policies(answers, verified_contexts: set) -> tuple[list, list]:
    """``(lenient, strict)`` views of one generation's answers.

    Lenient keeps every citation, marking the whole chunk where the quoted
    sentence was not found. Strict keeps only the citations that were located -
    including when that leaves the question with no attributed answer, which is
    what refusing an unverifiable citation costs and is the point of measuring it.
    """
    return answers, [a for a in answers if a.context in verified_contexts]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="output_qasper")
    parser.add_argument("--hash-map", default="output_qasper/pdf_hash_map.json")
    parser.add_argument("--work-dir", default="output_qasper/grobid")
    parser.add_argument("--out-dir", default="output_qasper/quote_exact")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--collection-suffix", default="_qasper")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    from generative_reader import QUOTE_LENIENT, GenerativeReader

    from zotero_rag.pipeline import ZoteroRAG

    golden_dir = Path(args.golden_dir)
    questions = load_jsonl(golden_dir / "golden_set.jsonl")
    pdf_hash_map = json.loads(Path(args.hash_map).read_text())
    gold_chunks = load_gold_chunks(golden_dir / "golden_set_aligned.jsonl", pdf_hash_map)
    questions = [q for q in questions if q["question_id"] in gold_chunks]
    if args.sample is not None:
        questions = questions[:args.sample]

    gold_path = golden_dir / "golden_gold.json"
    score_answers = gold_path.exists()
    gold_answers = {}
    if score_answers:
        gold_answers = get_answers_and_evidence(
            json.loads(gold_path.read_text()), text_evidence_only=True)

    rag = ZoteroRAG(grobid_url=args.grobid_url, qdrant_url=args.qdrant_url,
                    output_base_dir=args.work_dir,
                    qdrant_collection_suffix=args.collection_suffix)
    rag.qa_engine = GenerativeReader(ollama_url=args.ollama_url,
                                     citation_quote=QUOTE_LENIENT)
    print(f"scoring {len(questions)} aligned questions, one generation each")

    rows = {"baseline": [], "citation_quote_strict": []}
    config = dict(PRESETS["general"])
    for i, q in enumerate(questions, 1):
        started = time.perf_counter()
        answers = rag.answer_question(q["question"], question_type="general",
                                      overrides=config, num_paraphrases=0)
        elapsed = time.perf_counter() - started
        lenient, strict = split_policies(answers, rag.qa_engine.last_verified_contexts)
        for name, view in (("baseline", lenient), ("citation_quote_strict", strict)):
            row = score_question(q, view, rag, gold_chunks, gold_answers,
                                 score_answers=score_answers)
            if row is None:
                continue
            # One generation, so the whole cost belongs to the lenient row; the
            # strict row is a filter applied to it and times nothing of its own.
            row["latency_s"] = elapsed
            for stage in LATENCY_STAGES:
                row[f"latency_{stage}_s"] = rag.last_stage_times.get(stage, 0.0)
            rows[name].append(row)
        if i % 20 == 0:
            print(f"  {i}/{len(questions)}")

    out_dir = Path(args.out_dir) / "per_question"
    for name, name_rows in rows.items():
        write_per_question(name_rows, out_dir / f"{name}.jsonl")
        print(f"wrote: {out_dir / f'{name}.jsonl'} ({len(name_rows)} questions)")


if __name__ == "__main__":
    main()
