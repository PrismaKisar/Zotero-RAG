import json
import os

import pytest

from benchmark.align_evidence import (MIN_TOKENS, align_paragraph, align_records,
                                      manual_check_report, overlap, tokenize)

PARAGRAPH = ("The retriever encodes every paragraph of the paper with a dense model "
             "and stores the resulting vectors in the index, where they are searched "
             "together with their sparse counterparts.")
# A chunk that only covers part of PARAGRAPH: GROBID never actually cuts a
# paragraph mid-sentence like this, but a truncated chunk from a different
# part of the pipeline should still be rejected rather than credited.
TRUNCATED = ("The retriever encodes every paragraph of the paper with a dense model "
             "and stores the resulting vectors in the index")
# A chunk that contains PARAGRAPH in full but padded with unrelated material,
# the way GROBID sometimes merges several source paragraphs into one chunk.
COARSE = (PARAGRAPH + " Unrelated: the reranker then scores the merged candidates "
          "again with a cross encoder before the answering stage sees any of it, "
          "using a separate model trained on a disjoint held-out split of the data.")
UNRELATED = ("We thank the anonymous reviewers for their comments on an earlier draft "
             "of this manuscript and the funding agency for its support.")


def test_overlap_is_symmetric_and_normalised_by_the_longer_text():
    tokens = tokenize(PARAGRAPH)
    assert overlap(tokens, tokens) == 1.0
    assert overlap(tokens, tokenize(TRUNCATED)) == overlap(tokenize(TRUNCATED), tokens)
    assert overlap(tokens, tokenize(UNRELATED)) < 0.5


def test_short_texts_never_match():
    short = ["the", "index"] * (MIN_TOKENS // 2 - 1)
    assert overlap(short, tokenize(PARAGRAPH)) == 0.0


def test_truncated_chunk_is_rejected_despite_being_a_clean_subset():
    # TRUNCATED is a strict, exact-wording subset of PARAGRAPH (no noise at
    # all), which is exactly the case that saturated near 1.0 under the old
    # shorter-text normalisation. Normalising by the longer text catches it.
    assert align_paragraph(PARAGRAPH, [TRUNCATED]) == []


def test_coarse_chunk_is_rejected_despite_containing_the_full_paragraph():
    # COARSE contains every token of PARAGRAPH, but padded with enough
    # unrelated material that retrieving it would not read as a clean match.
    assert align_paragraph(PARAGRAPH, [COARSE]) == []


def test_chunk_mostly_made_of_other_material_is_left_out():
    assert align_paragraph(PARAGRAPH, [UNRELATED]) == []


def test_unmatched_paragraph_is_not_forced_onto_its_closest_chunk():
    assert align_paragraph(PARAGRAPH, [UNRELATED, TRUNCATED, COARSE]) == []


def _record(qid, evidence, paper="1111.1111"):
    return {"paper_id": paper, "question_id": qid, "question": "?",
            "evidence": evidence, "multi_evidence": len(evidence) > 1}


def test_partially_aligned_questions_are_dropped_and_counted():
    records = [_record("ok", [PARAGRAPH]),
               _record("partial", [PARAGRAPH, UNRELATED]),
               _record("none", [UNRELATED]),
               _record("nochunks", [PARAGRAPH], paper="9999.9999")]
    chunks = {"1111.1111": [TRUNCATED, PARAGRAPH]}
    aligned, dropped, stats = align_records(records, chunks)

    assert [r["question_id"] for r in aligned] == ["ok"]
    assert {r["question_id"] for r in dropped} == {"partial", "none", "nochunks"}
    assert stats["questions_dropped_partial_alignment"] == 1
    assert stats["questions_dropped_no_alignment"] == 1
    assert stats["questions_without_chunks"] == 1
    assert stats["question_alignment_rate"] == 0.25
    assert stats["evidence_paragraphs_unaligned"] == 2


def test_split_evidence_across_several_chunks_is_not_aligned():
    # A paragraph genuinely spread over two chunks (neither alone reaching
    # the threshold) is left unaligned: retrieving one chunk would not give
    # access to the full annotated evidence, so it doesn't count as a match.
    _, dropped, _ = align_records([_record("split", [PARAGRAPH])],
                                  {"1111.1111": [UNRELATED, TRUNCATED]})
    assert [r["question_id"] for r in dropped] == ["split"]


def test_one_line_evidence_is_counted_apart():
    _, dropped, stats = align_records([_record("tiny", ["which is equivalent to"])],
                                      {"1111.1111": [PARAGRAPH]})
    assert stats["evidence_paragraphs_too_short"] == 1
    assert stats["questions_dropped_no_alignment"] == 1
    assert len(dropped) == 1


def test_aligned_record_carries_the_matching_chunks():
    aligned, _, _ = align_records([_record("ok", [PARAGRAPH])],
                                  {"1111.1111": [UNRELATED, PARAGRAPH]})
    hits = aligned[0]["aligned_chunks"][PARAGRAPH]
    assert [h["chunk_index"] for h in hits] == [1]
    assert hits[0]["overlap"] >= 0.8


@pytest.mark.skipif(not os.environ.get("QASPER_DEV"),
                    reason="set QASPER_DEV to the dev split to validate the threshold")
def test_threshold_does_not_reject_evidence_it_should_accept():
    """Sanity check of the criterion against the paragraphs QASPER itself uses.

    Fed the annotated paragraphs as if they were our chunks, the procedure has
    to recover almost all of them: anything lost here is a false negative of the
    criterion rather than a difference between LaTeX and PDF segmentation.
    """
    qasper = json.loads(open(os.environ["QASPER_DEV"]).read())
    chunks = {pid: [p for section in paper["full_text"] for p in section["paragraphs"] if p.strip()]
              for pid, paper in qasper.items()}
    records = [{"paper_id": pid, "question_id": qa["question_id"], "question": qa["question"],
                "evidence": list(dict.fromkeys(e for a in qa["answers"]
                                               for e in a["answer"]["evidence"]
                                               if not e.startswith("FLOAT SELECTED")))}
               for pid, paper in qasper.items() for qa in paper["qas"]]
    records = [r for r in records if r["evidence"]]

    _, _, stats = align_records(records, chunks)
    scorable = stats["evidence_paragraphs_total"] - stats["evidence_paragraphs_too_short"]
    assert stats["evidence_paragraphs_aligned"] / scorable > 0.99


def test_manual_check_report_contains_both_texts():
    aligned, _, _ = align_records([_record("ok", [PARAGRAPH])],
                                  {"1111.1111": [PARAGRAPH]})
    report = manual_check_report(aligned, {"1111.1111": [PARAGRAPH]}, size=5, seed=1)
    assert PARAGRAPH in report
    assert "correct?" in report
