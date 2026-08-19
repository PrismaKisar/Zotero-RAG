from benchmark.build_qasper_golden_set import build, gold_subset, question_type


def _qa(qid, question, answers):
    return {"question_id": qid, "question": question,
            "answers": [{"answer": a} for a in answers]}


def _extractive(spans, evidence):
    return {"unanswerable": False, "extractive_spans": spans, "yes_no": None,
            "free_form_answer": "", "evidence": evidence}


ABSTRACTIVE = {"unanswerable": False, "extractive_spans": [], "yes_no": None,
               "free_form_answer": "rewritten", "evidence": ["p"]}
YES_NO = {"unanswerable": False, "extractive_spans": [], "yes_no": True,
          "free_form_answer": "", "evidence": ["p"]}
UNANSWERABLE = {"unanswerable": True, "extractive_spans": [], "yes_no": None,
                "free_form_answer": "", "evidence": []}

QASPER = {
    "1111.1111": {"title": "P1", "qas": [
        _qa("q1", "single evidence?", [_extractive(["a span"], ["par one"])]),
        _qa("q2", "multi evidence?", [_extractive(["s1"], ["par one"]),
                                      _extractive(["s2"], ["par two"])]),
        _qa("q3", "abstractive?", [ABSTRACTIVE]),
        _qa("q4", "boolean?", [YES_NO]),
        _qa("q5", "unanswerable?", [UNANSWERABLE]),
        _qa("q6", "from a table?", [_extractive(["s"], ["FLOAT SELECTED: Table 1"])]),
    ]},
    "2222.2222": {"title": "P2", "qas": [
        _qa("q7", "another one?", [_extractive(["s"], ["par x"])]),
    ]},
}


def test_question_type_ties_do_not_count_as_extractive():
    tied = _qa("t", "?", [_extractive(["s"], ["p"]), ABSTRACTIVE])
    assert question_type(tied) != "extractive"
    majority = _qa("m", "?", [_extractive(["s"], ["p"]), _extractive(["s"], ["p"]), ABSTRACTIVE])
    assert question_type(majority) == "extractive"


def test_only_extractive_text_questions_survive():
    records, stats = build(QASPER, papers=None, seed=0)
    assert {r["question_id"] for r in records} == {"q1", "q2", "q7"}
    assert stats["excluded_abstractive"] == 1
    assert stats["excluded_yes_no"] == 1
    assert stats["excluded_unanswerable"] == 1
    assert stats["excluded_table_or_figure"] == 1
    assert stats["questions_total"] == 7


def test_multi_evidence_is_labelled_not_excluded():
    records, stats = build(QASPER, papers=None, seed=0)
    multi = [r for r in records if r["multi_evidence"]]
    assert [r["question_id"] for r in multi] == ["q2"]
    assert multi[0]["evidence"] == ["par one", "par two"]
    assert stats["golden_multi_evidence"] == 1


def test_gold_subset_keeps_only_golden_set_questions():
    records, _ = build(QASPER, papers=None, seed=0)
    subset = gold_subset(QASPER, records)
    assert [qa["question_id"] for qa in subset["1111.1111"]["qas"]] == ["q1", "q2"]
    assert [qa["question_id"] for qa in subset["2222.2222"]["qas"]] == ["q7"]
    # The original release must not be mutated: it is read again by other steps.
    assert len(QASPER["1111.1111"]["qas"]) == 6


def test_paper_sampling_is_seeded_and_limits_the_set():
    records, stats = build(QASPER, papers=1, seed=42)
    again, _ = build(QASPER, papers=1, seed=42)
    assert stats["papers_sampled"] == 1
    assert len({r["paper_id"] for r in records}) == 1
    assert records == again
