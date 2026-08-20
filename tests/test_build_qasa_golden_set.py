from benchmark.build_qasa_golden_set import build, resolve_arxiv_id


def _entry(paper_id, title, question_id, question, contexts):
    return {"paper_id": paper_id, "title": title, "question_id": question_id,
            "question": question,
            "evidential_info": [{"context": c, "rationale": "r"} for c in contexts]}


QASA = {
    "0": _entry("paper_1", "P1", 1, "single evidence?", ["ctx a"]),
    "1": _entry("paper_1", "P1", 3, "multi evidence?", ["ctx a", "ctx b"]),
    "2": _entry("paper_1", "P1", 5, "no evidence?", []),
    "3": _entry("paper_2", "P2", 1, "another one?", ["ctx c"]),
}


def test_questions_without_evidence_are_excluded():
    records, stats = build(QASA, papers=None, seed=0)
    assert {r["question_id"] for r in records} == {"paper_1#1", "paper_1#3", "paper_2#1"}
    assert stats["excluded_no_evidence"] == 1
    assert stats["questions_total"] == 4


def test_question_id_is_namespaced_by_paper():
    records, _ = build(QASA, papers=None, seed=0)
    # question_id resets per paper in the raw QASA release (both papers use id 1).
    ids = {r["question_id"] for r in records}
    assert "paper_1#1" in ids and "paper_2#1" in ids


def test_multi_evidence_is_deduplicated_and_labelled():
    records, stats = build(QASA, papers=None, seed=0)
    multi = next(r for r in records if r["question_id"] == "paper_1#3")
    assert multi["multi_evidence"] is True
    assert multi["evidence"] == ["ctx a", "ctx b"]
    assert stats["golden_multi_evidence"] == 1


def test_paper_sampling_is_seeded_and_limits_the_set():
    records, stats = build(QASA, papers=1, seed=42)
    again, _ = build(QASA, papers=1, seed=42)
    assert stats["papers_sampled"] == 1
    assert len({r["paper_id"] for r in records}) == 1
    assert records == again


ATOM_MATCH = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Exact Title Match</title>
  </entry>
</feed>"""

ATOM_MISMATCH = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00002v1</id>
    <title>Something Unrelated</title>
  </entry>
</feed>"""

ATOM_EMPTY = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


def test_resolve_arxiv_id_accepts_normalized_title_match():
    assert resolve_arxiv_id("Exact Title Match", fetch=lambda url: ATOM_MATCH) == "2301.00001v1"


def test_resolve_arxiv_id_rejects_a_different_top_result():
    assert resolve_arxiv_id("Exact Title Match", fetch=lambda url: ATOM_MISMATCH) is None


def test_resolve_arxiv_id_handles_no_results():
    assert resolve_arxiv_id("Nothing Like This", fetch=lambda url: ATOM_EMPTY) is None
