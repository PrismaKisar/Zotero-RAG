from benchmark.negative_scores import (
    OFFTOPIC,
    foreign_questions,
    read_offtopic,
    release_questions,
    unanswerable_questions,
)


def test_offtopic_file_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("# a comment\nWhat is the capital of Australia?\n\n  How loud is blue?  \n")
    assert [q["question"] for q in read_offtopic(path)] == [
        "What is the capital of Australia?", "How loud is blue?"]
    assert {q["source"] for q in read_offtopic(path)} == {"offtopic"}


def test_the_shipped_offtopic_list_is_not_empty_and_has_unique_ids():
    questions = read_offtopic(OFFTOPIC)
    assert len(questions) >= 30
    assert len({q["question_id"] for q in questions}) == len(questions)


def test_unanswerable_needs_every_annotator_and_a_paper_in_the_corpus():
    unanswerable = {"answer": {"unanswerable": True}}
    answered = {"answer": {"unanswerable": False, "extractive_spans": ["x"]}}
    release = {
        "p1": {"qas": [{"question_id": "all", "question": "q1", "answers": [unanswerable] * 2},
                       {"question_id": "split", "question": "q2",
                        "answers": [unanswerable, answered]},
                       {"question_id": "none", "question": "q3", "answers": []}]},
        "p2": {"qas": [{"question_id": "elsewhere", "question": "q4",
                        "answers": [unanswerable]}]},
    }
    kept = unanswerable_questions(release, {"p1"})
    assert [q["question_id"] for q in kept] == ["all"]
    assert kept[0]["source"] == "unanswerable"


def test_release_questions_reads_qasper_answerable_questions_only():
    answered = {"answer": {"unanswerable": False, "extractive_spans": ["x"]}}
    unanswerable = {"answer": {"unanswerable": True}}
    release = {"p1": {"qas": [{"question_id": "a", "question": "q1", "answers": [answered]},
                              {"question_id": "u", "question": "q2", "answers": [unanswerable]}]}}
    assert release_questions(release) == [{"paper_id": "p1", "question_id": "a", "question": "q1"}]


def test_release_questions_reads_qasa_records():
    release = {"0": {"paper_id": "paper_1", "question_id": "7", "question": "q"}}
    assert release_questions(release) == [
        {"paper_id": "paper_1", "question_id": "paper_1#7", "question": "q"}]


def test_foreign_questions_skip_used_papers_and_sample_reproducibly():
    questions = [{"paper_id": p, "question_id": f"{p}#{i}", "question": f"{p} {i}"}
                 for p in ("used", "free") for i in range(5)]
    picked = foreign_questions(questions, used={"used"}, n=3, seed=0)
    assert len(picked) == 3
    assert all(q["question_id"].startswith("free#") for q in picked)
    assert {q["source"] for q in picked} == {"foreign"}
    assert picked == foreign_questions(questions, used={"used"}, n=3, seed=0)
    assert len(foreign_questions(questions, used={"used"}, n=50, seed=0)) == 5
