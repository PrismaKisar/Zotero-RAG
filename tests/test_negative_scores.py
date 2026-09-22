from benchmark.negative_scores import OFFTOPIC, read_offtopic, unanswerable_questions


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
