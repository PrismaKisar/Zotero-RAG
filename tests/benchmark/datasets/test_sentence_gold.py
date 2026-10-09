import types

from benchmark.datasets.sentence_gold import (
    gold_intervals,
    sentence_highlight_precision,
    sentence_intervals,
)

SENTENCES = [("We use BERT.", ""), ("It was trained on news.", ""), ("Results follow.", "")]
TEXT = " ".join(s for s, _ in SENTENCES)
CHUNK = types.SimpleNamespace(text=TEXT, sentences=SENTENCES)
GOLD = ("h", 1)


def test_sentences_are_located_by_scanning_and_an_altered_one_is_skipped():
    found = sentence_intervals(TEXT, [("We use BERT.", ""), ("missing", ""), ("Results follow.", "")])
    assert [(s, e) for s, e, _ in found] == [(0, 12), (37, 52)]


def test_a_sentence_is_gold_when_it_contains_a_normalised_span():
    assert gold_intervals(TEXT, SENTENCES, ["the news"]) == [(13, 36)]


def test_marking_only_the_gold_sentence_is_precise_and_the_paragraph_is_not():
    precise = sentence_highlight_precision({(GOLD, 13, 36)}, {GOLD: CHUNK}, {GOLD}, ["news"])
    whole = sentence_highlight_precision({(GOLD, 0, len(TEXT))}, {GOLD: CHUNK}, {GOLD}, ["news"])
    assert precise["highlight_sentence_precision"] == 1.0
    assert whole["highlight_sentence_precision"] == 23 / len(TEXT)


def test_marks_outside_gold_chunks_count_against_precision():
    other = ("h", 9)
    scores = sentence_highlight_precision({(GOLD, 13, 36), (other, 0, 23)},
                                          {GOLD: CHUNK, other: CHUNK}, {GOLD}, ["news"])
    assert scores["highlight_sentence_precision"] == 0.5


def test_nothing_marked_scores_zero():
    assert sentence_highlight_precision(set(), {GOLD: CHUNK}, {GOLD}, ["news"]) == {
        "highlight_sentence_precision": 0.0}


def test_absent_when_no_span_is_found_or_the_set_has_none():
    assert sentence_highlight_precision({(GOLD, 0, 5)}, {GOLD: CHUNK}, {GOLD}, ["absent"]) == {}
    assert sentence_highlight_precision({(GOLD, 0, 5)}, {GOLD: CHUNK}, {GOLD}, []) == {}
