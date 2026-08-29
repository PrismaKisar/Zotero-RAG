"""The two citation policies derived from a single generation.

The point of this module is that strict and lenient differ by the policy and by
nothing else. These tests pin the filter itself; the reader-side half - which
citations count as verified - is covered in test_generative_reader.py.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from benchmark.quote_policies import split_policies


class _Answer:
    def __init__(self, context):
        self.context = context


def test_lenient_keeps_every_citation():
    answers = [_Answer("chunk a"), _Answer("chunk b")]
    lenient, _ = split_policies(answers, {"chunk a"})
    assert lenient == answers


def test_strict_keeps_only_the_citations_that_were_located():
    answers = [_Answer("chunk a"), _Answer("chunk b"), _Answer("chunk c")]
    _, strict = split_policies(answers, {"chunk a", "chunk c"})
    assert [a.context for a in strict] == ["chunk a", "chunk c"]


def test_strict_can_leave_the_question_with_no_answer_at_all():
    """That is the cost of refusing an unverifiable citation, not a bug."""
    _, strict = split_policies([_Answer("chunk a")], set())
    assert strict == []


def test_both_policies_are_views_of_the_same_generation():
    """If strict ever stopped being a subset, the comparison would not be paired."""
    answers = [_Answer("chunk a"), _Answer("chunk b")]
    lenient, strict = split_policies(answers, {"chunk b"})
    assert all(a in lenient for a in strict)
