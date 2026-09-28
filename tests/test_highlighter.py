"""PDF highlighting from GROBID's "page,x,y,w,h" sentence boxes."""

import sys
from pathlib import Path

import fitz

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from highlighter import highlight_pdf
from models import Answer


def _pdf(tmp_path, pages=2):
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    path = tmp_path / "paper.pdf"
    doc.save(path)
    return str(path)


def _answer(pdf_path, coords, color=(0.0, 1.0, 1.0)):
    return Answer(text="a", context="a", page_number=0, title="T", query="q",
                  sentence_coords=coords, color=color, pdf_path=pdf_path)


def _annots(path):
    with fitz.open(path) as doc:
        return [[(a.type[1], a.colors.get("stroke")) for a in page.annots()] for page in doc]


def test_every_box_is_highlighted_and_each_page_gets_one_note(tmp_path):
    out = str(tmp_path / "out.pdf")
    answer = _answer(_pdf(tmp_path), ["1,10,10,50,10;1,10,30,50,10", "2,10,10,50,10"])

    assert highlight_pdf([answer], out) == out

    first, second = _annots(out)
    assert sorted(kind for kind, _ in first) == ["Highlight", "Highlight", "Text"]
    assert sorted(kind for kind, _ in second) == ["Highlight", "Text"]
    assert all(tuple(color) == answer.color for kind, color in first if kind == "Highlight")


def test_malformed_and_out_of_range_boxes_are_skipped(tmp_path):
    out = str(tmp_path / "out.pdf")
    answer = _answer(_pdf(tmp_path, pages=1), ["1,10,10", "x,1,2,3,4", "9,10,10,50,10", ""])

    assert highlight_pdf([answer], out) == out
    assert _annots(out) == [[]]


def test_a_second_run_keeps_the_first_run_s_highlights(tmp_path):
    out = str(tmp_path / "out.pdf")
    pdf = _pdf(tmp_path, pages=1)
    highlight_pdf([_answer(pdf, ["1,10,10,50,10"])], out)
    highlight_pdf([_answer(pdf, ["1,10,40,50,10"])], out)

    assert [kind for kind, _ in _annots(out)[0]].count("Highlight") == 2


def test_nothing_to_highlight_writes_nothing(tmp_path):
    out = tmp_path / "out.pdf"
    assert highlight_pdf([], str(out)) is None
    assert not out.exists()


def test_an_unreadable_source_returns_none(tmp_path):
    assert highlight_pdf([_answer(str(tmp_path / "missing.pdf"), ["1,1,1,1,1"])],
                         str(tmp_path / "out.pdf")) is None
