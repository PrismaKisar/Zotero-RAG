"""Section naming in zotero_rag/ingest/pdf_processor.py (no GROBID)."""

import xml.etree.ElementTree as ET

import pytest

from zotero_rag.ingest.pdf_processor import (
    PDFProcessor,
    reference_from_tei,
    section_type,
)
from zotero_rag.models import Reference

TEI = "http://www.tei-c.org/ns/1.0"


@pytest.fixture
def processor(tmp_path):
    return PDFProcessor(output_base_dir=str(tmp_path))


def test_sections_are_named_from_their_heading():
    assert section_type("3 methodology") == "methods"
    assert section_type("related work") == "body"
    assert section_type("") == "body"


def _p(*sentences):
    return "<p>" + "".join(f'<s coords="{c}">{t}</s>' for t, c in sentences) + "</p>"


LONG = "one two three four five six seven eight nine ten"


def test_tei_paragraphs_become_chunks_with_short_ones_merged_forward(processor):
    root = ET.fromstring(
        f'<TEI xmlns="{TEI}"><teiHeader><profileDesc><abstract><div>'
        f'{_p(("Short abstract.", "1,1,1,1,1"))}{_p((LONG, "1,2,2,2,2"))}'
        f'</div></abstract></profileDesc></teiHeader><text><body>'
        f'<div><head>2 Methods</head>{_p((LONG, "3,1,1,1,1"), ("Second.", ""))}</div>'
        f'<div>{_p(("  ", ""))}{_p(("Trailing bit.", "4,1,1,1,1"))}</div>'
        f'</body></text></TEI>')

    chunks = processor._extract_chunks_from_tei(root)

    assert [(c.chunk_index, c.section, c.page_number) for c in chunks] == [
        (0, "abstract", 0), (1, "methods", 2)]
    assert chunks[0].text == f"Short abstract. {LONG}"
    assert chunks[0].sentences == [("Short abstract.", "1,1,1,1,1"), (LONG, "1,2,2,2,2")]
    assert chunks[1].text == f"{LONG} Second. Trailing bit."
    assert chunks[1].sentences[-1] == ("Trailing bit.", "4,1,1,1,1")


def test_the_reference_comes_from_the_tei_header_not_the_cited_papers():
    author = '<author><persName><forename>A</forename><surname>{}</surname></persName></author>'
    root = ET.fromstring(
        f'<TEI xmlns="{TEI}"><teiHeader><fileDesc><sourceDesc><biblStruct>'
        f'<analytic>{author.format("Dasigi")}{author.format("Lo")}</analytic>'
        f'<monogr><imprint><date type="published" when="2021-06"/></imprint></monogr>'
        f'</biblStruct></sourceDesc></fileDesc></teiHeader>'
        f'<text><back><listBibl><biblStruct><analytic>{author.format("Cited")}</analytic>'
        f'</biblStruct></listBibl></back></text></TEI>')
    assert reference_from_tei(root) == Reference(["Dasigi", "Lo"], "2021")


def test_a_header_without_authors_or_date_gives_an_empty_reference():
    root = ET.fromstring(f'<TEI xmlns="{TEI}"><teiHeader/></TEI>')
    assert reference_from_tei(root) == Reference([], None)
