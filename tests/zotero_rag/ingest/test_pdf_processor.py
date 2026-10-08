"""Section naming in zotero_rag/ingest/pdf_processor.py (no GROBID)."""

import xml.etree.ElementTree as ET

import pytest

from zotero_rag.ingest.pdf_processor import PDFProcessor, reference_from_tei
from zotero_rag.models import Reference

TEI = "http://www.tei-c.org/ns/1.0"


@pytest.fixture
def processor(tmp_path):
    return PDFProcessor(output_base_dir=str(tmp_path))


def test_sections_are_named_from_their_heading(processor):
    assert processor._determine_section_type("3 methodology") == "methods"
    assert processor._determine_section_type("related work") == "body"
    assert processor._determine_section_type("") == "body"


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
