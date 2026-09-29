"""TEI parsing in zotero_rag/pdf_processor.py, on hand-written TEI (no GROBID)."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from pdf_processor import PDFProcessor

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _tei(header: str, body: str = "") -> ET.Element:
    return ET.fromstring(f'<TEI xmlns="{NS["tei"]}"><teiHeader>{header}</teiHeader>'
                         f'<text><body>{body}</body></text></TEI>')


@pytest.fixture
def processor(tmp_path):
    return PDFProcessor(output_base_dir=str(tmp_path))


def test_the_main_title_wins_over_any_other(processor):
    root = _tei('<fileDesc><titleStmt><title>Plain</title>'
                '<title type="main">Main</title></titleStmt></fileDesc>')
    assert processor._extract_title_from_tei(root, NS) == "Main"


def test_the_source_description_is_the_fallback(processor):
    root = _tei('<fileDesc><sourceDesc><biblStruct><analytic><title>From  source'
                '</title></analytic></biblStruct></sourceDesc></fileDesc>')
    assert processor._extract_title_from_tei(root, NS) == "From source"


def test_a_header_title_beats_a_cited_paper_in_the_body(processor):
    root = _tei('<fileDesc><sourceDesc><biblStruct><monogr><title>Ours</title></monogr>'
                '</biblStruct></sourceDesc></fileDesc>',
                '<listBibl><biblStruct><analytic><title>Cited</title></analytic>'
                '</biblStruct></listBibl>')
    assert processor._extract_title_from_tei(root, NS) == "Ours"


def test_no_title_is_an_empty_string(processor):
    assert processor._extract_title_from_tei(_tei(""), NS) == ""


def test_sections_are_named_from_their_heading(processor):
    assert processor._determine_section_type("3 methodology") == "methods"
    assert processor._determine_section_type("related work") == "body"
    assert processor._determine_section_type("") == "body"
