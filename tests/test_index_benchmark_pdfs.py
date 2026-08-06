
from benchmark.index_benchmark_pdfs import build_ingest_items


def test_build_ingest_items_titles_by_paper_id(tmp_path):
    for paper_id in ("1605.03481", "1601.02403"):
        (tmp_path / f"{paper_id}.pdf").write_bytes(b"%PDF-1.4")

    items = build_ingest_items(tmp_path)

    assert [item.title for item in items] == ["1601.02403", "1605.03481"]
    assert all(item.source.path.endswith(".pdf") for item in items)


def test_build_ingest_items_ignores_non_pdf_files(tmp_path):
    (tmp_path / "notes.txt").write_text("not a pdf")
    (tmp_path / "1601.02403.pdf").write_bytes(b"%PDF-1.4")

    items = build_ingest_items(tmp_path)

    assert [item.title for item in items] == ["1601.02403"]
