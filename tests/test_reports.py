"""The two formats every study shares: JSONL dumps in, Markdown tables out."""

from benchmark.reports import load_jsonl, markdown_table


def test_load_jsonl_reads_one_record_per_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n')
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n  \n{"a": 2}\n')
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_markdown_table_prints_floats_to_three_places_and_the_rest_as_is():
    assert markdown_table(["name", "score", "n"], [["x", 0.12345, 7], ["y", 1.0, "-"]]) == (
        "| name | score | n |\n"
        "|---|---|---|\n"
        "| x | 0.123 | 7 |\n"
        "| y | 1.000 | - |")


def test_markdown_table_with_no_rows_is_just_its_header():
    assert markdown_table(["a"], []) == "| a |\n|---|"
