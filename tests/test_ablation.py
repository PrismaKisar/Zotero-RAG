import json

import pytest

from benchmark.ablation import (
    CI_METRICS,
    RECALL_K,
    answer_f1,
    attributed_ids,
    build_configs,
    build_fieldnames,
    check_schema_compatible,
    csv_row,
    load_completed_configs,
    load_gold_chunks,
    run_config,
    run_oracle,
    sample_questions,
    write_per_question,
)


class _FakeParagraph:
    def __init__(self, pdf_hash, para_idx, text=""):
        self.pdf_hash = pdf_hash
        self.para_idx = para_idx
        self.text = text


class _FakeReranked:
    def __init__(self, pdf_hash, para_idx, text=""):
        self.paragraph = _FakeParagraph(pdf_hash, para_idx, text)


class _FakeAnswer:
    def __init__(self, text="some answer", context="ctx-1"):
        self.text = text
        self.context = context


class _FakeRag:
    """Stubs the two stages scoring reads: last_candidates (pre-rerank) and
    last_reranked (post-rerank), which can carry different orderings."""

    def __init__(self, answers=None):
        self._answers = [_FakeAnswer()] if answers is None else answers
        self.last_candidates = [
            {"paragraph": _FakeParagraph("h", 2, "ctx-2"), "retrieval_score": 0.9},
            {"paragraph": _FakeParagraph("h", 1, "ctx-1"), "retrieval_score": 0.5},
        ]
        self.last_reranked = [_FakeReranked("h", 1, "ctx-1"), _FakeReranked("h", 2, "ctx-2")]

    def answer_question(self, question, question_type, overrides, num_paraphrases):
        return self._answers


GOLD_ANSWERS = {"q1": [{"answer": "some answer", "type": "extractive"}]}
QUESTIONS = [{"question_id": "q1", "question": "What is the model?"}]


def test_build_configs_includes_baseline_oracle_and_one_variant_per_grid_value():
    baseline = {"a": 1, "b": 2}
    grid = {"a": [10, 20], "b": [99]}

    configs = build_configs(baseline, grid)

    assert configs[0] == {"param": "baseline", "value": None, "config": {"a": 1, "b": 2}}
    assert configs[1]["param"] == "oracle_context"
    assert {"param": "a", "value": 10, "config": {"a": 10, "b": 2}} in configs
    assert {"param": "b", "value": 99, "config": {"a": 1, "b": 99}} in configs
    assert len(configs) == 5


def test_build_configs_never_mutates_baseline():
    baseline = {"a": 1}
    build_configs(baseline, {"a": [2]})
    assert baseline == {"a": 1}


def test_grid_ablates_parameters_that_reorder_results():
    """Regression guard: a threshold-only grid cannot move recall@k."""
    from benchmark.ablation import GRID
    assert {"retrieval_mode", "rerank_enabled", "result_limit"} <= set(GRID)


def test_load_gold_chunks_maps_paper_id_to_pdf_hash(tmp_path):
    aligned = tmp_path / "golden_set_aligned.jsonl"
    aligned.write_text(
        '{"paper_id": "1601.02403", "question_id": "q1", '
        '"aligned_chunks": {"ev1": [{"chunk_index": 3, "overlap": 1.0}], '
        '"ev2": [{"chunk_index": 5, "overlap": 0.9}]}}\n'
    )

    gold = load_gold_chunks(aligned, {"1601.02403": "abc123"})

    assert gold == {"q1": {("abc123", 3), ("abc123", 5)}}


def test_load_gold_chunks_skips_papers_missing_from_hash_map(tmp_path):
    aligned = tmp_path / "golden_set_aligned.jsonl"
    aligned.write_text(
        '{"paper_id": "9999.9999", "question_id": "q1", '
        '"aligned_chunks": {"ev1": [{"chunk_index": 0, "overlap": 1.0}]}}\n'
    )
    assert load_gold_chunks(aligned, {}) == {}


def test_sample_questions_is_deterministic_for_a_given_seed():
    questions = [{"question_id": str(i)} for i in range(20)]
    assert sample_questions(questions, 5, seed=1) == sample_questions(questions, 5, seed=1)
    assert len(sample_questions(questions, 5, seed=1)) == 5


def test_sample_questions_returns_all_when_n_exceeds_pool():
    questions = [{"question_id": str(i)} for i in range(3)]
    assert sample_questions(questions, 10, seed=1) == questions


def test_answer_f1_takes_the_best_reference():
    f1, answer_type = answer_f1("neural attention", [
        {"answer": "completely different", "type": "abstractive"},
        {"answer": "neural attention", "type": "extractive"},
    ])
    assert f1 == 1.0
    assert answer_type == "extractive"


def test_answer_f1_without_references():
    assert answer_f1("anything", []) == (0.0, "none")


def test_attributed_ids_maps_answers_back_to_chunks():
    reranked = [_FakeReranked("h", 1, "ctx-1"), _FakeReranked("h", 2, "ctx-2")]
    answers = [_FakeAnswer(context="ctx-2")]
    assert attributed_ids(answers, reranked) == {("h", 2)}
    # an answer whose context is not among the reranked chunks is not attributable
    assert attributed_ids([_FakeAnswer(context="unknown")], reranked) == set()
    assert attributed_ids([], reranked) == set()


def test_run_config_scores_retrieval_and_rerank_orderings_independently():
    rows = run_config(_FakeRag(), QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS)

    assert len(rows) == 1
    row = rows[0]
    assert row["answer_f1"] == 1.0
    # pre-rerank order is (h,2) then (h,1) -> gold is second
    assert row["mrr"] == 0.5
    # post-rerank order is (h,1) first -> gold is first
    assert row["mrr_reranked"] == 1.0
    assert row[f"recall@{RECALL_K}"] == 1.0
    assert row[f"recall@{RECALL_K}_reranked"] == 1.0


def test_run_config_evidence_prf_scores_only_attributed_chunks():
    rows = run_config(_FakeRag(), QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS)
    row = rows[0]
    # the answer came from ctx-1 = (h,1), which is exactly the gold chunk
    assert row["evidence_precision"] == 1.0
    assert row["evidence_f1"] == 1.0
    # while precision@k is diluted by the second, non-gold retrieved chunk
    assert row[f"precision@{RECALL_K}"] == 0.5


def test_run_config_penalises_attributing_the_wrong_chunk():
    rag = _FakeRag(answers=[_FakeAnswer(context="ctx-2")])
    row = run_config(rag, QUESTIONS, {"": None}, {"q1": {("h", 1)}}, GOLD_ANSWERS)[0]
    assert row["evidence_f1"] == 0.0
    assert row[f"recall@{RECALL_K}"] == 1.0  # retrieval still found it


def test_run_config_skips_questions_without_gold_chunks():
    assert run_config(_FakeRag(), QUESTIONS, {}, {}, GOLD_ANSWERS) == []


class _FakeQdrant:
    def __init__(self, paragraphs):
        self._paragraphs = paragraphs
        self.opened = False

    def open_connection(self):
        self.opened = True

    def close_connection(self):
        self.opened = False

    def fetch_paragraphs(self, ids):
        return [p for p in self._paragraphs if (p.pdf_hash, p.para_idx) in set(ids)]


class _FakeQaEngine:
    def __init__(self, answers):
        self._answers = answers
        self.seen_candidates = None

    def extract_answers(self, question, candidates, config, question_variations=None):
        self.seen_candidates = candidates
        return self._answers


class _OracleRag:
    def __init__(self, paragraphs, answers):
        self.qdrant_manager = _FakeQdrant(paragraphs)
        self.qa_engine = _FakeQaEngine(answers)


def test_run_oracle_feeds_gold_chunks_to_the_reader():
    from models import Paragraph

    gold_para = Paragraph(text="gold text", page_num=1, para_idx=1, title="T", pdf_hash="h")
    rag = _OracleRag([gold_para], [_FakeAnswer(text="some answer")])

    rows = run_oracle(rag, QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS)

    assert len(rows) == 1
    assert rows[0]["answer_f1"] == 1.0
    # the reader saw exactly the gold paragraph, never the retriever's output
    assert [c.paragraph.text for c in rag.qa_engine.seen_candidates] == ["gold text"]
    # oracle rows carry no retrieval metrics: they would be 1.0 by construction
    assert f"recall@{RECALL_K}" not in rows[0]
    assert rag.qdrant_manager.opened is False  # connection closed again


def test_run_oracle_skips_questions_whose_gold_chunks_are_not_indexed():
    rag = _OracleRag([], [_FakeAnswer()])
    assert run_oracle(rag, QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS) == []


def test_build_fieldnames_pairs_ci_columns_with_their_metric():
    fields = build_fieldnames()
    assert fields[:3] == ["param", "value", "n_questions"]
    for metric in CI_METRICS:
        assert fields.index(f"{metric}_ci_low") == fields.index(metric) + 1
        assert fields.index(f"{metric}_ci_high") == fields.index(metric) + 2


def test_csv_row_averages_rows_and_fills_missing_metrics():
    rows = run_config(_FakeRag(), QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS) * 5
    row = csv_row({"param": "baseline", "value": None}, rows)

    assert set(row) == set(build_fieldnames())
    assert row["n_questions"] == 5
    assert row["answer_f1"] == 1.0
    assert row["answer_f1_ci_low"] <= row["answer_f1"] <= row["answer_f1_ci_high"]


def test_csv_row_leaves_retrieval_columns_blank_for_oracle_rows():
    oracle_rows = [{"answer_f1": 0.5, "record": {"question_id": "q1"}}] * 3
    row = csv_row({"param": "oracle_context", "value": None}, oracle_rows)
    assert row["answer_f1"] == 0.5
    assert row[f"recall@{RECALL_K}"] == ""


def test_write_per_question_dumps_one_json_object_per_row(tmp_path):
    rows = run_config(_FakeRag(), QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS)
    path = tmp_path / "per_question" / "baseline.jsonl"

    write_per_question(rows, path)

    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(written) == 1
    assert written[0]["record"]["question_id"] == "q1"
    assert written[0]["answer_f1"] == 1.0


def test_load_completed_configs_empty_when_file_missing(tmp_path):
    assert load_completed_configs(tmp_path / "missing.csv") == set()


def test_load_completed_configs_reads_param_value_pairs(tmp_path):
    out = tmp_path / "results.csv"
    out.write_text("param,value,answer_f1\n"
                   "baseline,,0.5\n"
                   "retrieval_threshold,0.35,0.4\n")

    assert load_completed_configs(out) == {("baseline", ""), ("retrieval_threshold", "0.35")}


def test_check_schema_compatible_allows_missing_file(tmp_path):
    check_schema_compatible(tmp_path / "missing.csv", ["a", "b"])  # no raise


def test_check_schema_compatible_allows_matching_header(tmp_path):
    out = tmp_path / "results.csv"
    out.write_text("a,b\n1,2\n")
    check_schema_compatible(out, ["a", "b"])  # no raise


def test_check_schema_compatible_rejects_old_schema(tmp_path):
    out = tmp_path / "results.csv"
    out.write_text("param,value,answer_f1,recall@10,mrr\n")
    with pytest.raises(SystemExit):
        check_schema_compatible(out, build_fieldnames())
