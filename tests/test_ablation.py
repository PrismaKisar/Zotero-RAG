import json
import types

import pytest

from benchmark.ablation import (
    CI_METRICS,
    HARNESS_PARAMS,
    RECALL_K,
    RECALL_KS,
    answer_exact_match,
    answer_f1,
    apply_reader,
    attributed_ids,
    build_configs,
    build_fieldnames,
    check_corpus_indexed,
    check_schema_compatible,
    csv_row,
    load_completed_configs,
    load_gold_chunks,
    reader_key,
    retrieval_trace,
    run_config,
    run_oracle,
    sample_questions,
    split_harness_params,
    wanted_reader,
    write_per_question,
)


class _FakeChunk:
    def __init__(self, pdf_hash, chunk_index, text=""):
        self.pdf_hash = pdf_hash
        self.chunk_index = chunk_index
        self.text = text


class _FakeReranked:
    def __init__(self, pdf_hash, chunk_index, text=""):
        self.chunk = _FakeChunk(pdf_hash, chunk_index, text)


class _FakeAnswer:
    def __init__(self, text="some answer", context="ctx-1"):
        self.text = text
        self.context = context


class _FakeRag:
    """Stubs the two stages scoring reads: last_candidates (pre-rerank) and
    last_reranked (post-rerank), which can carry different orderings."""

    def __init__(self, answers=None):
        self._answers = [_FakeAnswer()] if answers is None else answers
        self.seen_call = None
        self.last_candidates = [
            {"chunk": _FakeChunk("h", 2, "ctx-2"), "retrieval_score": 0.9},
            {"chunk": _FakeChunk("h", 1, "ctx-1"), "retrieval_score": 0.5},
        ]
        self.last_reranked = [_FakeReranked("h", 1, "ctx-1"), _FakeReranked("h", 2, "ctx-2")]

    def answer_question(self, question, question_type, overrides, num_paraphrases,
                        pdf_hashes=None):
        self.seen_call = {"overrides": overrides, "num_paraphrases": num_paraphrases,
                          "pdf_hashes": pdf_hashes}
        return self._answers


GOLD_ANSWERS = {"q1": [{"answer": "some answer", "type": "extractive"}]}
QUESTIONS = [{"question_id": "q1", "question": "What is the model?"}]


def test_build_configs_includes_baseline_oracle_and_one_variant_per_grid_value():
    baseline = {"a": 1, "b": 2}
    grid = {"a": [10, 20], "b": [99]}

    configs = build_configs(baseline, grid)

    assert configs[0] == {"param": "baseline", "value": None, "config": {"a": 1, "b": 2}}
    assert [c["param"] for c in configs[:3]] == ["baseline", "oracle_paper", "oracle_context"]
    assert {"param": "a", "value": 10, "config": {"a": 10, "b": 2}} in configs
    assert {"param": "b", "value": 99, "config": {"a": 1, "b": 99}} in configs
    assert len(configs) == 6


def test_build_configs_never_mutates_baseline():
    baseline = {"a": 1}
    build_configs(baseline, {"a": [2]})
    assert baseline == {"a": 1}


def test_grid_ablates_parameters_that_reorder_results():
    """Regression guard: a threshold-only grid cannot move recall@k."""
    from benchmark.ablation import GRID
    assert {"retrieval_mode", "rerank_enabled", "result_limit"} <= set(GRID)


def test_grid_ablates_the_reader_as_well_as_retrieval():
    """The reader axes answer a different question than the retrieval ones."""
    from benchmark.ablation import GRID
    assert {"num_paraphrases", "qa_model", "reader"} <= set(GRID)


def test_split_harness_params_keeps_harness_axes_out_of_the_pipeline_config():
    config = {"retrieval_mode": "dense", "num_paraphrases": 2, "reader": "generative"}
    pipeline_config, harness = split_harness_params(config)

    assert pipeline_config == {"retrieval_mode": "dense"}
    assert harness == {"num_paraphrases": 2, "reader": "generative"}


def test_split_harness_params_omits_axes_the_config_leaves_at_baseline():
    pipeline_config, harness = split_harness_params({"result_limit": 10})
    assert pipeline_config == {"result_limit": 10}
    assert harness == {}


def test_harness_params_are_not_question_preset_fields():
    """They must be split out: resolve() would carry them into every stage's config."""
    from question_presets import PRESETS
    assert not set(HARNESS_PARAMS) & set(PRESETS["general"])


def test_run_config_forwards_the_paraphrase_axis_without_leaking_it():
    rag = _FakeRag()
    run_config(rag, QUESTIONS, {"retrieval_mode": "dense", "num_paraphrases": 2},
               {"q1": {("h", 1)}}, GOLD_ANSWERS)

    assert rag.seen_call["num_paraphrases"] == 2
    assert rag.seen_call["overrides"] == {"retrieval_mode": "dense"}


def test_run_config_defaults_the_paraphrase_axis_to_off():
    rag = _FakeRag()
    run_config(rag, QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS)
    assert rag.seen_call["num_paraphrases"] == 0


def test_reader_key_tells_the_two_reader_kinds_apart():
    extractive = types.SimpleNamespace(model_name="deepset/roberta-base-squad2")
    generative = types.SimpleNamespace(reader_kind="generative", model_name="llama3.2:3b")

    assert reader_key(extractive) == "deepset/roberta-base-squad2"
    assert reader_key(generative) == "generative"


BASELINE_MODEL = "deepset/deberta-v3-large-squad2"


def test_wanted_reader_defaults_to_the_pipeline_as_built():
    assert wanted_reader({}, BASELINE_MODEL) == BASELINE_MODEL


def test_wanted_reader_follows_the_qa_model_axis():
    assert wanted_reader({"qa_model": "deepset/roberta-base-squad2"},
                         BASELINE_MODEL) == "deepset/roberta-base-squad2"


def test_wanted_reader_lets_the_generative_arm_override_the_model_axis():
    assert wanted_reader({"reader": "generative", "qa_model": BASELINE_MODEL},
                         BASELINE_MODEL) == "generative"


def test_apply_reader_is_a_no_op_when_the_loaded_reader_already_matches():
    """The early return must precede the heavy imports, or every config pays for them."""
    baseline = types.SimpleNamespace(model_name=BASELINE_MODEL)
    rag = types.SimpleNamespace(qa_engine=baseline)

    apply_reader(rag, baseline, {}, "http://localhost:11434")

    assert rag.qa_engine is baseline


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
    def __init__(self, chunks):
        self._chunks = chunks
        self.opened = False

    def open_connection(self):
        self.opened = True

    def close_connection(self):
        self.opened = False

    def fetch_chunks(self, ids):
        return [p for p in self._chunks if (p.pdf_hash, p.chunk_index) in set(ids)]


class _FakeQaEngine:
    def __init__(self, answers):
        self._answers = answers
        self.seen_candidates = None

    def extract_answers(self, question, candidates, config, question_variations=None):
        self.seen_candidates = candidates
        return self._answers


class _OracleRag:
    def __init__(self, chunks, answers):
        self.qdrant_manager = _FakeQdrant(chunks)
        self.qa_engine = _FakeQaEngine(answers)


def test_run_oracle_feeds_gold_chunks_to_the_reader():
    from models import Chunk

    gold_chunk = Chunk(text="gold text", page_number=1, chunk_index=1, title="T", pdf_hash="h")
    rag = _OracleRag([gold_chunk], [_FakeAnswer(text="some answer")])

    rows = run_oracle(rag, QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS)

    assert len(rows) == 1
    assert rows[0]["answer_f1"] == 1.0
    # the reader saw exactly the gold chunk, never the retriever's output
    assert [c.chunk.text for c in rag.qa_engine.seen_candidates] == ["gold text"]
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


class _FakeClient:
    def __init__(self, exists=True, count=42):
        self._exists, self._count = exists, count

    def collection_exists(self, name):
        return self._exists

    def count(self, name):
        return types.SimpleNamespace(count=self._count)


class _FakeManager:
    def __init__(self, **kwargs):
        self.chunk_collection = "zotero_rag_test"
        self.client = _FakeClient(**kwargs)
        self.closed = False

    def open_connection(self):
        pass

    def close_connection(self):
        self.closed = True


def test_check_corpus_indexed_returns_chunk_count():
    rag = types.SimpleNamespace(qdrant_manager=_FakeManager(count=4978))
    assert check_corpus_indexed(rag) == 4978
    assert rag.qdrant_manager.closed


def test_check_corpus_indexed_rejects_missing_collection():
    rag = types.SimpleNamespace(qdrant_manager=_FakeManager(exists=False))
    with pytest.raises(SystemExit, match="does not exist"):
        check_corpus_indexed(rag)


def test_check_corpus_indexed_rejects_empty_collection():
    """An empty collection yields 0.0 on every metric, which reads as a real result."""
    rag = types.SimpleNamespace(qdrant_manager=_FakeManager(count=0))
    with pytest.raises(SystemExit, match="is empty"):
        check_corpus_indexed(rag)


def test_run_config_searches_the_whole_corpus_by_default():
    rag = _FakeRag()
    run_config(rag, QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS)
    assert rag.seen_call["pdf_hashes"] is None


def test_oracle_paper_scopes_retrieval_to_the_questions_own_papers():
    rag = _FakeRag()
    run_config(rag, QUESTIONS, {}, {"q1": {("h", 1), ("h", 4)}}, GOLD_ANSWERS,
               scope_to_gold_paper=True)
    assert rag.seen_call["pdf_hashes"] == ["h"]


def test_run_config_reports_every_protocol_k_not_just_the_deepest():
    """The first campaign could only report k=10; recall@1 is the primary metric."""
    row = run_config(_FakeRag(), QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS)[0]
    for k in RECALL_KS:
        assert f"recall@{k}" in row and f"precision@{k}" in row
    # gold (h,1) sits second in the pre-rerank order, so it is missed at k=1
    assert row["recall@1"] == 0.0
    assert row["recall@3"] == 1.0


def test_run_config_stores_the_ranked_ids_so_new_metrics_need_no_rerun():
    row = run_config(_FakeRag(), QUESTIONS, {}, {"q1": {("h", 1)}}, GOLD_ANSWERS)[0]
    assert row["record"]["ranked_ids"] == [["h", 2], ["h", 1]]
    assert row["record"]["gold_ids"] == [["h", 1]]


def test_retrieval_trace_separates_wrong_paper_from_wrong_chunk():
    right_paper = retrieval_trace([("h", 9)], [], set(), {("h", 1)})
    wrong_paper = retrieval_trace([("other", 9)], [], set(), {("h", 1)})
    assert right_paper["gold_paper_in_top10"]
    assert not wrong_paper["gold_paper_in_top10"]


def test_answer_exact_match_ignores_articles_case_and_punctuation():
    refs = [{"answer": "The BERT model.", "type": "extractive"}]
    assert answer_exact_match("bert model", refs) == 1.0
    assert answer_exact_match("a bert model variant", refs) == 0.0


def test_answer_exact_match_without_references():
    assert answer_exact_match("anything", []) == 0.0
