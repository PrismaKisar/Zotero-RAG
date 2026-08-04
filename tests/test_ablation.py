from benchmark.ablation import build_configs, load_gold_chunks, sample_questions


def test_build_configs_includes_baseline_and_one_variant_per_grid_value():
    baseline = {"a": 1, "b": 2}
    grid = {"a": [10, 20], "b": [99]}

    configs = build_configs(baseline, grid)

    assert configs[0] == {"param": "baseline", "value": None, "config": {"a": 1, "b": 2}}
    assert {"param": "a", "value": 10, "config": {"a": 10, "b": 2}} in configs
    assert {"param": "a", "value": 20, "config": {"a": 20, "b": 2}} in configs
    assert {"param": "b", "value": 99, "config": {"a": 1, "b": 99}} in configs
    assert len(configs) == 4


def test_build_configs_never_mutates_baseline():
    baseline = {"a": 1}
    build_configs(baseline, {"a": [2]})
    assert baseline == {"a": 1}


def test_load_gold_chunks_maps_paper_id_to_pdf_hash(tmp_path):
    aligned = tmp_path / "golden_set_aligned.jsonl"
    aligned.write_text(
        '{"paper_id": "1601.02403", "question_id": "q1", '
        '"aligned_chunks": {"ev1": [{"chunk_index": 3, "overlap": 1.0}], '
        '"ev2": [{"chunk_index": 5, "overlap": 0.9}]}}\n'
    )
    hash_map = {"1601.02403": "abc123"}

    gold = load_gold_chunks(aligned, hash_map)

    assert gold == {"q1": {("abc123", 3), ("abc123", 5)}}


def test_sample_questions_is_deterministic_for_a_given_seed():
    questions = [{"question_id": str(i)} for i in range(20)]
    first = sample_questions(questions, 5, seed=1)
    second = sample_questions(questions, 5, seed=1)
    assert first == second
    assert len(first) == 5


def test_sample_questions_returns_all_when_n_exceeds_pool():
    questions = [{"question_id": str(i)} for i in range(3)]
    assert sample_questions(questions, 10, seed=1) == questions


def test_load_gold_chunks_skips_papers_missing_from_hash_map(tmp_path):
    aligned = tmp_path / "golden_set_aligned.jsonl"
    aligned.write_text(
        '{"paper_id": "9999.9999", "question_id": "q1", '
        '"aligned_chunks": {"ev1": [{"chunk_index": 0, "overlap": 1.0}]}}\n'
    )

    gold = load_gold_chunks(aligned, {})

    assert gold == {}
