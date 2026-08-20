import json

from benchmark.compare_datasets import (
    analyze_dataset,
    cliffs_delta,
    coverage,
    flesch_kincaid,
    jaccard,
    load_dataset,
    mann_whitney_p,
    question_form,
    to_markdown,
)


def test_question_form():
    assert question_form("Why does the model fail?") == "why (causal)"
    assert question_form("How were the judgements assembled?") == "how (procedural)"
    assert question_form("What is the architecture?") == "what/which (factual)"
    assert question_form("Is the model pretrained?") == "yes/no (verification)"
    assert question_form("Authors used DropPath. What changed?") == "other"


def test_overlap_metrics():
    assert jaccard(["neural", "model"], ["neural", "model"]) == 1.0
    assert jaccard(["neural"], ["corpus"]) == 0.0
    # stopwords are ignored on the question side
    assert coverage(["what", "is", "bleu"], ["bleu", "score"]) == 1.0


def test_readability_and_stats():
    easy = "The cat sat on the mat. The dog ran. " * 5
    hard = "Consequently, the heterogeneous representational configuration necessitates " \
           "an exceptionally sophisticated optimization methodology throughout evaluation. " * 5
    assert flesch_kincaid(easy) < flesch_kincaid(hard)
    assert flesch_kincaid("") is None

    low, high = list(range(20)), list(range(100, 120))
    assert mann_whitney_p(low, high) < 0.001
    assert cliffs_delta(low, high) == -1.0
    assert cliffs_delta(high, low) == +1.0
    assert mann_whitney_p([1, 2], [3, 4]) is None  # too small to approximate


def _fixture(tmp_path):
    (tmp_path / "chunks").mkdir()
    (tmp_path / "chunks" / "p1.json").write_text(json.dumps([
        "The model uses an attention mechanism over the encoder states.",
        "We report a BLEU score of 32.4 on the test set (Smith et al., 2019).",
    ]))
    (tmp_path / "golden_set_aligned.jsonl").write_text(json.dumps({
        "paper_id": "p1",
        "title": "T",
        "question_id": "q1",
        "question": "Why is attention used?",
        "gold_spans": ["attention mechanism"],
        "evidence": ["The model uses an attention mechanism over the encoder states."],
        "multi_evidence": False,
        "aligned_chunks": {
            "The model uses an attention mechanism over the encoder states.":
                [{"chunk_index": 0, "overlap": 1.0}],
        },
    }) + "\n")
    return tmp_path


def test_analyze_dataset_and_to_markdown(tmp_path):
    metrics, cats = analyze_dataset(*load_dataset(_fixture(tmp_path)))

    assert metrics["paper_chunks"] == [2]
    assert metrics["q_words"] == [4]
    assert metrics["ev_spans"] == [1]
    assert metrics["aligned_chunks"] == [1]
    assert metrics["spread"] == [0.0]
    assert metrics["position"] == [0.0]
    assert metrics["cite_density"][0] > 0
    assert cats["question_form"]["why (causal)"] == 1
    assert cats["extractive"]["short extractive span"] == 1

    md = to_markdown({"A": (metrics, cats)})
    assert "| Questions | 1 |" in md
    assert "why (causal) | 1 (100.0%)" in md
