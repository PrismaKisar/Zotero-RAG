import math

import numpy as np

from benchmark.qa_score_distribution import (
    auc,
    chunk_scores,
    report,
    span_scores,
    sweep,
)

# [CLS], two question tokens, [SEP], three context tokens, [SEP]
SEQUENCE_IDS = [None, 0, 0, None, 1, 1, 1, None]


def _logits(cls_logit):
    start = np.array([cls_logit, 0, 0, 0, 3.0, 0.5, 0.1, 0])
    end = np.array([cls_logit, 0, 0, 0, 0.2, 3.0, 0.1, 0])
    return start, end


def test_keeping_no_answer_rescales_the_span_without_moving_it():
    start, end = _logits(cls_logit=2.0)
    s = span_scores(start, end, SEQUENCE_IDS)
    assert (s["start"], s["end"]) == (4, 5)
    context = [4, 5, 6]
    p_start_cls = math.exp(2.0) / (math.exp(2.0) + np.exp(start[context]).sum())
    p_end_cls = math.exp(2.0) / (math.exp(2.0) + np.exp(end[context]).sum())
    assert math.isclose(s["with_null"], s["masked"] * (1 - p_start_cls) * (1 - p_end_cls))
    assert math.isclose(s["null"], p_start_cls * p_end_cls)


def test_the_former_score_ignores_how_sure_the_model_is_that_there_is_no_answer():
    unsure, sure = span_scores(*_logits(-5.0), SEQUENCE_IDS), span_scores(*_logits(8.0), SEQUENCE_IDS)
    assert math.isclose(unsure["masked"], sure["masked"])
    assert sure["with_null"] < 0.01 < unsure["with_null"]


def test_chunk_scores_take_the_best_window_and_the_least_no_answer():
    windows = [{"masked": 0.4, "with_null": 0.3, "null": 0.5, "start": 1, "end": 2},
               None,
               {"masked": 0.6, "with_null": 0.1, "null": 0.2, "start": 3, "end": 4}]
    assert chunk_scores(windows) == {"masked": 0.6, "with_null": 0.3, "null": 0.2,
                                     "start": 3, "end": 4}
    assert chunk_scores([None]) is None


def test_auc_is_one_when_gold_always_wins_and_half_when_nothing_separates():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert auc([0.1], [0.9]) == 0.0


ROWS = [
    {"question_id": "a", "gold": True, "masked": 0.9, "with_null": 0.8, "null": 0.0},
    {"question_id": "a", "gold": False, "masked": 0.3, "with_null": 0.05, "null": 0.5},
    {"question_id": "b", "gold": True, "masked": 0.1, "with_null": 0.1, "null": 0.0},
    {"question_id": "b", "gold": False, "masked": 0.05, "with_null": 0.01, "null": 0.9},
]


def test_sweep_counts_chunks_and_questions():
    [row] = sweep(ROWS, "masked", thresholds=(0.2,))
    assert row["gold_kept"] == 0.5 and row["other_kept"] == 0.5
    assert row["precision"] == 0.5
    assert row["kept_per_question"] == 1.0
    assert row["questions_gold_kept"] == 0.5
    assert row["questions_empty"] == 0.5


def test_report_renders_distribution_separation_and_both_sweeps():
    text = report("set", ROWS, n_questions=2)
    for part in ("## Distribution", "## Separation", "## Sweep: masked",
                 "## Sweep: with_null", "2 gold and 2 non-gold"):
        assert part in text
