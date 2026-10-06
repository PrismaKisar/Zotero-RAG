"""The pipeline's settings: one fixed configuration, measured, not user-facing.

There are no question types and nothing for the user to tune. Every value below
was measured on the golden sets, or fixed because measuring showed that the
alternative did not help; exposing them would let a user undo a calibration they
cannot see. ``resolve`` exists for the benchmarks alone, which vary one value at
a time around these (benchmark/ablation.py).

Stage 1, retrieval: ``retrieval_mode`` hybrid and ``result_limit`` 30, the
reranker's budget. The dense side is EmbeddingGemma-300m, BM25 the sparse one:
against the former bge-base + BM25 it adds 0.05 recall@10 over the QASA library
(+0.049 [+0.029, +0.072], 604 questions) and 0.10 recall@30 over QASPER's, and
every dense model alone lost to every hybrid (benchmark/retrieval_models.py).
There is no score threshold. A ``retrieval_threshold`` on the
dense cosine used to sit there and was removed rather than re-tuned: measured
with exact search (benchmark/retrieval_score_distribution.py, on bge-base), the 30th
candidate of every QASPER question scores 0.55 or more, so the shipped 0.35-0.45
never cut a thing; inside the band where a cut does act, gold evidence scores
*below* the non-gold candidates it competes with, and the scale shifts between
datasets.

Stages 2 and 3 decide what gets highlighted, and they decide two different
things. ``rerank_gate`` is read off the *best* candidate alone: no candidate
reaching 0.85 means the library does not answer this question and nothing is
highlighted, however many middling candidates it has. A question that clears the
gate then keeps every candidate within ``rerank_margin`` of that best one. The
gate was 0.88 until the development sweep (runs/dev/ablation), where 0.85 was
the one change that helped on both sets, in paired tests: +0.011 [+0.001,
+0.025] evidence F1 on QASA, +0.006 [+0.000, +0.014] on QASPER, with 0.91
worse on both.

The cut is relative because an absolute one does not transfer between questions:
gte's probabilities crowd into 0.80-0.93, and the crowd sits at a different
height per question, so a fixed cut lands mid-crowd on some and above it on
others. Measured (benchmark/failure_attribution.py), the absolute 0.90 it
replaces threw away gold evidence the reranker had itself ranked first, on 23%
of the answerable questions of both sets, and that was the single largest cause
of empty answers. The pair was searched against both golden sets and against
questions the library does not answer (benchmark/answer_gate.py); against the
absolute 0.90/0.05 it is worth +0.161 [+0.133, +0.189] evidence F1 on QASA and
+0.019 [-0.016, +0.055] on QASPER, raising precision and recall together, and it
leaves 7% rather than 30% of answerable QASA questions empty. The end-to-end
campaign found what the offline search predicted, +0.160 [+0.134, +0.188] on
QASA, with attribution precision +0.186 and no more text highlighted, and it cut
latency from 3.1s to 2.5s a question by handing the reader fewer chunks. On 340
held-out QASPER questions, tuned on nothing, it is +0.017 [+0.000, +0.034]: the
direction holds where the gain is small.

The reader is the extractive one, chosen over the generative reader with the
one-sentence rule by crossing the two with the cut (runs/archive/output_*/crossed). Under the
relative cut the extractive reader attributes better on QASA, +0.030 [+0.010,
+0.049] evidence F1 and +0.113 recall, marking half the text at a third of the
latency; on QASPER the generative reader's answer-F1 edge, +0.027 [-0.001,
+0.055], does not separate, and neither does any attribution difference. The
relative cut was measured on the extractive reader and helps the generative one
far less (+0.030 evidence F1 on QASA against +0.160), so the two effects do not
add and the pair adopted is the one measured together.

``qa_score_threshold`` is 0, which switches the reader's veto off: it is kept as
a setting because the ablation varies it, and because with the relative cut it
only removed evidence. What buys the silence is the gate, which leaves 97.5% of
the 40 off-topic and nonsense questions with nothing highlighted; the remaining
one is the accepted cost of the evidence the old cut was destroying. Every value
here is on the scales of gte-reranker-modernbert-base and
deberta-v3-large-squad2, which is why neither model is configurable: changing one
means re-running the search.

Two filters were measured and removed. Section diversity, interleaving
introduction and method answers, cost answer F1 on QASPER (0.123 to 0.092) and
left attribution where it was. ``min_answer_words`` dropped a chunk whose span
was shorter than N words, but the highlight is that span widened to whole
sentences, so a correct short answer threw away the right evidence (+0.19
evidence recall at 1 against 3, on both sets).
"""

SETTINGS = {
    "retrieval_mode": "hybrid",
    "result_limit": 30,
    "rerank_enabled": True,
    "rerank_gate": 0.85,
    "rerank_margin": 0.02,
    "qa_score_threshold": 0.0,
}


def resolve(overrides: dict | None = None) -> dict:
    """The settings with ``overrides`` on top; only the benchmarks pass any."""
    return {**SETTINGS, **(overrides or {})}
