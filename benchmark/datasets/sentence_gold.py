"""Highlight precision below the chunk: how much of the ink lands on gold sentences.

``highlight_precision`` (ablation.py) counts a mark anywhere inside a gold chunk
as wholly on-gold, so it cannot tell a reader that marks the answering sentence
from one that marks the whole paragraph around it, which is the difference the
hybrid reader exists to make. Here the gold unit is the sentence: a sentence of a
gold chunk is gold when it contains one of the question's annotated answer
spans. Only QASPER annotates spans, so the metric is absent on QASA, and absent
for a question none of whose spans can be found in the PDF text.

A secondary metric: the paragraph level stays the primary one.
"""

from benchmark.metrics.qasper_evaluator import normalize_answer


def sentence_intervals(text: str, sentences) -> list[tuple[int, int, str]]:
    """``(start, end, sentence)`` for each sentence found in ``text``, in order.

    Sentences are located by scanning, as ``Chunk.sentence_coords_within`` does,
    and one the extractor altered is skipped rather than misplacing the rest.
    """
    found, cursor = [], 0
    for sentence, _ in sentences:
        at = text.find(sentence, cursor)
        if at < 0:
            continue
        cursor = at + len(sentence)
        found.append((at, cursor, sentence))
    return found


def gold_intervals(text: str, sentences, gold_spans: list[str]) -> list[tuple[int, int]]:
    """Character intervals of the sentences of ``text`` containing a gold span."""
    spans = [normalize_answer(s) for s in gold_spans]
    spans = [s for s in spans if s]
    return [(start, end) for start, end, sentence in sentence_intervals(text, sentences)
            if any(s in normalize_answer(sentence) for s in spans)]


def overlap(start: int, end: int, intervals) -> int:
    return sum(max(0, min(end, e) - max(start, s)) for s, e in intervals)


def sentence_highlight_precision(marks, chunks: dict, gold_ids, gold_spans: list[str]) -> dict:
    """Share of the marked characters that fall on gold sentences.

    Args:
        marks: ``(chunk_id, start, end)`` triples, deduplicated, as the page shows them.
        chunks: chunk id -> Chunk, for every chunk a mark can sit in.
        gold_ids: the question's gold chunk ids.
        gold_spans: the question's annotated answer spans.

    Returns:
        ``{"highlight_sentence_precision": p}``, or ``{}`` when the question has
        no span located in any gold chunk it can see.
    """
    if not gold_spans:
        return {}
    gold = {chunk_id: gold_intervals(chunks[chunk_id].text, chunks[chunk_id].sentences,
                                     gold_spans)
            for chunk_id in gold_ids if chunk_id in chunks}
    if not any(gold.values()):
        return {}
    total = sum(end - start for _, start, end in marks)
    on_gold = sum(overlap(start, end, gold.get(chunk_id, []))
                  for chunk_id, start, end in marks)
    return {"highlight_sentence_precision": on_gold / total if total else 0.0}
