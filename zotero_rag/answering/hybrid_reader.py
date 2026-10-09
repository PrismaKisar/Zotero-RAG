"""Hybrid reader: the generative reader chooses, the extractive one locates.

The generative reader attributes better than the extractive one but marks whole
chunks, since it has no span; the extractive reader marks the sentences its span
falls in. Here the generated answer decides *which* chunks are evidence, and the
extractive reader runs on those chunks alone to decide *where* in each the mark
goes. The answer text stays the generated one.
"""

from dataclasses import replace

from zotero_rag.models import YELLOW, Answer, RerankedChunk


class HybridReader:
    """A ``models.Reader`` composed of a generative and an extractive one."""

    def __init__(self, generative, extractive):
        self.generative = generative
        self.extractive = extractive
        self.reader_kind = f"hybrid:{generative.reader_kind}:{extractive.reader_kind}"

    @property
    def last_quote_stats(self) -> dict[str, int]:
        return self.generative.last_quote_stats

    @property
    def generation_failures(self) -> int:
        return self.generative.generation_failures

    @generation_failures.setter
    def generation_failures(self, value: int) -> None:
        self.generative.generation_failures = value

    def extract_answers(self, question: str, candidates: list[RerankedChunk],
                        config: dict, color: tuple[float, float, float] = YELLOW,
                        progress_callback=None) -> list[Answer]:
        """One answer per chunk the generated answer cites, marked at the
        extractive span inside it, or over the whole chunk where the extractive
        reader finds none there."""
        generated = self.generative.extract_answers(question, candidates, config, color,
                                                    progress_callback)
        if not generated:
            return []
        cited = {answer.context for answer in generated}
        spans = self.extractive.extract_answers(
            question, [c for c in candidates if c.chunk.text in cited], config, color)
        # Best span per chunk: the extractive answers arrive best first.
        located = {}
        for span in spans:
            located.setdefault(span.context, span)
        return [replace(located[answer.context], text=answer.text)
                if answer.context in located else answer
                for answer in generated]
