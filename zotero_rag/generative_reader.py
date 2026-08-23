"""Abstractive reader: one generated answer, attributed to the chunks it cites.

The extractive engine can only return a span occurring verbatim in a chunk, so
every QASPER answer phrased as a synthesis of several passages is unreachable by
construction — the oracle-context row measures that ceiling, this reader is the
other arm of the comparison.

It emits the same ``Answer`` shape as ``QAEngine.extract_answers``, one per cited
chunk, so the ablation needs no special case: ``answers[0].text`` is scored by
Answer F1 and the per-chunk copies are what evidence precision/recall see.
"""

import logging
import re

import ollama
from models import Answer, RerankedChunk

logger = logging.getLogger(__name__)

DEFAULT_GENERATIVE_MODEL = "llama3.2:3b"
DEFAULT_CONTEXT_CHUNKS = 8

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
SOURCES_LINE_PATTERN = re.compile(r"(?im)^\s*sources?\s*:.*$")
# Removing an inline "[3]" leaves the space in front of it stranded before the
# punctuation that followed.
ORPHANED_SPACE_PATTERN = re.compile(r"\s+([.,;:!?])")
NO_ANSWER = "NO ANSWER"

PROMPT_TEMPLATE = (
    "You answer questions about scientific papers using only the numbered passages below.\n\n"
    "<passages>\n{passages}\n</passages>\n\n"
    f"Question: {{question}}\n\n"
    "Rules:\n"
    "- Answer in at most two sentences, using only what the passages state.\n"
    "- Cite every passage you relied on inline as [1], [2], and so on.\n"
    f"- If the passages do not answer the question, reply exactly: {NO_ANSWER}\n\n"
    "Answer:"
)


def parse_response(raw: str, num_chunks: int) -> tuple[str, list[int]]:
    """Split the model's reply into answer text and the chunks it cited.

    Args:
        raw: The model's raw reply.
        num_chunks: How many passages the prompt offered, to reject stray indices.

    Returns:
        ``(answer_text, cited_indices)`` where the indices are 0-based, deduplicated
        and in citation order. The text is empty when the model declined to answer.
    """
    cited = []
    for match in CITATION_PATTERN.findall(raw):
        index = int(match) - 1
        if 0 <= index < num_chunks and index not in cited:
            cited.append(index)

    text = SOURCES_LINE_PATTERN.sub("", raw)
    text = CITATION_PATTERN.sub("", text)
    text = ORPHANED_SPACE_PATTERN.sub(r"\1", " ".join(text.split()))

    if text.upper().startswith(NO_ANSWER):
        return "", []
    return text, cited


class GenerativeReader:
    """Answers from the retrieved chunks with a local LLM, citing what it used.

    Duck-types the two attributes ``ZoteroRAG.answer_question`` reads off its
    reader — ``enable_question_expansion`` and ``extract_answers`` — so it drops
    into the pipeline and the ablation harness with no special case. Expansion is
    off because it belongs to the extractive engine's T5 paraphraser, which this
    reader does not own.
    """

    enable_question_expansion = False
    # Lets the ablation tell readers apart without importing this module eagerly.
    reader_kind = "generative"

    def __init__(self,
                ollama_url: str = "http://localhost:11434",
                model_name: str = DEFAULT_GENERATIVE_MODEL,
                max_context_chunks: int = DEFAULT_CONTEXT_CHUNKS):
        """Initialize the reader and fail fast if the model is not pulled.

        Args:
            ollama_url: URL of the Ollama service.
            model_name: Ollama model tag to generate with.
            max_context_chunks: Top-ranked chunks to put in the prompt.

        Raises:
            RuntimeError: If Ollama is unreachable or lacks ``model_name``.
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.max_context_chunks = max_context_chunks
        self.client = ollama.Client(host=ollama_url)
        self._require_model()
        logger.info(f"GenerativeReader initialized with {model_name} at {ollama_url}")

    def _require_model(self) -> None:
        """Raise unless Ollama is up and serving ``self.model_name``.

        A missing model otherwise surfaces as an empty answer for every question,
        which reads as a real (terrible) result rather than a setup mistake.
        """
        try:
            available = {entry.model for entry in self.client.list().models}
        except Exception as exc:  # any transport failure is the same setup error
            raise RuntimeError(
                f"Ollama unreachable at {self.ollama_url}: {exc}. "
                f"Start it, then run `ollama pull {self.model_name}`."
            ) from exc

        if self.model_name not in available:
            raise RuntimeError(
                f"Ollama at {self.ollama_url} has no model {self.model_name!r} "
                f"(available: {sorted(available) or 'none'}). "
                f"Run `ollama pull {self.model_name}`."
            )

    def _build_prompt(self, question: str, candidates: list[RerankedChunk]) -> str:
        """Render the numbered-passage prompt for ``question``."""
        passages = "\n\n".join(f"[{i + 1}] {c.chunk.text}" for i, c in enumerate(candidates))
        return PROMPT_TEMPLATE.format(passages=passages, question=question)

    def _to_answer(self, text: str, candidate: RerankedChunk,
                question: str, color: tuple[float, float, float]) -> Answer:
        """Wrap the generated ``text`` as an Answer attributed to ``candidate``."""
        chunk = candidate.chunk
        return Answer(
            text=text,
            context=chunk.text,
            page_number=chunk.page_number,
            title=chunk.title,
            section=chunk.section,
            start_char=0,
            end_char=len(chunk.text),
            # ponytail: a generated answer has no span probability, so
            # qa_score_threshold does not apply to this reader; the citation is
            # the whole chunk, which is also what the highlighter marks.
            score=1.0,
            query=question,
            color=color,
            sentence_coords=[coords for _, coords in chunk.sentences if coords],
            retrieval_score=candidate.retrieval_score,
            rerank_score=candidate.rerank_score,
            pdf_hash=chunk.pdf_hash,
        )

    def extract_answers(self,
                    question: str,
                    candidates: list[RerankedChunk],
                    config: dict,
                    color: tuple[float, float, float] = (1, 1, 0),
                    progress_callback=None,
                    question_variations: list[str] | None = None) -> list[Answer]:
        """Generate one answer over the top candidates, attributed to its citations.

        Args:
            question: The question to answer.
            candidates: Reranked chunks, best first.
            config: Resolved question-type config. Unused: no knob in it applies
                to a generated answer, but the signature matches ``QAEngine``.
            color: Highlight color for the answer (R, G, B).
            progress_callback: Function(current, total, message) for progress updates.
            question_variations: Unused; this reader always asks the original question.

        Returns:
            One Answer per cited chunk, or an empty list if the model declined.
        """
        if not candidates:
            return []

        used = candidates[:self.max_context_chunks]
        prompt = self._build_prompt(question, used)
        if progress_callback:
            progress_callback(0, 1, f"Generating an answer over {len(used)} chunks")

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": 256,
                    "num_ctx": max(2048, min(len(prompt) // 4 + 512, 16384)),
                },
                keep_alive=-1,
            )
            raw = response["response"]
        except Exception as exc:  # noqa: BLE001 - a failed generation is no answer
            logger.error(f"Generation failed for question {question!r}: {exc}")
            return []

        text, cited = parse_response(raw, len(used))
        if not text:
            logger.debug(f"No answer generated for question {question!r}")
            return []

        # An uncited answer still came from the context, so attribute it to the
        # top-ranked chunk rather than dropping it and scoring a false zero.
        if not cited:
            cited = [0]

        if progress_callback:
            progress_callback(1, 1, f"Generated an answer citing {len(cited)} chunks")

        return [self._to_answer(text, used[i], question, color) for i in cited]
