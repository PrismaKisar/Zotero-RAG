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

DEFAULT_GENERATIVE_MODEL = "qwen3.5:2b"
DEFAULT_CONTEXT_CHUNKS = 8

# Qwen3.5 ships a reasoning mode that is on by default on some paths. It is
# pinned off here rather than left to the server's default: it would multiply
# latency, and a model that deliberates before answering is a different
# component, not the same one better configured. Ollama ignores the flag on
# models without the capability, so this stays correct if the model changes.
THINKING = False

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
SOURCES_LINE_PATTERN = re.compile(r"(?im)^\s*sources?\s*:.*$")
# Removing an inline "[3]" leaves the space in front of it stranded before the
# punctuation that followed.
ORPHANED_SPACE_PATTERN = re.compile(r"\s+([.,;:!?])")
NO_ANSWER = "NO ANSWER"

# Citation-quote modes. "off" is the reader as measured so far: a citation names
# a chunk and the whole chunk is marked. The other two ask the model to reproduce
# the sentence it used, which is what turns the citation from a claim into a
# checkable fact - "lenient" keeps an unverifiable citation and marks the whole
# chunk as before, "strict" discards it.
QUOTE_OFF = "off"
QUOTE_LENIENT = "lenient"
QUOTE_STRICT = "strict"

QUOTE_BLOCK_PATTERN = re.compile(r"(?ims)^[ \t]*quotes?[ \t]*:[ \t]*$(.*)\Z")
# The closing delimiter is optional because a model that opens a quote and runs
# out of tokens still produced a usable prefix to look for.
QUOTE_ENTRY_PATTERN = re.compile(r"(?m)^\s*\[(\d+)\]\s*[\"“]?(.+?)[\"”]?\s*$")

# How long the answer is asked to be. This is a prompt rule, not a property of
# the reader: Exact Match is 0.000 under every configuration either campaign has
# measured, and against QASPER references that are frequently noun phrases, a
# two-sentence answer cannot match one however correct it is. The default is the
# rule as shipped, so the axis varies against the reader already measured.
ANSWER_STYLES = {
    "two_sentences": "- Answer in at most two sentences, using only what the passages state.\n",
    "one_sentence": "- Answer in one sentence, using only what the passages state.\n",
    "phrase": ("- Answer with the shortest phrase that answers the question - a name, a\n"
               "  number, a noun phrase - using only what the passages state. Do not\n"
               "  write a sentence and do not repeat the question.\n"),
}
DEFAULT_ANSWER_STYLE = "two_sentences"

PROMPT_TEMPLATE = (
    "You answer questions about scientific papers using only the numbered passages below.\n\n"
    "<passages>\n{passages}\n</passages>\n\n"
    "Question: {question}\n\n"
    "Rules:\n"
    "{answer_rule}"
    "- Cite every passage you relied on inline as [1], [2], and so on.\n"
    f"- If the passages do not answer the question, reply exactly: {NO_ANSWER}\n"
    "{quote_rule}\n"
    "Answer:"
)

QUOTE_RULE = (
    "- After the answer, write a line containing only `Quotes:`, then one line per\n"
    "  passage you cited, of the form [n] \"sentence\" - the single sentence from\n"
    "  that passage supporting your answer, copied from it word for word.\n"
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

    text = QUOTE_BLOCK_PATTERN.sub("", raw)
    text = SOURCES_LINE_PATTERN.sub("", text)
    text = CITATION_PATTERN.sub("", text)
    text = ORPHANED_SPACE_PATTERN.sub(r"\1", " ".join(text.split()))

    if text.upper().startswith(NO_ANSWER):
        return "", []
    return text, cited


def parse_quotes(raw: str, num_chunks: int) -> dict[int, str]:
    """The sentence the model claims to have used, per 0-based chunk index.

    Only the block after the ``Quotes:`` marker is read. Citations also appear
    inline in the answer itself, and a line there beginning with "[2] ..." would
    otherwise be mistaken for a quote entry.
    """
    block = QUOTE_BLOCK_PATTERN.search(raw)
    if not block:
        return {}
    quotes = {}
    for number, quote in QUOTE_ENTRY_PATTERN.findall(block.group(1)):
        index = int(number) - 1
        if 0 <= index < num_chunks and index not in quotes and quote.strip():
            quotes[index] = quote.strip()
    return quotes


def locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Character span of ``quote`` inside ``text``, or None if it is not there.

    Matching is tolerant of whitespace and nothing else. Whitespace is not
    content: PDF extraction breaks sentences across lines, so a model that
    reproduced a sentence perfectly would still fail a literal ``str.find`` for
    a reason that has nothing to do with whether it quoted faithfully. Case and
    punctuation are left strict, because those are the model's own output.
    """
    tokens = [re.escape(token) for token in quote.split()]
    if not tokens:
        return None
    match = re.search(r"\s+".join(tokens), text)
    return (match.start(), match.end()) if match else None


def span_sentence_coords(chunk, span: tuple[int, int] | None) -> list:
    """Coordinates of the sentences ``span`` touches, or all of them if it is None.

    The highlighter draws per sentence, so narrowing the character span without
    narrowing the coordinates would report a tight citation and still paint the
    whole paragraph.

    ponytail: sentence offsets are recovered by scanning ``chunk.text`` rather
    than stored, since nothing else needs them; a sentence the extractor altered
    is skipped instead of misplacing every sentence after it.
    """
    if span is None:
        return [coords for _, coords in chunk.sentences if coords]
    start, end = span
    found_coords, cursor = [], 0
    for sentence, coords in chunk.sentences:
        at = chunk.text.find(sentence, cursor)
        if at < 0:
            continue
        cursor = at + len(sentence)
        if coords and at < end and cursor > start:
            found_coords.append(coords)
    return found_coords


class GenerativeReader:
    """Answers from the retrieved chunks with a local LLM, citing what it used.

    Duck-types the two attributes ``ZoteroRAG.answer_question`` reads off its
    reader — ``enable_question_expansion`` and ``extract_answers`` — so it drops
    into the pipeline and the ablation harness with no special case. Expansion is
    off because it belongs to the extractive engine's T5 paraphraser, which this
    reader does not own.
    """

    enable_question_expansion = False

    def __init__(self,
                ollama_url: str = "http://localhost:11434",
                model_name: str = DEFAULT_GENERATIVE_MODEL,
                max_context_chunks: int = DEFAULT_CONTEXT_CHUNKS,
                citation_quote: str = QUOTE_OFF,
                answer_style: str = DEFAULT_ANSWER_STYLE):
        """Initialize the reader and fail fast if the model is not pulled.

        Args:
            ollama_url: URL of the Ollama service.
            model_name: Ollama model tag to generate with.
            max_context_chunks: Top-ranked chunks to put in the prompt.
            citation_quote: One of QUOTE_OFF, QUOTE_LENIENT, QUOTE_STRICT.
                Defaults to off: an intervention that has not been accepted does
                not arrive as the shipped behaviour.
            answer_style: A key of ANSWER_STYLES, choosing how long the answer is
                asked to be. Defaults to the rule as shipped, for the same reason.

        Raises:
            RuntimeError: If Ollama is unreachable or lacks ``model_name``.
        """
        if citation_quote not in (QUOTE_OFF, QUOTE_LENIENT, QUOTE_STRICT):
            raise ValueError(f"unknown citation_quote {citation_quote!r}")
        if answer_style not in ANSWER_STYLES:
            raise ValueError(f"unknown answer_style {answer_style!r}; "
                             f"expected one of {sorted(ANSWER_STYLES)}")
        # A non-positive size slices the candidate list to nothing and the reader
        # answers every question from an empty prompt, which scores as a reader
        # that knows nothing rather than as a misconfiguration.
        if max_context_chunks < 1:
            raise ValueError(f"max_context_chunks must be >= 1, got {max_context_chunks}")
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.max_context_chunks = max_context_chunks
        self.citation_quote = citation_quote
        self.answer_style = answer_style
        # Lets the ablation tell readers apart without importing this module
        # eagerly. Every setting that changes what the model sees is part of the
        # identity: two prompts are two readers, and so are two context sizes,
        # and a shared load between them would score one twice.
        self.reader_kind = (f"generative:{citation_quote}:{answer_style}"
                            f":{max_context_chunks}")
        # Read by the ablation after each question; see quote_scores().
        self.last_quote_stats = {"cited": 0, "matched": 0}
        # Chunk texts whose quoted sentence was located, joined on the same key
        # attributed_ids() uses. This is what lets one lenient generation be
        # scored under both policies, instead of running strict as a second
        # generation that differs from the first by sampling as well as policy.
        self.last_verified_contexts = set()
        # Questions lost to a generation that failed twice. A benchmark scores
        # those as unanswered, so a run has to be able to say how many of its
        # zeros came from the service rather than from the reader.
        self.generation_failures = 0
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
        quote_rule = "" if self.citation_quote == QUOTE_OFF else QUOTE_RULE
        return PROMPT_TEMPLATE.format(passages=passages, question=question,
                                      answer_rule=ANSWER_STYLES[self.answer_style],
                                      quote_rule=quote_rule)

    def _to_answer(self, text: str, candidate: RerankedChunk,
                question: str, color: tuple[float, float, float],
                span: tuple[int, int] | None = None) -> Answer:
        """Wrap the generated ``text`` as an Answer attributed to ``candidate``.

        ``span`` is where the model's quoted sentence was found in the chunk, or
        None when there is no verified quote - in which case the citation is the
        whole chunk, which is also what the highlighter then marks.
        """
        chunk = candidate.chunk
        return Answer(
            text=text,
            context=chunk.text,
            page_number=chunk.page_number,
            title=chunk.title,
            section=chunk.section,
            start_char=span[0] if span else 0,
            end_char=span[1] if span else len(chunk.text),
            # ponytail: a generated answer has no span probability, so
            # qa_score_threshold does not apply to this reader.
            score=1.0,
            query=question,
            color=color,
            sentence_coords=span_sentence_coords(chunk, span),
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
        self.last_quote_stats = {"cited": 0, "matched": 0}
        self.last_verified_contexts = set()
        if not candidates:
            return []

        used = candidates[:self.max_context_chunks]
        prompt = self._build_prompt(question, used)
        if progress_callback:
            progress_callback(0, 1, f"Generating an answer over {len(used)} chunks")

        # ponytail: one retry, no backoff. The failure this exists for is the
        # Ollama runner dying and being restarted by the server, which the next
        # call finds already back up. It matters because a benchmark scores a
        # failed generation as an unanswered question - an infrastructure blip
        # would otherwise enter the results as a property of the configuration
        # being measured. A second failure is left to stand: at that point the
        # service is down rather than blipping, and retrying harder would hide
        # it for longer.
        raw = None
        for attempt in (1, 2):
            try:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    think=THINKING,
                    options={
                        "temperature": 0.1,
                        "num_predict": 256,
                        "num_ctx": max(2048, min(len(prompt) // 4 + 512, 16384)),
                    },
                    keep_alive=-1,
                )
                raw = response["response"]
                break
            except Exception as exc:  # noqa: BLE001 - a failed generation is no answer
                logger.error(f"Generation failed (attempt {attempt}) for question "
                             f"{question!r}: {exc}")
        if raw is None:
            self.generation_failures += 1
            return []

        text, cited = parse_response(raw, len(used))
        if not text:
            logger.debug(f"No answer generated for question {question!r}")
            return []

        # An uncited answer still came from the context, so attribute it to the
        # top-ranked chunk rather than dropping it and scoring a false zero.
        if not cited:
            cited = [0]

        quotes = {} if self.citation_quote == QUOTE_OFF else parse_quotes(raw, len(used))
        answers, matched = [], 0
        for i in cited:
            span = locate_quote(quotes[i], used[i].chunk.text) if i in quotes else None
            if span:
                matched += 1
                self.last_verified_contexts.add(used[i].chunk.text)
            elif self.citation_quote == QUOTE_STRICT:
                # An unverifiable citation is not evidence, and dropping it is
                # the whole point of the strict arm - including when it leaves
                # the question with no attributed answer at all.
                continue
            answers.append(self._to_answer(text, used[i], question, color, span))
        # Left unconditional, the off arm would report a match rate of 0.0 for
        # every question - a reader that was never asked to quote scored as one
        # that fabricates every citation.
        if self.citation_quote != QUOTE_OFF:
            self.last_quote_stats = {"cited": len(cited), "matched": matched}

        if progress_callback:
            progress_callback(1, 1, f"Generated an answer citing {len(cited)} chunks")

        return answers
