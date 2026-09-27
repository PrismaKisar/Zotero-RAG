"""Question answering engine using extractive QA models."""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from device import resolve_device
from models import Answer, RerankedChunk
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_QA_MODEL = "deepset/deberta-v3-large-squad2"
DEFAULT_QA_BATCH_SIZE = {"cuda": 128, "mps": 16, "cpu": 8}
MAX_SEQUENCE_TOKENS = 512
MAX_ANSWER_TOKENS = 60
CONTEXT_STRIDE = 128

# Windows per forward pass. batch_size bounds the (question, context) pairs given
# to the tokenizer, not the sequences that reach the model: with overflowing
# windows enabled a long context expands into several, so one "batch of 16" can
# arrive at the model as many times that. On a 16GB machine the QASA sweep was
# killed outright on the configs that expand the most - long chunks, under the
# since-removed question paraphrasing - and the number in the log ("N total
# sequences") counted the pairs, so nothing in the output said how much was
# actually being allocated.
# ponytail: fixed cap, not a memory-aware one. Slicing an already-padded batch
# leaves each window's logits bit-identical, so this bounds peak memory without
# changing a single score; raise it on a machine with room.
MAX_WINDOWS_PER_FORWARD = 16


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return the softmax of ``logits``, shifted for numerical stability."""
    exponentials = np.exp(logits - np.max(logits))
    return exponentials / exponentials.sum()


def best_span(start_logits: np.ndarray,
            end_logits: np.ndarray,
            sequence_ids: list[int | None]) -> tuple[int, int, float] | None:
    """Pick the jointly most probable answer span inside the context tokens.

    Taking ``argmax`` of the start and end logits independently can return an end
    before its start, or two tokens from unrelated spans; and sigmoid over the mean
    of two logits is not a probability, so scores were not comparable across
    candidates. This scores every valid (start, end) pair instead and returns a
    real probability.

    The probability keeps the model's "no answer" option in its normalisation: a
    SQuAD 2.0 model puts that mass on the leading [CLS] token. Renormalising over
    the context alone gave every chunk a confident-looking span however unrelated
    it was, and threw away the one signal that separates them: on the hybrid
    top-30 the median no-answer mass is 0.81-0.95 on non-gold chunks and under
    0.01 on gold ones, and keeping it lifts the gold/non-gold AUC of the score
    from 0.70 to 0.73 on QASPER and 0.68 to 0.75 on QASA
    (benchmark/qa_score_distribution.py). The span itself cannot change: the
    extra term only rescales the context probabilities.

    Args:
        start_logits: Start logits for one sequence.
        end_logits: End logits for the same sequence.
        sequence_ids: Per-token sequence id; ``1`` marks context tokens.

    Returns:
        ``(start_token, end_token, probability)``, or None if no context token exists.
    """
    is_context = np.array([sid == 1 for sid in sequence_ids])
    if not is_context.any():
        return None

    normalised = is_context.copy()
    if sequence_ids[0] is None:
        normalised[0] = True  # [CLS]: the no-answer option
    start_probs = softmax(np.where(normalised, start_logits, -np.inf)) * is_context
    end_probs = softmax(np.where(normalised, end_logits, -np.inf)) * is_context

    # Rows are starts, columns are ends: keep end >= start and bound the span length.
    scores = np.triu(np.outer(start_probs, end_probs))
    scores = np.tril(scores, MAX_ANSWER_TOKENS)

    start_token, end_token = np.unravel_index(scores.argmax(), scores.shape)
    return int(start_token), int(end_token), float(scores[start_token, end_token])


@dataclass
class Windows:
    """Every overflow window of a batch of contexts, with the model's logits for it.

    Row ``k`` of each field is one window; ``sample_mapping[k]`` is the index of
    the context it was cut from, and ``offsets[k]`` maps its tokens back to
    character positions in that context.
    """
    sample_mapping: list[int]
    offsets: np.ndarray
    sequence_ids: list[list[int | None]]
    start_logits: np.ndarray
    end_logits: np.ndarray


class QAEngine:
    """Handles extractive question answering over text passages."""
    
    def __init__(self, 
                model_name: str = DEFAULT_QA_MODEL,
                device: str | None = None,
                batch_size: int | None = None):
        """Initialize the QA engine.

        Args:
            model_name: Name of the HuggingFace QA model.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detect if None.
            batch_size: Sequences per forward pass. Device default if None.
        """
        self.model_name = model_name
        # The Reader seam (models.Reader): an extractive reader is its model,
        # never quotes, and has no service to lose a generation to.
        self.reader_kind = model_name
        self.last_quote_stats = {"cited": 0, "matched": 0}
        self.generation_failures = 0
        self.device = resolve_device(device)
        self.model = None
        self.tokenizer = None
        # 512-token sequences through a large QA model: 128 only fits a dedicated GPU.
        # MPS shares its memory with the OS, and CPU OOMs outright (exit 137).
        self.batch_size = batch_size or DEFAULT_QA_BATCH_SIZE[self.device]
        
        self._load_model_direct()

    def _load_model_direct(self):
        """Load model and tokenizer directly for manual batching."""
        try:
            logger.info(f"Loading QA Model: {self.model_name}...")
            
            # Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            
            # Half precision where the accelerator supports it: bfloat16 on CUDA,
            # float16 on MPS (which has no bfloat16 path). CPU stays float32.
            dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(self.device, torch.float32)
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name,
                dtype=dtype
            ).to(self.device)
            
            import sys
            # ponytail: only compile on CUDA. inductor needs a C++ compiler (g++) and
            # gives marginal gains on CPU QA — skipping avoids a hard toolchain dependency
            # and the lazy compile failure that surfaced as "0 spans" in slim containers.
            if sys.platform != 'win32' and self.device == "cuda":
                try:
                    self.model = torch.compile(self.model)
                except Exception:
                    logger.debug("torch.compile failed, using eager model", exc_info=True)

            logger.info(f"QA Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Could not load QA model {self.model_name}: {e}")
            raise
    
    def _logits_in_windows(self, inputs) -> tuple[np.ndarray, np.ndarray]:
        """Start/end logits for every window, at most MAX_WINDOWS_PER_FORWARD at a time.

        ``inputs`` is already padded to one common length, so a row slice is the
        same tensor the model would have seen in the full batch, and attention
        masks make padding inert. The concatenated result is therefore identical
        to one forward pass over everything - this bounds memory, not behaviour.
        """
        starts, ends = [], []
        for first in range(0, inputs["input_ids"].shape[0], MAX_WINDOWS_PER_FORWARD):
            window = {k: v[first:first + MAX_WINDOWS_PER_FORWARD] for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**window)
            starts.append(outputs.start_logits.float().cpu().numpy())
            ends.append(outputs.end_logits.float().cpu().numpy())
        return np.concatenate(starts), np.concatenate(ends)

    def score_windows(self, question: str, contexts: list[str]) -> Windows:
        """Ask ``question`` of every context, one window at a time.

        A context longer than the window used to be silently cut, losing every
        answer past the 512th token. Overflowing windows keep the tail, and the
        stride keeps a span straddling a boundary intact in one of them. All the
        contexts are padded together, so the caller decides the batch.
        """
        encoded = self.tokenizer(
            [question] * len(contexts), contexts,
            add_special_tokens=True, return_tensors="pt", padding=True,
            truncation="only_second", max_length=MAX_SEQUENCE_TOKENS,
            stride=CONTEXT_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True
        )

        sample_mapping = encoded.pop("overflow_to_sample_mapping").tolist()
        offsets = encoded.pop("offset_mapping").numpy()
        sequence_ids = [encoded.sequence_ids(k) for k in range(len(sample_mapping))]
        start_logits, end_logits = self._logits_in_windows(encoded.to(self.device))
        return Windows(sample_mapping, offsets, sequence_ids, start_logits, end_logits)

    def extract_answers(self,
                    question: str,
                    candidates: list[RerankedChunk],
                    config: dict,
                    color: tuple[float, float, float] = (1, 1, 0),
                    progress_callback=None) -> list[Answer]:
        """Extract answers for a given question from candidate chunks.

        Args:
            question: The question to answer.
            candidates: List of RerankedChunk objects to extract answers from.
            config: The resolved settings (see ``settings.resolve``).
                Reads ``qa_score_threshold``.
            color: Highlight color for the answer (R, G, B).
            progress_callback: Optional callback for progress updates (batch_idx, total_batches, message).

        Returns:
            One answer per candidate that yields a span, best first, above the
            threshold and without repeating a (title, text) pair.
        """
        # candidate index -> (score, start_char, end_char) of its best window
        best: dict[int, tuple[float, int, int]] = {}
        num_batches = (len(candidates) + self.batch_size - 1) // self.batch_size
        for b, first in enumerate(range(0, len(candidates), self.batch_size)):
            if progress_callback:
                progress_callback(b, num_batches, f"Running Inference Batch {b+1}/{num_batches}")
            contexts = [c.chunk.text for c in candidates[first:first + self.batch_size]]
            try:
                windows = self.score_windows(question, contexts)
            except Exception as e:  # noqa: BLE001 - per-batch: skip it and keep the others
                logger.error(f"Batch {b} failed: {e}")
                continue

            for k, offsets in enumerate(windows.offsets):
                span = best_span(windows.start_logits[k], windows.end_logits[k],
                                 windows.sequence_ids[k])
                if span is None:
                    continue
                start_token, end_token, score = span
                start_char, end_char = int(offsets[start_token][0]), int(offsets[end_token][1])
                idx = first + windows.sample_mapping[k]
                if start_char < end_char and score > best.get(idx, (-1.0,))[0]:
                    best[idx] = (score, start_char, end_char)

        answers = []
        for idx, (score, start_char, end_char) in best.items():
            candidate = candidates[idx]
            chunk = candidate.chunk
            expanded = chunk.expand_to_sentences(start_char, end_char)
            answers.append(Answer(
                text=expanded.text,
                context=chunk.text,
                page_number=chunk.page_number,
                title=chunk.title,
                section=chunk.section,
                start_char=expanded.start_char,
                end_char=expanded.end_char,
                score=score,
                query=question,
                color=color,
                sentence_coords=expanded.sentence_coords,
                retrieval_score=candidate.retrieval_score,
                rerank_score=candidate.rerank_score,
                pdf_hash=chunk.pdf_hash
            ))
        answers.sort(key=lambda x: x.score, reverse=True)

        unique_answers, seen = [], set()
        for ans in answers:
            sig = (ans.title, " ".join(ans.text.lower().split()))
            if ans.score >= config['qa_score_threshold'] and sig not in seen:
                seen.add(sig)
                unique_answers.append(ans)

        logger.info(f"QA: {len(candidates)} candidates -> {len(answers)} spans -> "
                    f"{len(unique_answers)} answers (threshold {config['qa_score_threshold']:.2f})")
        return unique_answers
