"""Question answering engine using extractive QA models."""

import logging

import numpy as np
import torch
from device import resolve_device
from models import Answer, Chunk, ExpandedAnswerSpan, RerankedChunk
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

DEFAULT_QA_BATCH_SIZE = {"cuda": 128, "mps": 16, "cpu": 8}
MAX_SEQUENCE_TOKENS = 512
MAX_ANSWER_TOKENS = 60
CONTEXT_STRIDE = 128


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Return the softmax of ``logits``, shifted for numerical stability."""
    exponentials = np.exp(logits - np.max(logits))
    return exponentials / exponentials.sum()


def _best_span(start_logits: np.ndarray,
            end_logits: np.ndarray,
            sequence_ids: list[int | None]) -> tuple[int, int, float] | None:
    """Pick the jointly most probable answer span inside the context tokens.

    Taking ``argmax`` of the start and end logits independently can return an end
    before its start, or two tokens from unrelated spans; and sigmoid over the mean
    of two logits is not a probability, so scores were not comparable across
    candidates. This scores every valid (start, end) pair instead and returns a
    real probability.

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

    start_probs = _softmax(np.where(is_context, start_logits, -np.inf))
    end_probs = _softmax(np.where(is_context, end_logits, -np.inf))

    # Rows are starts, columns are ends: keep end >= start and bound the span length.
    scores = np.triu(np.outer(start_probs, end_probs))
    scores = np.tril(scores, MAX_ANSWER_TOKENS)

    start_token, end_token = np.unravel_index(scores.argmax(), scores.shape)
    return int(start_token), int(end_token), float(scores[start_token, end_token])


class QAEngine:
    """Handles extractive question answering over text passages."""
    
    def __init__(self, 
                model_name: str = "deepset/deberta-v3-large-squad2", 
                device: str | None = None,
                enable_question_expansion: bool = True,
                batch_size: int | None = None):
        """Initialize the QA engine.

        Args:
            model_name: Name of the HuggingFace QA model.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detect if None.
            enable_question_expansion: Generate question variations to improve retrieval.
            batch_size: Sequences per forward pass. Device default if None.
        """
        self.model_name = model_name
        self.device = resolve_device(device)
        self.enable_question_expansion = enable_question_expansion
        self.model = None
        self.tokenizer = None
        self.paraphraser = None
        # 512-token sequences through a large QA model: 128 only fits a dedicated GPU.
        # MPS shares its memory with the OS, and CPU OOMs outright (exit 137).
        self.batch_size = batch_size or DEFAULT_QA_BATCH_SIZE[self.device]
        
        self._load_model_direct()
        if enable_question_expansion:
            self._load_paraphraser()

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
    
    def _load_paraphraser(self):
        """Load paraphrasing model for question expansion."""
        try:
            self.paraphraser = pipeline(
                "text2text-generation",
                model="humarin/chatgpt_paraphraser_on_T5_base",
                device=self.device
            )
            logger.info(f"Question paraphraser loaded on {self.device}")
        except Exception as e:  # noqa: BLE001 - optional model: continue without the paraphraser
            logger.warning(f"Could not load paraphraser: {e}")
            self.paraphraser = None
    
    def expand_question(self, question: str, num_variations: int = 2) -> list[str]:
        """Generate question variations to improve retrieval coverage.
        
        Args:
            question: The original question.
            num_variations: Number of paraphrases to generate.
            
        Returns:
            List of question variations including the original.
        """
        variations = [question]  # Always include original
        
        if not self.paraphraser or not self.enable_question_expansion:
            return variations
        
        try:
            # Generate multiple candidates with higher diversity
            all_candidates = []
            
            # Generate more candidates than needed to filter best ones
            result = self.paraphraser(
                f"paraphrase: {question}",  # Add explicit instruction
                max_new_tokens=64,  # Use max_new_tokens instead of max_length
                num_return_sequences=num_variations * 3,  # Generate 3x more for filtering
                num_beams=num_variations * 3,
                temperature=1.5,  # Higher temperature for more diversity
                top_k=50,
                top_p=0.95,
                do_sample=True,
                early_stopping=True
            )
            
            # Collect and filter paraphrases
            for res in result:
                paraphrase = res['generated_text'].strip()
                
                # Remove common prefixes that might be added
                paraphrase = paraphrase.replace('paraphrase:', '').strip()
                
                # Check if meaningfully different from original
                if paraphrase and paraphrase != question:
                    normalized = paraphrase.lower().strip()
                    if normalized != question.lower():
                        # Calculate similarity to original
                        orig_words = set(question.lower().split())
                        new_words = set(paraphrase.lower().split())
                        
                        intersection = len(orig_words & new_words)
                        union = len(orig_words | new_words)
                        similarity = intersection / union if union > 0 else 1.0
                        
                        # Keep if diverse enough from original (< 75% similar)
                        if similarity < 0.75:
                            all_candidates.append((similarity, paraphrase))
            
            # Sort by diversity from original (lower similarity = more diverse)
            all_candidates.sort(key=lambda x: x[0])
            
            # Select candidates while checking diversity among variations themselves
            for similarity, candidate in all_candidates:
                if len(variations) - 1 >= num_variations:
                    break
                
                # Check candidate is diverse from already-selected variations
                is_diverse = True
                candidate_words = set(candidate.lower().split())
                
                for existing_var in variations:
                    existing_words = set(existing_var.lower().split())
                    var_intersection = len(candidate_words & existing_words)
                    var_union = len(candidate_words | existing_words)
                    var_similarity = var_intersection / var_union if var_union > 0 else 1.0
                    
                    # Require < 70% similarity to all existing variations
                    if var_similarity >= 0.70:
                        is_diverse = False
                        break
                
                if is_diverse:
                    variations.append(candidate)
            
            logger.info(f"Generated {len(variations)-1} question variations from {len(all_candidates)} candidates")
            for i, var in enumerate(variations):
                logger.debug(f"  Q{i}: {var}")
                
        except Exception as e:  # noqa: BLE001 - expansion optional: keep the original question
            logger.warning(f"Question expansion failed: {e}")
        
        return variations
    
    def _expand_to_sentences(self, 
                            chunk: Chunk, 
                            start_char: int, 
                            end_char: int) -> ExpandedAnswerSpan:
        """Expand answer span to include complete sentences and return their coordinates.
        
        Args:
            chunk: The Chunk object containing the answer.
            start_char: Start position of answer in context.
            end_char: End position of answer in context.
            
        Returns:
            (expanded_text, new_start, new_end, sentence_coords) tuple.
        """
        if not chunk.sentences:
            return ExpandedAnswerSpan(
                text=chunk.text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
                sentence_coords=[],
            )

        # Map character positions to sentences
        start_sentence_idx = -1
        end_sentence_idx = -1
        
        current_pos = 0
        for i, (sent_text, _) in enumerate(chunk.sentences):
            sent_len = len(sent_text)
            sent_end = current_pos + sent_len
            
            # Check if answer start falls in this sentence
            if start_sentence_idx == -1 and start_char < sent_end + 1:
                start_sentence_idx = i
            
            # Check if answer end falls in this sentence
            if end_char <= sent_end + 1:
                end_sentence_idx = i
                break
                
            current_pos += sent_len + 1  # +1 for space between sentences
            
        if start_sentence_idx == -1:
            start_sentence_idx = 0
        if end_sentence_idx == -1:
            end_sentence_idx = len(chunk.sentences) - 1
            
        # Extract full text of all involved sentences
        involved_sentences = chunk.sentences[start_sentence_idx : end_sentence_idx + 1]
        
        expanded_text = " ".join(s[0] for s in involved_sentences)
        sentence_coords = [s[1] for s in involved_sentences if s[1]]
        
        # Calculate new start/end relative to the whole chunk text
        new_start = 0
        for i in range(start_sentence_idx):
            new_start += len(chunk.sentences[i][0]) + 1
            
        new_end = new_start + len(expanded_text)
        
        return ExpandedAnswerSpan(
            text=expanded_text,
            start_char=new_start,
            end_char=new_end,
            sentence_coords=sentence_coords,
        )
    
    def extract_answers(self,
                    question: str,
                    candidates: list[RerankedChunk],
                    config: dict,
                    color: tuple[float, float, float] = (1, 1, 0),
                    progress_callback=None,
                    question_variations: list[str] | None = None) -> list[Answer]:
        """Extract answers for a given question from candidate chunks.

        Args:
            question: The question to answer.
            candidates: List of RerankedChunk objects to extract answers from.
            config: Resolved question-type config (see ``question_presets.resolve``).
                Reads ``qa_score_threshold``, ``min_answer_words`` and
                ``section_diversity_enabled``.
            color: Highlight color for the answer (R, G, B).
            progress_callback: Optional callback for progress updates (batch_idx, total_batches, message).
            question_variations: Optional list of question variations to use instead of generating them.

        Returns:
            List of Answer objects extracted from the candidates.
        """
        if self.model is None:
            raise RuntimeError("QA model not loaded.")
        
        if not candidates:
            logger.info("No candidates provided for QA extraction.")
            return []
        
        # --- 0. INITIALIZE STATS & CONFIG ---
        questions_to_use = question_variations if question_variations else [question]

        logger.info(f"QA Configuration -> Threshold: {config['qa_score_threshold']:.3f} | Min Words: {config['min_answer_words']}")
        
        # Initialize tracking dictionaries
        filter_stats = {
            'total_inputs': 0,      # Total (candidates * variations)
            'successful_raw': 0,    # Raw answers extracted from model
            'too_few_words': 0,     # Filtered: Answer too short
            'below_threshold': 0,   # Filtered: Confidence too low
            'duplicates': 0,        # Filtered: Duplicate content
            'final_count': 0        # Final answers returned
        }

        # --- 1. PREPARE BATCH DATA ---
        batch_questions = []
        batch_contexts = []
        metadata_map = [] 
        
        for i, candidate in enumerate(candidates):
            chunk = candidate.chunk
            combined_text = chunk.text
            
            # Prepare variations
            for q_var in questions_to_use:
                batch_questions.append(q_var)
                batch_contexts.append(combined_text)
                metadata_map.append({
                    'chunk': chunk,
                    'retrieval_score': candidate.retrieval_score,
                    'rerank_score': candidate.rerank_score,
                    'q_var': q_var,
                    'context_text': combined_text,
                    'candidate_idx': i
                })
        
        filter_stats['total_inputs'] = len(batch_questions)
        logger.info(f"Preparing inference for {len(candidates)} candidates x {len(questions_to_use)} variations = {len(batch_questions)} total sequences.")

        # --- 2. BATCH TOKENIZATION & INFERENCE ---
        BATCH_SIZE = self.batch_size
        all_answers_raw = []
        num_batches = (len(batch_questions) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for b in range(num_batches):
            if progress_callback:
                progress_callback(b, num_batches, f"Running Inference Batch {b+1}/{num_batches}")
                
            start_idx = b * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            
            b_q = batch_questions[start_idx:end_idx]
            b_c = batch_contexts[start_idx:end_idx]
            
            try:
                # A context longer than the window used to be silently cut, losing every
                # answer past the 512th token. Overflowing windows keep the tail, and
                # the stride keeps a span straddling a boundary intact in one of them.
                encoded = self.tokenizer(
                    b_q, b_c,
                    add_special_tokens=True, return_tensors="pt", padding=True,
                    truncation="only_second", max_length=MAX_SEQUENCE_TOKENS,
                    stride=CONTEXT_STRIDE, return_overflowing_tokens=True,
                    return_offsets_mapping=True
                )

                sample_mapping = encoded.pop("overflow_to_sample_mapping").tolist()
                offset_mapping = encoded.pop("offset_mapping").numpy()
                sequence_ids = [encoded.sequence_ids(k) for k in range(len(sample_mapping))]
                inputs = encoded.to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                start_logits = outputs.start_logits.float().cpu().numpy()
                end_logits = outputs.end_logits.float().cpu().numpy()

                # Extract spans from logits
                for k, (start_logit, end_logit, offsets) in enumerate(zip(start_logits, end_logits, offset_mapping)):
                    global_idx = start_idx + sample_mapping[k]

                    span = _best_span(start_logit, end_logit, sequence_ids[k])
                    if span is None:
                        continue

                    start_token, end_token, score_norm = span
                    start_char_idx = offsets[start_token][0]
                    end_char_idx = offsets[end_token][1]

                    if start_char_idx >= end_char_idx:
                        continue

                    meta = metadata_map[global_idx]
                    ans_text = meta['context_text'][start_char_idx:end_char_idx]

                    all_answers_raw.append({
                        'text': ans_text,
                        'start': start_char_idx,
                        'end': end_char_idx,
                        'score': score_norm,
                        'meta': meta
                    })
                    filter_stats['successful_raw'] += 1

            except Exception as e:  # noqa: BLE001 - per-batch: skip it and keep the others
                logger.error(f"Batch {b} failed: {e}")
                continue

        # --- 3. RECONSTRUCT & FILTER (WORD COUNT) ---
        grouped_results = {}
        for res in all_answers_raw:
            c_idx = res['meta']['candidate_idx']
            if c_idx not in grouped_results: grouped_results[c_idx] = []
            grouped_results[c_idx].append(res)
            
        answers = []
        for c_idx, group in grouped_results.items():
            # Pick best variation score
            best_res = max(group, key=lambda x: x['score'])
            
            # Filter: Word Count
            if len(best_res['text'].split()) < config['min_answer_words']:
                filter_stats['too_few_words'] += 1
                logger.debug(f"Rejected (Too Short): {best_res['text'][:30]}...")
                continue
                
            meta = best_res['meta']
            
            # Recalculate offsets relative to original chunk
            chunk = meta['chunk']
            raw_start, raw_end = best_res['start'], best_res['end']
            
            target_chunk = chunk
            local_start, local_end = raw_start, raw_end

            # Sentence expansion
            expanded = self._expand_to_sentences(
                target_chunk, local_start, local_end
            )
            
            answers.append(Answer(
                text=expanded.text,
                context=target_chunk.text,
                page_number=target_chunk.page_number,
                title=target_chunk.title,
                section=target_chunk.section,
                start_char=expanded.start_char,
                end_char=expanded.end_char,
                score=best_res['score'],
                query=meta['q_var'],
                color=color,
                sentence_coords=expanded.sentence_coords,
                retrieval_score=meta['retrieval_score'],
                rerank_score=meta['rerank_score'],
                pdf_hash=target_chunk.pdf_hash
            ))

        # Sort by score descending
        answers.sort(key=lambda x: x.score, reverse=True)
        
        # --- 4. DEDUPLICATION & THRESHOLDING ---
        unique_answers = []
        
        # Helper for stats counting within deduplication
        def is_valid_score(ans):
            if ans.score < config['qa_score_threshold']:
                filter_stats['below_threshold'] += 1
                return False
            return True

        if config.get('section_diversity_enabled', False):
            # ... [Section Diversity Logic] ...
            intro_sections = ['abstract', 'introduction', 'intro']
            method_sections = ['methodology', 'methods', 'approach', 'algorithm', 'implementation']
            
            intro_answers, method_answers, other_answers = [], [], []
            
            for ans in answers:
                if not is_valid_score(ans): continue
                    
                sec = (ans.section or '').lower()
                if any(s in sec for s in intro_sections): intro_answers.append(ans)
                elif any(s in sec for s in method_sections): method_answers.append(ans)
                else: other_answers.append(ans)
            
            def dedup_list(alist):
                seen, uniq = set(), []
                for a in alist:
                    sig = (a.title, " ".join(a.text.lower().split()))
                    if sig not in seen:
                        seen.add(sig)
                        uniq.append(a)
                    else:
                        filter_stats['duplicates'] += 1
                return uniq

            intro_u = dedup_list(intro_answers)
            method_u = dedup_list(method_answers)
            other_u = dedup_list(other_answers)
            
            # Interleave
            max_len = max(len(intro_u), len(method_u))
            for i in range(max_len):
                if i < len(intro_u): unique_answers.append(intro_u[i])
                if i < len(method_u): unique_answers.append(method_u[i])
            unique_answers.extend(other_u)
            
        else:
            # Standard Deduplication
            seen_answers = set()
            for ans in answers:
                if not is_valid_score(ans): continue
                
                sig = (ans.title, " ".join(ans.text.lower().split()))
                if sig not in seen_answers:
                    seen_answers.add(sig)
                    unique_answers.append(ans)
                else:
                    filter_stats['duplicates'] += 1

        filter_stats['final_count'] = len(unique_answers)

        # --- 5. FINAL SUMMARY LOG ---
        logger.info("-" * 60)
        logger.info("QA EXTRACTION SUMMARY")
        logger.info(f"Input: {len(candidates)} candidates -> {filter_stats['total_inputs']} sequences (expanded)")
        logger.info(f"Raw Outputs: {filter_stats['successful_raw']} spans found")
        logger.info("Filtering:")
        logger.info(f"  - Too Short (<{config['min_answer_words']} words): {filter_stats['too_few_words']}")
        logger.info(f"  - Low Confidence (<{config['qa_score_threshold']:.2f}): {filter_stats['below_threshold']}")
        logger.info(f"  - Duplicates: {filter_stats['duplicates']}")
        logger.info(f"Final Results: {filter_stats['final_count']} unique answers")
        logger.info("-" * 60)
        
        return unique_answers
