import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from zotero_rag.answering.highlighter import answers_by_pdf, highlight_pdf
from zotero_rag.models import YELLOW, Answer
from zotero_rag.pipeline import ZoteroRAG

logger = logging.getLogger(__name__)

# No model here: the thresholds in settings.py are calibrated on the
# pipeline's own models, and the dense one was chosen by the same campaign.
DEFAULTS = {
    'output_base_dir': './output',
    'grobid_url': 'http://localhost:8070',
    'grobid_timeout': 180,
    'create_highlighted_pdfs': True,
    'rebuild_index': False,
}


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = {**DEFAULTS, **yaml.safe_load(f)}
    config['defaults'] = {'highlight_color': list(YELLOW), **config.get('defaults', {})}
    return config


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    """The answer as the results file stores it: everything but the highlighting internals."""
    record = asdict(answer)
    del record['sentence_coords'], record['pdf_hash']
    return record


def run_from_config(config_path: str) -> dict[str, list[Answer]]:
    """Run ZoteroRAG pipeline from a YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Dictionary mapping questions to their answers.
    """
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    logger.info("Initializing RAG system")
    rag = ZoteroRAG(
        grobid_url=config['grobid_url'],
        grobid_timeout=config['grobid_timeout'],
        model_device=config.get('model_device'),
        encode_batch_size=config.get('encode_batch_size'),
        rerank_batch_size=config.get('rerank_batch_size'),
        qa_batch_size=config.get('qa_batch_size'),
        output_base_dir=config['output_base_dir'],
    )
    index_library(rag, config)

    if not config.get('questions'):
        logger.warning("No questions specified in config file")
        return {}
    all_results = answer_questions(rag, config['questions'], config['defaults'])

    # Both the highlighted PDFs and the results file land here, so it is resolved
    # before either block runs - not inside the optional one.
    output_dir = os.path.join(config['output_base_dir'], "highlighted_results")
    os.makedirs(output_dir, exist_ok=True)
    if config['create_highlighted_pdfs']:
        write_highlighted_pdfs(all_results, output_dir)
    if config.get('results_file'):
        write_results(all_results, os.path.join(output_dir, config['results_file']))
    return all_results


def index_library(rag: ZoteroRAG, config: dict[str, Any]) -> None:
    """Ingest the configured source and index what is new.

    Indexing is incremental: PDFs already in Qdrant are skipped on upsert, so a
    run only pays for what is new unless rebuild_index asks for a clean slate.
    """
    if config['rebuild_index']:
        logger.info("rebuild_index is set: clearing the existing index")
        rag.clear_index()

    if config.get('source_type', 'zotero') == 'folder':
        logger.info("Ingesting PDFs from folder %s", config['folder_path'])
        ingest_result = rag.ingest_pdfs_from_folder(config['folder_path'])
    else:
        logger.info("Ingesting PDFs from Zotero")
        ingest_result = rag.ingest_pdfs_from_zotero(
            zotero_collection=config.get('zotero_collection'),
            zotero_data_dir=config.get('zotero_data_dir'),
        )

    for failure in ingest_result.failed_pdfs:
        logger.warning("Ingest failed for %s: %s",
                       failure.get('title', 'Unknown PDF'), failure.get('error', 'Unknown error'))

    upsert_result = rag.upsert_pdfs(ingest_result.ingested_pdfs)
    logger.info("Indexed %d chunks from %d PDFs",
                upsert_result.indexed_chunks, upsert_result.processed_pdfs)


def answer_questions(rag: ZoteroRAG, questions: list[dict],
                     defaults: dict[str, Any]) -> dict[str, list[Answer]]:
    """Each configured question's answers; only the highlight colour is per question."""
    all_results = {}
    for i, question_config in enumerate(questions):
        question = question_config.get('question')
        if not question:
            logger.warning(f"Question {i+1} missing 'question' field, skipping")
            continue

        logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
        highlight_color = question_config.get('highlight_color', defaults['highlight_color'])
        answers = rag.answer_question(
            question=question,
            highlight_color=tuple(highlight_color) if highlight_color else highlight_color,
        )
        all_results[question] = answers

        logger.info(f"Found {len(answers)} answers for question: {question}")
        for j, answer in enumerate(answers[:3]):
            logger.info(f"  Answer {j+1}: {answer.text[:100]}... (score: {answer.score:.3f}, rerank: {answer.rerank_score:.3f})")
    return all_results


def write_highlighted_pdfs(all_results: dict[str, list[Answer]], output_dir: str) -> None:
    """One highlighted copy per source PDF, carrying every answer drawn from it."""
    logger.info("Creating highlighted PDFs")
    every_answer = (answer for answers in all_results.values() for answer in answers)
    for pdf_path, pdf_answers in answers_by_pdf(every_answer).items():
        pdf_filename = Path(pdf_path).stem
        output_path = os.path.join(output_dir, f"{pdf_filename}_highlighted.pdf")
        logger.info(f"Highlighting {pdf_filename} with {len(pdf_answers)} answers")
        if result_path := highlight_pdf(pdf_answers, output_path):
            logger.info(f"  Saved to: {result_path}")
        else:
            logger.warning("  Failed to create highlighted PDF")


def write_results(all_results: dict[str, list[Answer]], output_path: str) -> None:
    logger.info(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump({question: [answer_to_dict(a) for a in answers]
                   for question, answers in all_results.items()}, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point for CLI usage: logs go to the terminal, as a command's do."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if len(sys.argv) < 2:
        print("Usage: python -m zotero_rag.app.run_from_config <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        logger.info(f"Running ZoteroRAG from config: {config_path}")
        results = run_from_config(config_path)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        total_answers = sum(len(answers) for answers in results.values())
        print(f"\nProcessed {len(results)} questions")
        print(f"Found {total_answers} total answers")
        
        for question, answers in results.items():
            print(f"\n{question}")
            print(f"  {len(answers)} answers")
        
        print("\n" + "="*80)
        
    except Exception:
        logger.exception("Error running from config")
        sys.exit(1)


if __name__ == "__main__":
    main()
