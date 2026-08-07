import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from highlighter import highlight_pdf
from models import Answer
from pipeline import ZoteroRAG

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults
    config.setdefault('defaults', {})
    config['defaults'].setdefault('highlight_color', [1, 1, 0])
    
    config.setdefault('output_base_dir', './output')
    # No model here: the thresholds in settings.py are calibrated on the
    # pipeline's own models, and the dense one was chosen by the same campaign.
    config.setdefault('grobid_url', 'http://localhost:8070')
    config.setdefault('grobid_timeout', 180)
    config.setdefault('create_highlighted_pdfs', True)
    config.setdefault('rebuild_index', False)
    
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
    
    # Initialize ZoteroRAG
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

    # Indexing is incremental: PDFs already in Qdrant are skipped on upsert, so a
    # run only pays for what is new unless rebuild_index asks for a clean slate.
    if config['rebuild_index']:
        logger.info("rebuild_index is set: clearing the existing index")
        rag.clear_index()

    source_type = config.get('source_type', 'zotero')
    if source_type == 'folder':
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

    # Process questions
    questions = config.get('questions', [])
    if not questions:
        logger.warning("No questions specified in config file")
        return {}
    
    defaults = config['defaults']
    all_results = {}
    
    for i, question_config in enumerate(questions):
        question = question_config.get('question')
        if not question:
            logger.warning(f"Question {i+1} missing 'question' field, skipping")
            continue
        
        logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
        
        # Only the highlight colour is per question: the pipeline's settings are fixed
        highlight_color = question_config.get('highlight_color', defaults['highlight_color'])
        
        # Convert highlight_color to tuple
        if highlight_color:
            highlight_color = tuple(highlight_color)
        
        # Answer the question
        answers = rag.answer_question(
            question=question,
            highlight_color=highlight_color,
        )
        
        all_results[question] = answers
        logger.info(f"Found {len(answers)} answers for question: {question}")
        
        # Print top answers
        for j, answer in enumerate(answers[:3]):
            logger.info(f"  Answer {j+1}: {answer.text[:100]}... (score: {answer.score:.3f}, rerank: {answer.rerank_score:.3f})")
    
    # Both the highlighted PDFs and the results file land here, so it is resolved
    # before either block runs - not inside the optional one.
    output_dir = os.path.join(config['output_base_dir'], "highlighted_results")
    os.makedirs(output_dir, exist_ok=True)

    # Create highlighted PDFs if requested
    if config['create_highlighted_pdfs']:
        logger.info("Creating highlighted PDFs")
        
        answers_by_pdf = defaultdict(list)
        for answers in all_results.values():
            for answer in answers:
                answers_by_pdf[answer.pdf_path].append(answer)

        for pdf_path, pdf_answers in answers_by_pdf.items():
            pdf_filename = Path(pdf_path).stem
            output_path = os.path.join(output_dir, f"{pdf_filename}_highlighted.pdf")

            logger.info(f"Highlighting {pdf_filename} with {len(pdf_answers)} answers")
            result_path = highlight_pdf(pdf_answers, output_path)
            
            if result_path:
                logger.info(f"  Saved to: {result_path}")
            else:
                logger.warning("  Failed to create highlighted PDF")
    
    # Save results to JSON if requested
    results_file = config.get('results_file')
    if results_file:
        logger.info(f"Saving results to {results_file}")
        
        json_results = {}
        for question, answers in all_results.items():
            json_results[question] = [answer_to_dict(a) for a in answers]
        
        output_path = os.path.join(output_dir, results_file)
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    return all_results


def main():
    """Main entry point for CLI usage: logs go to the terminal, as a command's do."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if len(sys.argv) < 2:
        print("Usage: python -m zotero_rag.run_from_config <config.yaml>")
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
