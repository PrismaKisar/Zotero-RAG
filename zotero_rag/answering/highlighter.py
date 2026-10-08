"""PDF highlighting utilities using TEI coordinates."""

import logging
import os
import tempfile

import fitz  # PyMuPDF

from zotero_rag.models import Answer

logger = logging.getLogger(__name__)


def _boxes(sentence_coords: list[str]) -> list[tuple[int, fitz.Rect]]:
    """(page index, rect) of every well-formed GROBID box "page,x,y,w,h".

    GROBID numbers pages from 1 and separates the boxes of one sentence with ";".
    """
    boxes = []
    for group in ";".join(sentence_coords).split(";"):
        try:
            page, x, y, width, height = (float(v) for v in group.split(",")[:5])
        except ValueError:
            continue
        boxes.append((int(page) - 1, fitz.Rect(x, y, x + width, y + height)))
    return boxes


def highlight_pdf(answers: list[Answer], output_path: str) -> str | None:
    """Highlight the sentences of ``answers`` (all from one PDF) into ``output_path``.

    Highlights accumulate: an existing ``output_path`` is the source, so earlier
    runs keep theirs. Each answer gets one note with its scores on every page it
    touches, at its first box there.

    Returns:
        ``output_path``, or None if there was nothing to do or highlighting failed.
    """
    if not answers:
        return None

    try:
        source_pdf = output_path if os.path.exists(output_path) else answers[0].pdf_path
        doc = fitz.open(source_pdf)
        for answer in answers:
            noted_pages = set()
            for page_number, rect in _boxes(answer.sentence_coords):
                if not 0 <= page_number < len(doc):
                    logger.warning(f"Page {page_number} referenced in coords but PDF only has {len(doc)} pages")
                    continue
                page = doc[page_number]
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=answer.color)
                highlight.update()
                if page_number not in noted_pages:
                    noted_pages.add(page_number)
                    page.add_text_annot(rect.tl, (
                        f"Q: {answer.query}\n\n"
                        f"Scores:\n"
                        f"• Retrieval: {answer.retrieval_score:.4f}\n"
                        f"• Rerank: {answer.rerank_score:.4f}\n"
                        f"• QA Confidence: {answer.score:.4f}"
                    )).update()

        # A full (non-incremental) save to a temporary file, then an atomic
        # replace: the source may be output_path itself.
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf', dir=os.path.dirname(output_path))
        os.close(temp_fd)
        try:
            doc.save(temp_path, deflate=True)
            doc.close()
            os.replace(temp_path, output_path)
        except Exception:
            os.remove(temp_path)
            raise

        logger.info(f"PDF highlighted and saved to {output_path}")
        return output_path

    except Exception:
        logger.exception("Error highlighting PDF")
        return None
