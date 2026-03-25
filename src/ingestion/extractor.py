"""PDF text extraction layer.

Uses PyMuPDF (fitz) as primary extractor with pdfplumber as fallback
when PyMuPDF produces suspiciously short pages (likely image-only or
complex table layouts).
"""

from dataclasses import dataclass
from pathlib import Path

MIN_PAGE_CHARS = 50  # pages below this are flagged as potentially image-only


@dataclass
class RawPage:
    page_number: int  # 1-indexed
    text: str
    source_file: str


def extract_pdf(path: Path) -> list[RawPage]:
    """Extract text from a PDF, returning one RawPage per page.

    Tries PyMuPDF first; falls back to pdfplumber if extraction yields
    mostly empty pages (common with complex table-heavy annexes).
    """
    path = Path(path)
    pages = _extract_with_pymupdf(path)

    # Fall back if more than 20% of pages look empty
    nonempty = [p for p in pages if len(p.text.strip()) >= MIN_PAGE_CHARS]
    if len(pages) > 0 and len(nonempty) / len(pages) < 0.8:
        import logging

        logging.warning(
            f"PyMuPDF produced {len(pages) - len(nonempty)} near-empty pages in "
            f"{path.name}; falling back to pdfplumber."
        )
        pages = _extract_with_pdfplumber(path)

    assert len(pages) > 0, f"No pages extracted from {path}"

    # Log extraction stats
    total_chars = sum(len(p.text) for p in pages)
    short_pages = [p for p in pages if len(p.text.strip()) < MIN_PAGE_CHARS]
    if short_pages:
        import logging

        logging.warning(
            f"{path.name}: {len(short_pages)} pages have <{MIN_PAGE_CHARS} chars "
            f"(possibly image-only): pages {[p.page_number for p in short_pages]}"
        )

    import logging

    logging.info(
        f"Extracted {len(pages)} pages / {total_chars:,} chars from {path.name}"
    )
    return pages


def _extract_with_pymupdf(path: Path) -> list[RawPage]:
    import fitz  # pymupdf

    pages: list[RawPage] = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(RawPage(page_number=i + 1, text=text, source_file=path.name))
    return pages


def _extract_with_pdfplumber(path: Path) -> list[RawPage]:
    import pdfplumber

    pages: list[RawPage] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(RawPage(page_number=i + 1, text=text, source_file=path.name))
    return pages
