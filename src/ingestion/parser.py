"""EU AI Act document structure parser.

Converts flat page text into labelled structural units matching the EU legal
document format. This is the most domain-specific file in the project — regex
patterns here are tuned to the EU AI Act's exact formatting conventions.

Key structural units detected:
  - Articles (Article 1, Article 2(1), ...)
  - Recitals  ((1), (2), ... in the preamble)
  - Annexes   (ANNEX I, ANNEX II, ...)
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from src.ingestion.extractor import RawPage

# ---------------------------------------------------------------------------
# Section type taxonomy
# ---------------------------------------------------------------------------

# Keywords used to classify a structural unit's predominant obligation type.
# Classification is best-effort from title / first sentence; human review of
# the golden dataset should catch mis-classifications.
_OBLIGATION_KEYWORDS = re.compile(
    r"\b(shall|must|required|obligation|comply)\b", re.IGNORECASE
)
_DEFINITION_KEYWORDS = re.compile(
    r"\b(means|definition|defined as|for the purposes of)\b", re.IGNORECASE
)
_PROHIBITION_KEYWORDS = re.compile(
    r"\b(prohibited|forbidden|shall not|must not|ban)\b", re.IGNORECASE
)
_PROCEDURAL_KEYWORDS = re.compile(
    r"\b(procedure|process|step|assessment|conformity|notify|register)\b",
    re.IGNORECASE,
)


class SectionType(str, Enum):
    OBLIGATION = "obligation"
    DEFINITION = "definition"
    PROHIBITION = "prohibition"
    PROCEDURAL = "procedural"
    RECITAL = "recital"
    ANNEX = "annex"
    GENERAL = "general"  # fallback when none of the above match


# ---------------------------------------------------------------------------
# Structural unit dataclass
# ---------------------------------------------------------------------------


@dataclass
class StructuralUnit:
    article_number: str | None  # e.g. "Article 17", "Annex III", "Recital 42"
    section_type: SectionType
    title: str  # section heading, empty string if not found
    text: str  # full text of the unit
    chunk_index: int  # position within the parent document
    parent_document: str  # source filename
    source_url: str  # official publication URL or OJ reference
    publication_date: str  # version/publication date
    cross_references: list[str] = field(default_factory=list)  # mentioned articles


# ---------------------------------------------------------------------------
# Regex patterns for EU AI Act structure
# ---------------------------------------------------------------------------

# Matches "Article 6", "Article 6(1)", "Article 17" (standalone header line)
_ARTICLE_HEADER = re.compile(r"^Article\s+(\d+)\b", re.IGNORECASE)

# Matches recital headers like "(42)" at the start of a line (preamble numbering)
_RECITAL_HEADER = re.compile(r"^\((\d+)\)\s")

# Matches annex headers like "ANNEX I", "ANNEX XIII"
_ANNEX_HEADER = re.compile(r"^ANNEX\s+([IVX]+)\b", re.IGNORECASE)

# Cross-reference extraction — finds mentions like "Article 17(1)", "Annex III"
_CROSS_REF = re.compile(
    r"\b(Article\s+\d+(?:\(\d+\))?|Annex\s+[IVX]+|Recital\s+\d+)\b", re.IGNORECASE
)

# Article title line: typically ALL CAPS or Title Case on line after "Article N"
_TITLE_LINE = re.compile(r"^[A-Z][A-Za-z ,\-–()]{5,80}$")


# ---------------------------------------------------------------------------
# Validation: well-known articles that must be present after parsing
# ---------------------------------------------------------------------------
_REQUIRED_ARTICLES = {"6", "9", "13", "17", "50"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_document(
    pages: list[RawPage],
    source_url: str = "",
    publication_date: str = "",
) -> list[StructuralUnit]:
    """Parse a list of raw pages into labelled structural units.

    Concatenates all pages into a single text stream, then splits on
    detected structural boundaries (Article / Recital / Annex headers).
    """
    full_text = "\n".join(p.text for p in pages)
    parent_document = pages[0].source_file if pages else ""

    units = _split_into_units(full_text, parent_document, source_url, publication_date)
    _validate_units(units)
    return units


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _split_into_units(
    text: str,
    parent_document: str,
    source_url: str,
    publication_date: str,
) -> list[StructuralUnit]:
    """Walk through the document line by line and emit a new StructuralUnit
    whenever an Article / Recital / Annex header is detected."""
    lines = text.split("\n")
    units: list[StructuralUnit] = []

    current_article: str | None = None
    current_section_type: SectionType = SectionType.GENERAL
    current_title: str = ""
    current_lines: list[str] = []
    chunk_index = 0
    expecting_title = False  # True immediately after an Article header line

    def _flush():
        nonlocal chunk_index
        if not current_lines:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            return
        units.append(
            StructuralUnit(
                article_number=current_article,
                section_type=current_section_type,
                title=current_title,
                text=body,
                chunk_index=chunk_index,
                parent_document=parent_document,
                source_url=source_url,
                publication_date=publication_date,
                cross_references=_extract_cross_references(body),
            )
        )
        chunk_index += 1

    for line in lines:
        stripped = line.strip()

        # --- Detect new structural boundary ---
        article_match = _ARTICLE_HEADER.match(stripped)
        annex_match = _ANNEX_HEADER.match(stripped)
        recital_match = _RECITAL_HEADER.match(stripped)

        if article_match:
            _flush()
            current_lines = [stripped]
            current_article = f"Article {article_match.group(1)}"
            current_section_type = SectionType.GENERAL  # refined after title found
            current_title = ""
            expecting_title = True

        elif annex_match:
            _flush()
            current_lines = [stripped]
            current_article = f"Annex {annex_match.group(1).upper()}"
            current_section_type = SectionType.ANNEX
            current_title = stripped
            expecting_title = False

        elif recital_match:
            _flush()
            current_lines = [stripped]
            current_article = f"Recital {recital_match.group(1)}"
            current_section_type = SectionType.RECITAL
            current_title = ""
            expecting_title = False

        else:
            # Try to pick up the title line (line immediately after Article header)
            if expecting_title and stripped and _TITLE_LINE.match(stripped):
                current_title = stripped
                current_section_type = _classify_section(stripped + " " + stripped)
                expecting_title = False

            current_lines.append(stripped)

    _flush()
    return units


def _classify_section(text: str) -> SectionType:
    """Heuristic classification of a structural unit based on keyword signals."""
    if _PROHIBITION_KEYWORDS.search(text):
        return SectionType.PROHIBITION
    if _DEFINITION_KEYWORDS.search(text):
        return SectionType.DEFINITION
    if _OBLIGATION_KEYWORDS.search(text):
        return SectionType.OBLIGATION
    if _PROCEDURAL_KEYWORDS.search(text):
        return SectionType.PROCEDURAL
    return SectionType.GENERAL


def _extract_cross_references(text: str) -> list[str]:
    """Return deduplicated list of article/annex/recital references found in text."""
    return list(dict.fromkeys(m.group(0) for m in _CROSS_REF.finditer(text)))


def _validate_units(units: list[StructuralUnit]) -> None:
    """Assert that well-known articles are present — catches parser regressions."""
    found_articles = {
        u.article_number.replace("Article ", "").strip()
        for u in units
        if u.article_number and u.article_number.startswith("Article")
    }
    missing = _REQUIRED_ARTICLES - found_articles
    if missing:
        import logging

        logging.warning(
            f"Parser validation: expected articles not found: "
            f"{sorted(missing)}. Corpus may be incomplete or parsing failed."
        )
