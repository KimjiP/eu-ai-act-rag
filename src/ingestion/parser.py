"""EU AI Act document structure parser.

Turns the page text of the Official Journal PDF into labelled structural units:
the preamble, each recital, each article, each definition in Article 3, and each
annex. This is the most domain-specific file in the project; the rules follow the
OJ layout of Regulation (EU) 2024/1689.

How the layout is handled:
  - Every page ends with the OJ furniture block (EN, the "OJ L" date, the page
    number and the ELI link), followed by that page's footnotes. Both are cut
    before parsing, because footnote numbers use the same "(N)" format as
    recital numbers.
  - The document has fixed regions: preamble, recitals (after "Whereas:"),
    articles (after "HAVE ADOPTED THIS REGULATION:"), the signature block
    ("Done at ...") and the annexes. Each header type is only recognised in its
    own region.
  - A header must be a whole line ("Article 7", "(12)", "ANNEX III") and must
    continue the numbering (Article 7 only after Article 6). Cross-references that
    happen to start a line in the PDF, such as "Article 13;" or a standalone
    "Article 49" inside Annex VIII, therefore stay body text.
  - Article 3 is split into one unit per definition ("Article 3(56)"), because
    definitions are looked up one at a time.

Documents without the region markers (short test fixtures) are parsed in a
lenient mode that recognises every header type everywhere, still as whole lines
and in sequence.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from src.ingestion.extractor import RawPage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section type taxonomy
# ---------------------------------------------------------------------------

# Keywords used to classify a structural unit's predominant obligation type.
# Classification is best-effort from the article title.
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
    article_number: str | None  # e.g. "Article 17", "Article 3(56)", "Annex III", "Recital 42"
    section_type: SectionType
    title: str  # section heading, empty string if not found
    text: str  # full text of the unit
    chunk_index: int  # position within the parent document
    parent_document: str  # source filename
    source_url: str  # official publication URL or OJ reference
    publication_date: str  # version/publication date
    cross_references: list[str] = field(default_factory=list)  # mentioned articles


# ---------------------------------------------------------------------------
# Layout patterns
# ---------------------------------------------------------------------------

_ARTICLE_HEADER = re.compile(r"^Article (\d+)$")
_RECITAL_HEADER = re.compile(r"^\((\d+)\)$")
_ANNEX_HEADER = re.compile(r"^ANNEX ([IVXLC]+)$")
# Article 3 definitions: "(1)" alone on a line, or "(10) ‘term’ means ..." on one line
_DEFINITION_HEADER = re.compile(r"^\((\d+)\)(?:\s+\S.*)?$")
# Chapter and section headings between articles, e.g. "CHAPTER III", "SECTION 2"
_HEADING = re.compile(r"^(CHAPTER [IVXLC]+|SECTION \d+)$")
# A title ending in a connecting word continues on the next line
_OPEN_ENDING = re.compile(r"\b(with|to|in|of|and|for|under|by|on)$")

_RECITALS_MARKER = "Whereas:"
_ENACTING_MARKER = re.compile(r"^HAVE ADOPTED THIS (REGULATION|DIRECTIVE|DECISION):$")
_SIGNATURE_MARKER = re.compile(r"^Done at .+, \d{1,2} \w+ \d{4}\.$")

_DEFINITIONS_ARTICLE = 3

# OJ page furniture, printed on every page
_ELI_LINE = re.compile(r"^ELI: http://data\.europa\.eu/eli/\S+$")
_FURNITURE_LINE = re.compile(
    r"^(EN|Official Journal|of the European Union|L series|OJ L, \d{1,2}\.\d{1,2}\.\d{4}"
    r"|\d{4}/\d+|\d{1,2}\.\d{1,2}\.\d{4}|\d+/\d+|ELI: http://data\.europa\.eu/eli/\S+)$"
)

# Cross-reference extraction — finds mentions like "Article 17(1)", "Annex III"
_CROSS_REF = re.compile(
    r"\b(Article\s+\d+(?:\(\d+\))?|Annex\s+[IVX]+|Recital\s+\d+)\b", re.IGNORECASE
)

# Header kinds recognised in each region of the document
_REGION_HEADERS = {
    "preamble": set(),
    "recitals": {"recital"},
    "articles": {"article"},
    "signature": {"annex"},
    "annexes": {"annex"},
    "any": {"recital", "article", "annex"},  # lenient mode
}

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_document(
    pages: list[RawPage],
    source_url: str = "",
    publication_date: str = "",
) -> list[StructuralUnit]:
    """Parse a list of raw pages into labelled structural units."""
    parent_document = pages[0].source_file if pages else ""
    lines = [line for page in pages for line in _strip_page_furniture(page.text)]
    strict = any(line == _RECITALS_MARKER or _ENACTING_MARKER.match(line) for line in lines)

    units = _split_into_units(lines, strict, parent_document, source_url, publication_date)
    _validate_units(units, strict)
    return units


def summarize_structure(units: list[StructuralUnit]) -> dict[str, list[int]]:
    """Numbers found per unit kind, e.g. {"recital": [1, 2, ...], "annex": [1, 2, ...]}.

    Definitions are listed separately from articles; annexes as integers.
    """
    found: dict[str, list[int]] = {"recital": [], "article": [], "definition": [], "annex": []}
    for unit in units:
        label = unit.article_number or ""
        if m := re.fullmatch(r"Recital (\d+)", label):
            found["recital"].append(int(m[1]))
        elif m := re.fullmatch(r"Article \d+\((\d+)\)", label):
            found["definition"].append(int(m[1]))
        elif m := re.fullmatch(r"Article (\d+)", label):
            found["article"].append(int(m[1]))
        elif m := re.fullmatch(r"Annex ([IVXLC]+)", label):
            found["annex"].append(_roman_to_int(m[1]))
    return {kind: sorted(numbers) for kind, numbers in found.items()}


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


@dataclass
class _Draft:
    """A unit being collected line by line."""

    label: str
    section_type: SectionType
    lines: list[str]
    title: str = ""
    # "expecting": next line is the title; "open": title may wrap onto the next line
    title_state: str = "done"


def _strip_page_furniture(page_text: str) -> list[str]:
    """Return the body lines of one page, without the OJ furniture block and footnotes.

    On every OJ page the body comes first, then the furniture block (EN, the
    "OJ L" date, the page number and the ELI link), then the page's footnotes.
    Pages without an ELI line, such as test fixtures, are returned whole.
    """
    lines = [line.strip() for line in page_text.split("\n")]
    eli = next((i for i, line in enumerate(lines) if _ELI_LINE.match(line)), None)
    if eli is None:
        return lines
    start = eli
    while start > 0 and (not lines[start - 1] or _FURNITURE_LINE.match(lines[start - 1])):
        start -= 1
    return lines[:start]


def _split_into_units(
    lines: list[str],
    strict: bool,
    parent_document: str,
    source_url: str,
    publication_date: str,
) -> list[StructuralUnit]:
    """Walk the document line by line and emit a unit whenever a valid header starts one."""
    units: list[StructuralUnit] = []
    last = {"recital": 0, "article": 0, "annex": 0, "definition": 0}
    region = "preamble" if strict else "any"
    draft: _Draft | None = (
        _Draft("Preamble", SectionType.GENERAL, [], title="Preamble") if strict else None
    )
    in_heading = False  # between a CHAPTER/SECTION heading and the next article
    in_definitions = False  # inside Article 3
    definitions_title = ""

    def flush() -> None:
        nonlocal draft
        if draft is not None:
            body = "\n".join(draft.lines).strip()
            if body:
                section_type = draft.section_type
                if section_type == SectionType.GENERAL and draft.label.startswith("Article "):
                    section_type = _classify_section(draft.title)
                units.append(
                    StructuralUnit(
                        article_number=draft.label,
                        section_type=section_type,
                        title=draft.title,
                        text=body,
                        chunk_index=len(units),
                        parent_document=parent_document,
                        source_url=source_url,
                        publication_date=publication_date,
                        cross_references=_extract_cross_references(body),
                    )
                )
        draft = None

    for i, line in enumerate(lines):
        if not line:
            continue

        # --- Region transitions (documents with OJ markers only) ---
        if strict:
            if region == "preamble" and line == _RECITALS_MARKER:
                draft.lines.append(line)
                flush()
                region = "recitals"
                continue
            if region in ("preamble", "recitals") and _ENACTING_MARKER.match(line):
                flush()
                region, in_heading = "articles", True
                continue
            if region == "articles" and _SIGNATURE_MARKER.match(line):
                flush()
                region, in_definitions = "signature", False
                continue

        allowed = _REGION_HEADERS[region]

        # --- Headers ---
        if "recital" in allowed and not in_definitions:
            m = _RECITAL_HEADER.match(line)
            if (
                m
                and _continues(int(m[1]), last["recital"], strict)
                and _starts_sentence(_next_nonempty(lines, i))
            ):
                flush()
                last["recital"] = int(m[1])
                draft = _Draft(f"Recital {m[1]}", SectionType.RECITAL, [line])
                continue

        if "article" in allowed:
            m = _ARTICLE_HEADER.match(line)
            if m and _continues(int(m[1]), last["article"], strict):
                flush()
                number = int(m[1])
                last["article"] = number
                in_heading, in_definitions = False, number == _DEFINITIONS_ARTICLE
                draft = _Draft(f"Article {number}", SectionType.GENERAL, [line], title_state="expecting")
                continue
            if _HEADING.match(line):
                flush()
                in_heading, in_definitions = True, False
                continue

        if "annex" in allowed:
            m = _ANNEX_HEADER.match(line)
            if m and _continues(_roman_to_int(m[1]), last["annex"], strict):
                flush()
                last["annex"] = _roman_to_int(m[1])
                in_definitions = False
                if region == "signature":
                    region = "annexes"
                draft = _Draft(f"Annex {m[1]}", SectionType.ANNEX, [line], title_state="expecting")
                continue

        if in_definitions:
            m = _DEFINITION_HEADER.match(line)
            if m and _continues(int(m[1]), last["definition"], strict):
                if draft is not None and draft.label == f"Article {_DEFINITIONS_ARTICLE}":
                    # Article 3's own draft only holds its heading and intro sentence;
                    # each definition repeats the heading instead.
                    definitions_title = draft.title
                    draft = None
                flush()
                last["definition"] = int(m[1])
                heading = [f"Article {_DEFINITIONS_ARTICLE}", definitions_title]
                draft = _Draft(
                    f"Article {_DEFINITIONS_ARTICLE}({m[1]})",
                    SectionType.DEFINITION,
                    [h for h in heading if h] + [line],
                    title=definitions_title,
                )
                continue

        # --- Body text ---
        if in_heading or region == "signature" or draft is None:
            continue  # chapter/section titles, signature block, text before the first unit
        if draft.title_state == "expecting":
            draft.title, draft.title_state = line, "open"
        elif draft.title_state == "open":
            if line[0].islower() or _OPEN_ENDING.search(draft.title):  # title wrapped
                draft.title += " " + line
            else:
                draft.title_state = "done"
        draft.lines.append(line)

    flush()
    return units


def _continues(number: int, last: int, strict: bool) -> bool:
    """True if `number` is the next in its sequence (the first must be 1 in strict mode)."""
    if last:
        return number == last + 1
    return number == 1 if strict else number >= 1


def _next_nonempty(lines: list[str], index: int) -> str:
    return next((line for line in lines[index + 1 :] if line), "")


def _starts_sentence(line: str) -> bool:
    """Recital text starts with a capital letter; a wrapped footnote marker does not."""
    return bool(line) and (line[0].isupper() or line[0] in "‘'\"")


def _roman_to_int(numeral: str) -> int:
    values = [_ROMAN_VALUES[c] for c in numeral]
    return sum(
        -v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values)
    )


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


def _validate_units(units: list[StructuralUnit], strict: bool) -> None:
    """Log what was found, and warn about gaps or duplicates in the numbering."""
    found = summarize_structure(units)
    logger.info(
        "Parsed %d recitals, %d articles, %d definitions, %d annexes (%d units).",
        len(found["recital"]),
        len(found["article"]) + (1 if found["definition"] else 0),
        len(found["definition"]),
        len(found["annex"]),
        len(units),
    )
    duplicates = [label for label, n in Counter(u.article_number for u in units).items() if n > 1]
    if duplicates:
        logger.warning(f"Parser validation: duplicate unit labels {duplicates[:10]}")
    if not strict:
        return
    articles = sorted(set(found["article"]) | ({_DEFINITIONS_ARTICLE} if found["definition"] else set()))
    for kind, numbers in (
        ("recital", found["recital"]),
        ("article", articles),
        ("definition", found["definition"]),
        ("annex", found["annex"]),
    ):
        if numbers and numbers != list(range(1, numbers[-1] + 1)):
            missing = sorted(set(range(1, numbers[-1] + 1)) - set(numbers))
            logger.warning(f"Parser validation: {kind} numbering has gaps, missing {missing[:20]}")
