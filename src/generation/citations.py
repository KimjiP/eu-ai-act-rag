"""Citation extraction and verification.

Extracts article/annex/recital citations from generated answers and
verifies them against the retrieved chunks that were passed to the LLM.
"""

import re
from dataclasses import dataclass, field

from src.retrieval.models import RetrievalResult

# Matches: Article 17, Article 17(1), Article 17(1)(a), Annex III, Annex III Section 2, Recital 42
# No trailing \b because citations often end with ')' which is a non-word character.
_CITATION_PATTERN = re.compile(
    r"\b(Article\s+\d+(?:\(\d+\))?(?:\([a-z]\))?|Annex\s+[IVX]+(?:[,\s]+Section\s+\d+)?|Recital\s+\d+)",
    re.IGNORECASE,
)

# Banned phrases indicating bad framing (the LLM thinks the user provided the docs)
_BAD_FRAMING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"based on (the )?(documents?|transcripts?|text|files?|materials?) you provided",
        r"(the )?(documents?|transcripts?|files?) (you|the user) (provided|shared|uploaded|sent)",
        r"according to (the )?(documents?|files?) you",
        r"from (the )?(documents?|files?) provided by you",
    ]
]


@dataclass
class CitationVerification:
    accuracy: float  # fraction of extracted citations matched to retrieved chunks
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # cited but not in chunks
    bad_framing_detected: bool = False


def extract_citations(answer: str) -> list[str]:
    """Return deduplicated list of citation strings found in the answer text."""
    return list(dict.fromkeys(m.group(0) for m in _CITATION_PATTERN.finditer(answer)))


def verify_citations(
    citations: list[str],
    chunks: list[RetrievalResult],
) -> CitationVerification:
    """Check each extracted citation against the retrieved chunks.

    A citation is considered "matched" if the cited article/annex/recital
    label appears in any chunk's article_number field (case-insensitive,
    prefix match to handle "Article 17" matching "Article 17(1)").
    """
    # Build set of normalised article labels from retrieved chunks
    chunk_labels = set()
    for chunk in chunks:
        if chunk.article_number:
            chunk_labels.add(chunk.article_number.lower().strip())

    matched: list[str] = []
    missing: list[str] = []

    for citation in citations:
        citation_norm = citation.lower().strip()
        # Accept if any chunk label starts with the cited reference (or vice versa)
        if any(
            citation_norm.startswith(label) or label.startswith(citation_norm)
            for label in chunk_labels
        ):
            matched.append(citation)
        else:
            missing.append(citation)

    total = len(citations)
    accuracy = len(matched) / total if total > 0 else 1.0  # no citations = no error

    return CitationVerification(
        accuracy=accuracy,
        matched=matched,
        missing=missing,
    )


def detect_bad_framing(answer: str) -> bool:
    """Return True if the answer contains phrases indicating the LLM thinks
    the user provided the documents (a known failure mode)."""
    return any(p.search(answer) for p in _BAD_FRAMING_PATTERNS)
