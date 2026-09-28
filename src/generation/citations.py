"""Citation extraction and verification, plus the other checks run on every answer.

Extracts article/annex/recital citations from generated answers and verifies
them against the retrieved chunks that were passed to the LLM.
"""

import re
from dataclasses import dataclass, field

from src.retrieval.models import RetrievalResult

# Matches: Article 17, Article 17(1), Article 17(1)(a), Article 5(1)(h)(iii), Article 3(56),
# Annex III, Annex III, Section 2, Recital 42.
# No trailing \b because citations often end with ')' which is a non-word character.
_CITATION_PATTERN = re.compile(
    r"\b(Article\s+\d+(?:\((?:\d+|[a-z]{1,4})\))*"
    r"|Annex\s+[IVX]+(?:[,\s]+Section\s+[0-9A-Z]\b)?"
    r"|Recital\s+\d+)",
    re.IGNORECASE,
)

# Text right after a citation that makes it a reference to another legal act,
# e.g. "Article 22 of Regulation (EU) No 1025/2012", "Article 16 TFEU"
_OTHER_ACT = re.compile(
    r"^,?\s*(?:(?:of|in)\s+(?:that\s+|the\s+)?(?:Regulation|Directive|Decision|Treaty|Charter|Council)"
    r"|(?:TFEU|TEU)\b)",
    re.IGNORECASE,
)

_REFERENCE = re.compile(r"^(Article|Annex|Recital)\s+(\d+|[IVXLC]+)(.*)$", re.IGNORECASE)
_PATH_PART = re.compile(r"\((\d+|[a-z]{1,4})\)|Section\s+(\w+)", re.IGNORECASE)

# The sentence the system prompt tells the model to use when the context cannot answer,
# and how far into the answer it must start for the answer to count as a decline
_DECLINE_PATTERN = re.compile(
    r"does not contain (sufficient|enough) information to answer", re.IGNORECASE
)
_DECLINE_WITHIN = 120

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


@dataclass(frozen=True)
class ProvisionRef:
    """A reference such as Article 17(1)(a): the provision, and the path below it."""

    kind: str  # "article" | "annex" | "recital"
    number: str  # "17", "III", "42"
    path: tuple[str, ...] = ()  # ("1", "a") for Article 17(1)(a)

    def overlaps(self, other: "ProvisionRef") -> bool:
        """True if one reference contains the other, e.g. Article 17 and Article 17(1).

        Article 3(12) and Article 3(56) do not overlap, and neither do Article 1
        and Article 13.
        """
        if (self.kind, self.number) != (other.kind, other.number):
            return False
        shorter, longer = sorted((self.path, other.path), key=len)
        return longer[: len(shorter)] == shorter


@dataclass
class CitationVerification:
    accuracy: float  # fraction of extracted citations matched to retrieved chunks
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # cited but not in chunks
    # the subset of `missing` that a retrieved chunk names, e.g. Article 17 refers to
    # "the risk management system referred to in Article 9"
    cross_referenced: list[str] = field(default_factory=list)
    bad_framing_detected: bool = False


def parse_reference(text: str) -> ProvisionRef | None:
    """Parse "Article 17(1)(a)", "Annex III, Section 2" or "Recital 42" (None otherwise)."""
    m = _REFERENCE.match(text.strip())
    if not m:
        return None
    kind = m[1].lower()
    number = m[2].upper() if kind == "annex" else m[2]
    if kind != "annex" and not number.isdigit():
        return None
    path = tuple(
        (paren or f"section {section}").lower() for paren, section in _PATH_PART.findall(m[3])
    )
    return ProvisionRef(kind, number, path)


def extract_citations(answer: str) -> list[str]:
    """Return deduplicated list of AI Act citations found in the answer text.

    References to other legal acts ("Article 22 of Regulation (EU) No 1025/2012")
    are skipped: they are not claims about the AI Act's own articles.
    """
    return list(
        dict.fromkeys(
            m.group(0)
            for m in _CITATION_PATTERN.finditer(answer)
            if not _OTHER_ACT.match(answer[m.end() :])
        )
    )


def verify_citations(
    citations: list[str],
    chunks: list[RetrievalResult],
) -> CitationVerification:
    """Check each extracted citation against the labels of the retrieved chunks.

    A citation matches a chunk when both refer to the same provision and one
    contains the other: "Article 17(1)" matches a chunk labelled "Article 17",
    and "Article 3" matches "Article 3(56)". "Article 13" never matches
    "Article 1", and "Annex III" never matches "Annex I".
    """
    chunk_refs = [
        ref for chunk in chunks if chunk.article_number
        if (ref := parse_reference(chunk.article_number))
    ]

    matched: list[str] = []
    missing: list[str] = []
    for citation in citations:
        ref = parse_reference(citation)
        if ref and any(ref.overlaps(chunk_ref) for chunk_ref in chunk_refs):
            matched.append(citation)
        else:
            missing.append(citation)

    total = len(citations)
    accuracy = len(matched) / total if total > 0 else 1.0  # no citations = no error
    context = "\n".join(chunk.text for chunk in chunks)

    return CitationVerification(
        accuracy=accuracy,
        matched=matched,
        missing=missing,
        cross_referenced=[c for c in missing if names_provision(context, c)],
    )


def names_provision(text: str, citation: str) -> bool:
    """True if `text` refers to the cited provision by number, e.g. "... referred to in Article 72"."""
    ref = parse_reference(citation)
    if ref is None:
        return False
    word = {"article": "Article", "annex": "Annex", "recital": "Recital"}[ref.kind]
    return bool(re.search(rf"\b{word}\s+{ref.number}(?![\dIVXLC])", text, re.IGNORECASE))


def detect_decline(answer: str) -> bool:
    """Return True if the answer opens with the system prompt's decline sentence.

    A decline leads with the sentence. An answer that uses it later, to say one
    part of the question is not covered, is a partial answer, not a decline.
    """
    return bool(_DECLINE_PATTERN.search(answer.lstrip()[:_DECLINE_WITHIN]))


def detect_bad_framing(answer: str) -> bool:
    """Return True if the answer contains phrases indicating the LLM thinks
    the user provided the documents (a known failure mode)."""
    return any(p.search(answer) for p in _BAD_FRAMING_PATTERNS)
