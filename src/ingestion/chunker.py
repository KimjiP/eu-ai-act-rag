"""Chunking strategies for the ingestion pipeline.

The `Chunk` dataclass defined here is the canonical interface between the
ingestion layer and the retrieval layer. Retrieval code never sees
`RawPage` or `StructuralUnit` — only `Chunk`.

Two strategies are provided so experiment E1 can be run without touching
any other module:
  - chunk_structural  (baseline) — one Chunk per StructuralUnit
  - chunk_fixed_size  (E1)       — sliding window over concatenated text
"""

import hashlib
import re
from dataclasses import dataclass, field

from src.ingestion.parser import StructuralUnit


# ---------------------------------------------------------------------------
# Chunk dataclass — interface between ingestion and retrieval
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str  # SHA-256 of text content (stable, content-addressable)
    text: str
    article_number: str | None
    section_type: str  # SectionType.value string for JSON/ChromaDB compatibility
    title: str
    chunk_index: int
    parent_document: str
    source_url: str
    publication_date: str
    ingestion_run_id: str
    chunk_hash: str  # same as chunk_id; kept as explicit field for clarity
    cross_references: list[str] = field(default_factory=list)


def _make_chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Strategy 1: Structure-aware (baseline)
# ---------------------------------------------------------------------------


def chunk_structural(
    units: list[StructuralUnit],
    ingestion_run_id: str = "",
) -> list[Chunk]:
    """One Chunk per StructuralUnit (baseline chunking strategy).

    Preserves the legal document structure exactly — each article,
    recital, or annex section becomes its own chunk.
    """
    chunks: list[Chunk] = []
    for unit in units:
        text = unit.text.strip()
        if not text:
            continue
        chunk_hash = _make_chunk_id(text)
        chunks.append(
            Chunk(
                chunk_id=chunk_hash,
                text=text,
                article_number=unit.article_number,
                section_type=unit.section_type.value,
                title=unit.title,
                chunk_index=unit.chunk_index,
                parent_document=unit.parent_document,
                source_url=unit.source_url,
                publication_date=unit.publication_date,
                ingestion_run_id=ingestion_run_id,
                chunk_hash=chunk_hash,
                cross_references=unit.cross_references,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Fixed-size overlapping (experiment E1)
# ---------------------------------------------------------------------------


def chunk_fixed_size(
    units: list[StructuralUnit],
    token_limit: int = 500,
    overlap: int = 50,
    ingestion_run_id: str = "",
) -> list[Chunk]:
    """Sliding-window chunks over concatenated document text.

    Tokens are approximated as whitespace-split words (close enough for
    comparison purposes; true tokenisation would require loading a tokenizer).
    Metadata (article_number, section_type) is inherited from the structural
    unit that contains the start of each window.
    """
    # Build a flat word list with per-word metadata back-references
    words: list[str] = []
    word_meta: list[StructuralUnit] = []  # which unit does this word belong to?

    for unit in units:
        unit_words = re.split(r"\s+", unit.text.strip())
        words.extend(unit_words)
        word_meta.extend([unit] * len(unit_words))

    chunks: list[Chunk] = []
    chunk_index = 0
    i = 0

    while i < len(words):
        window = words[i : i + token_limit]
        text = " ".join(window).strip()
        if not text:
            i += token_limit - overlap
            continue

        meta = word_meta[i]  # metadata from the unit where the window starts
        chunk_hash = _make_chunk_id(text)
        chunks.append(
            Chunk(
                chunk_id=chunk_hash,
                text=text,
                article_number=meta.article_number,
                section_type=meta.section_type.value,
                title=meta.title,
                chunk_index=chunk_index,
                parent_document=meta.parent_document,
                source_url=meta.source_url,
                publication_date=meta.publication_date,
                ingestion_run_id=ingestion_run_id,
                chunk_hash=chunk_hash,
                cross_references=meta.cross_references,
            )
        )
        chunk_index += 1
        i += token_limit - overlap

    return chunks
