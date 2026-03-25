"""Shared data models for the retrieval layer.

`RetrievalResult` is the canonical interface between retrieval and generation.
The generation layer never touches ChromaDB directly — it only receives
a list of RetrievalResult objects.
"""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float  # L2 distance from ChromaDB (lower = more similar) or RRF score
    article_number: str | None
    section_type: str
    title: str
    parent_document: str
    rank: int  # 0-indexed position in the result list
