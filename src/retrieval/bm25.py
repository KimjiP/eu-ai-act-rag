"""BM25 lexical retrieval over the chunk corpus.

The index is built from the processed chunks JSON file at startup.
This avoids any extra serialisation format while remaining fast enough
for the EU AI Act corpus size (~hundreds of chunks).
"""

import json
import logging
import re

from rank_bm25 import BM25Okapi

from src import config
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25Index:
    """BM25 index over all chunks from the processed chunks JSON."""

    def __init__(self, chunks_json_path=config.CHUNKS_JSON_PATH) -> None:
        with open(chunks_json_path) as f:
            chunks = json.load(f)

        self._chunks = chunks  # list of dicts
        tokenised = [_tokenise(c["text"]) for c in chunks]
        self._index = BM25Okapi(tokenised)
        logger.info(f"BM25 index built from {len(chunks)} chunks.")

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL,
    ) -> list[RetrievalResult]:
        """BM25 lexical search.

        Returns top_k results ordered by BM25 score (highest first).
        Score is the raw BM25 score (higher = more relevant).
        """
        tokens = _tokenise(query)
        scores = self._index.get_scores(tokens)

        # Get top_k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_indices):
            chunk = self._chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=float(scores[idx]),
                    article_number=chunk.get("article_number") or None,
                    section_type=chunk.get("section_type", ""),
                    title=chunk.get("title", ""),
                    parent_document=chunk.get("parent_document", ""),
                    rank=rank,
                )
            )

        logger.debug(f"BM25 search returned {len(results)} results for: {query[:60]!r}")
        return results


# Module-level singleton — built lazily on first use
_bm25_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index
