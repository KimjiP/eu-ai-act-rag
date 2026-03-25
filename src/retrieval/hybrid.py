"""Hybrid retrieval combining dense vector search and BM25 via Reciprocal Rank Fusion.

RRF formula: score(d) = sum_over_rankings( 1 / (k + rank(d)) )
where k=60 is the standard RRF constant that dampens the impact of top ranks.

alpha controls how many candidates come from each source before fusion:
  alpha=0.5 means equal top_k candidates from both dense and BM25.
"""

import logging

from src import config
from src.retrieval.bm25 import get_bm25_index
from src.retrieval.dense import search_dense
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)  # rank is 0-indexed, so +1


def search_hybrid(
    query: str,
    top_k: int = config.TOP_K_RETRIEVAL,
    alpha: float = config.HYBRID_ALPHA,
    filters: dict | None = None,
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> list[RetrievalResult]:
    """Hybrid retrieval using Reciprocal Rank Fusion of dense + BM25 results.

    Args:
        query: natural-language query string.
        top_k: number of results to return after fusion.
        alpha: ignored in RRF (both lists contribute equally); kept for API
               consistency in case a weighted fusion variant is needed later.
        filters: optional ChromaDB metadata filter (dense only; BM25 has no filter).
        collection_name: ChromaDB collection to query.

    Returns:
        List of RetrievalResult re-ranked by RRF score, length <= top_k.
    """
    # Fetch more candidates than needed to give RRF enough material to work with
    candidate_k = max(top_k * 3, 20)

    dense_results = search_dense(
        query, top_k=candidate_k, filters=filters, collection_name=collection_name
    )
    bm25_results = get_bm25_index().search(query, top_k=candidate_k)

    # Accumulate RRF scores per chunk_id
    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(dense_results):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + _rrf_score(
            rank
        )
        chunk_by_id[result.chunk_id] = result

    for rank, result in enumerate(bm25_results):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + _rrf_score(
            rank
        )
        if result.chunk_id not in chunk_by_id:
            chunk_by_id[result.chunk_id] = result

    # Sort by RRF score descending and take top_k
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[
        :top_k
    ]

    results: list[RetrievalResult] = []
    for final_rank, chunk_id in enumerate(sorted_ids):
        result = chunk_by_id[chunk_id]
        results.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=rrf_scores[chunk_id],  # RRF score (higher = more relevant)
                article_number=result.article_number,
                section_type=result.section_type,
                title=result.title,
                parent_document=result.parent_document,
                rank=final_rank,
            )
        )

    logger.debug(
        f"Hybrid search: {len(dense_results)} dense + {len(bm25_results)} BM25 → "
        f"{len(results)} fused results."
    )
    return results
