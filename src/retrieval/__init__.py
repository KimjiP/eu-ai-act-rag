"""Retrieval layer public API.

The rest of the application imports `search` from here — never from the
individual strategy modules directly. Changing the active strategy for
an experiment means updating config.RETRIEVAL_STRATEGY only.
"""

from src import config
from src.retrieval.models import RetrievalResult


def search(
    query: str,
    top_k: int = config.TOP_K_RETRIEVAL,
    filters: dict | None = None,
) -> list[RetrievalResult]:
    """Run retrieval using the strategy configured in config.RETRIEVAL_STRATEGY.

    Strategies:
        "dense"  — semantic vector search only (ChromaDB)
        "hybrid" — dense + BM25 via Reciprocal Rank Fusion (default)

    If config.RERANKER_ENABLED is True, results are re-scored by the
    cross-encoder reranker after the primary retrieval step.
    """
    if config.RETRIEVAL_STRATEGY == "dense":
        from src.retrieval.dense import search_dense

        results = search_dense(query, top_k=top_k, filters=filters)
    else:
        from src.retrieval.hybrid import search_hybrid

        results = search_hybrid(query, top_k=top_k, filters=filters)

    if config.RERANKER_ENABLED:
        from src.retrieval.reranker import get_reranker

        results = get_reranker().rerank(query, results, top_k=top_k)

    return results
