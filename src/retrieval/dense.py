"""Dense vector retrieval using ChromaDB."""

import logging

from openai import OpenAI

from src import config
from src.ingestion.store import get_collection
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _embed_query(query: str, model: str = config.EMBEDDING_MODEL) -> list[float]:
    client = _get_openai_client()
    response = client.embeddings.create(model=model, input=[query])
    return response.data[0].embedding


def search_dense(
    query: str,
    top_k: int = config.TOP_K_RETRIEVAL,
    filters: dict | None = None,
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> list[RetrievalResult]:
    """Semantic similarity search over the ChromaDB collection.

    Args:
        query: natural-language query string.
        top_k: number of results to return.
        filters: optional ChromaDB `where` filter dict (e.g. {"section_type": "obligation"}).
        collection_name: ChromaDB collection to query.

    Returns:
        List of RetrievalResult ordered by similarity (most similar first).
    """
    query_embedding = _embed_query(query)
    collection = get_collection(collection_name)

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if filters:
        kwargs["where"] = filters

    response = collection.query(**kwargs)

    results: list[RetrievalResult] = []
    for rank, (doc, meta, dist) in enumerate(
        zip(
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        )
    ):
        results.append(
            RetrievalResult(
                chunk_id=meta["chunk_hash"],
                text=doc,
                score=float(dist),
                article_number=meta.get("article_number") or None,
                section_type=meta.get("section_type", ""),
                title=meta.get("title", ""),
                parent_document=meta.get("parent_document", ""),
                rank=rank,
            )
        )

    logger.debug(f"Dense search returned {len(results)} results for: {query[:60]!r}")
    return results
