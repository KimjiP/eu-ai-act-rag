"""ChromaDB storage layer for the ingestion pipeline.

Upserts chunks using `chunk_hash` as the document ID, making all ingestion
runs idempotent — re-running the pipeline on the same PDFs does not create
duplicates.

Also exposes `get_collection()` for use by the retrieval layer.
"""

import logging
from dataclasses import asdict

import chromadb

from src import config
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    return _client


def get_collection(
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> chromadb.Collection:
    """Return the ChromaDB collection, creating it if it does not exist."""
    client = _get_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "l2"},
    )


def _chunk_to_metadata(chunk: Chunk) -> dict:
    """Convert a Chunk to a ChromaDB-compatible metadata dict.

    ChromaDB metadata values must be str, int, or float — no lists or None.
    """
    return {
        "article_number": chunk.article_number or "",
        "section_type": chunk.section_type,
        "title": chunk.title,
        "chunk_index": chunk.chunk_index,
        "parent_document": chunk.parent_document,
        "source_url": chunk.source_url,
        "publication_date": chunk.publication_date,
        "ingestion_run_id": chunk.ingestion_run_id,
        "chunk_hash": chunk.chunk_hash,
        # cross_references stored as pipe-separated string
        "cross_references": "|".join(chunk.cross_references),
    }


def store_chunks(
    chunks_with_embeddings: list[tuple[Chunk, list[float]]],
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> None:
    """Upsert chunks and their embeddings into ChromaDB."""
    collection = get_collection(collection_name)

    ids = [chunk.chunk_id for chunk, _ in chunks_with_embeddings]
    documents = [chunk.text for chunk, _ in chunks_with_embeddings]
    embeddings = [emb for _, emb in chunks_with_embeddings]
    metadatas = [_chunk_to_metadata(chunk) for chunk, _ in chunks_with_embeddings]

    # Upsert in batches of 500 (ChromaDB recommendation)
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    logger.info(
        f"Upserted {len(ids)} chunks into ChromaDB collection '{collection_name}'. "
        f"Collection now has {collection.count()} total chunks."
    )
