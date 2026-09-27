"""FastAPI application for the EU AI Act RAG pipeline.

Run with:
    uv run uvicorn src.api.main:app --reload

Endpoints:
    POST /query         — answer a regulatory question
    GET  /health        — liveness check
    GET  /chunks/{id}   — retrieve full chunk text for source inspection
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config
from src.api.query import QueryResponse, answer_query
from src.ingestion.store import get_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="EU AI Act Compliance Q&A",
    description="RAG pipeline for querying EU AI Act regulatory text with citations.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    top_k: int = config.TOP_K_RETRIEVAL
    filters: dict | None = None


class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    article_number: str | None
    section_type: str
    title: str
    parent_document: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/query")
async def query_endpoint(request: QueryRequest) -> dict:
    """Answer a regulatory question with citations from the EU AI Act corpus."""
    response: QueryResponse = answer_query(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
    )
    return {
        "query": response.query,
        "answered": response.answered,
        "answer": response.answer,
        "citations": response.citations,
        "citation_accuracy": (
            response.citation_verification.accuracy
            if response.citation_verification
            else None
        ),
        "retrieved_chunks": [
            {
                "chunk_id": c.chunk_id,
                "article_number": c.article_number,
                "title": c.title,
                "score": c.score,
                "rank": c.rank,
            }
            for c in response.retrieved_chunks
        ],
        "latency_ms": round(response.latency_ms, 1),
        "cost_usd": round(response.cost_usd, 6),
        "experiment_id": response.experiment_id,
        "corpus": config.CORPUS_DESCRIPTION,
    }


@app.get("/health")
async def health() -> dict:
    """Liveness check — returns collection size."""
    collection = get_collection()
    return {"status": "ok", "collection_size": collection.count()}


@app.get("/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(chunk_id: str) -> ChunkResponse:
    """Return full chunk text by chunk_id for source inspection ('one-click view')."""
    collection = get_collection()
    result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])

    if not result["ids"]:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found.")

    doc = result["documents"][0]
    meta = result["metadatas"][0]
    return ChunkResponse(
        chunk_id=chunk_id,
        text=doc,
        article_number=meta.get("article_number") or None,
        section_type=meta.get("section_type", ""),
        title=meta.get("title", ""),
        parent_document=meta.get("parent_document", ""),
    )
