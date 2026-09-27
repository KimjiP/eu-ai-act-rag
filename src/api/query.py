"""Query orchestrator — the core end-to-end pipeline.

Used by both the FastAPI layer and the Streamlit frontend.
"""

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field

from src import config
from src.generation.citations import (
    CitationVerification,
    detect_bad_framing,
    detect_decline,
    extract_citations,
    verify_citations,
)
from src.generation.llm import LLMResponse, generate_response
from src.retrieval import search
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

_GRACEFUL_FAILURE_MESSAGE = (
    "The provided regulatory text does not contain sufficient information to answer "
    "this question with confidence. This may require national implementation guidance "
    "or other documents not currently in the corpus.\n\n"
    "⚠️ This is informational support for regulatory research only, not legal advice. "
    "Consult qualified legal counsel for compliance decisions."
)


@dataclass
class QueryResponse:
    query: str
    answered: bool
    answer: str | None
    retrieved_chunks: list[RetrievalResult]
    citations: list[str]
    citation_verification: CitationVerification | None
    bad_framing_detected: bool
    latency_ms: float
    cost_usd: float
    experiment_id: str = field(default_factory=lambda: config.EXPERIMENT_ID)


def answer_query(
    query: str,
    top_k: int = config.TOP_K_RETRIEVAL,
    filters: dict | None = None,
) -> QueryResponse:
    """Full RAG pipeline: retrieve → generate → verify citations.

    Graceful failure: the system prompt tells the model to decline with a fixed
    sentence when the retrieved text cannot answer the question, and
    `answered` is False when it does. If retrieval returns nothing at all,
    the pipeline declines without calling the LLM.
    """
    start = time.time()

    # 1. Retrieve
    chunks = search(query, top_k=top_k, filters=filters)

    # 2. Nothing retrieved: decline without an LLM call
    if not chunks:
        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"No chunks retrieved for query: {query[:80]!r} — graceful decline.")
        response = QueryResponse(
            query=query,
            answered=False,
            answer=_GRACEFUL_FAILURE_MESSAGE,
            retrieved_chunks=[],
            citations=[],
            citation_verification=None,
            bad_framing_detected=False,
            latency_ms=elapsed_ms,
            cost_usd=0.0,
        )
        _log_query(response)
        return response

    # 3. Generate
    llm_response: LLMResponse = generate_response(query, chunks)

    # 4. Citation extraction and verification
    citations = extract_citations(llm_response.answer)
    citation_verification = verify_citations(citations, chunks)
    bad_framing = detect_bad_framing(llm_response.answer)

    if bad_framing:
        logger.warning(f"Bad framing detected for query: {query[:80]!r}")

    elapsed_ms = (time.time() - start) * 1000
    response = QueryResponse(
        query=query,
        answered=not detect_decline(llm_response.answer),
        answer=llm_response.answer,
        retrieved_chunks=chunks,
        citations=citations,
        citation_verification=citation_verification,
        bad_framing_detected=bad_framing,
        latency_ms=elapsed_ms,
        cost_usd=llm_response.cost_usd,
    )
    _log_query(response)
    return response


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def _log_query(response: QueryResponse) -> None:
    """Append the full query+response to the JSONL query log."""
    try:
        config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "query": response.query,
            "answered": response.answered,
            "chunk_ids": [c.chunk_id for c in response.retrieved_chunks],
            "chunk_scores": [c.score for c in response.retrieved_chunks],
            "citations": response.citations,
            "citation_accuracy": (
                response.citation_verification.accuracy
                if response.citation_verification
                else None
            ),
            "bad_framing": response.bad_framing_detected,
            "latency_ms": round(response.latency_ms, 1),
            "cost_usd": round(response.cost_usd, 6),
            "experiment_id": response.experiment_id,
        }
        with open(config.QUERY_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write query log: {e}")
