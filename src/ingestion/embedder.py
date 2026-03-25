"""Embedding layer — calls OpenAI embeddings API in batches.

Logs estimated cost after each run so API spend is visible during experiments.
"""

import logging
import time

from openai import OpenAI

from src import config
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Pricing as of 2024 for text-embedding-3-small (USD per 1M tokens)
_PRICE_PER_1M = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

_BATCH_SIZE = 100  # stay well within OpenAI rate limits
_AVG_CHARS_PER_TOKEN = 4  # rough approximation
# text-embedding-3-small/large max is 8192 tokens; leave a small buffer
_MAX_TOKENS = 8000
_MAX_CHARS = _MAX_TOKENS * _AVG_CHARS_PER_TOKEN  # 32,000 chars


def _truncate(text: str, chunk_id: str) -> str:
    """Truncate text to fit within the model's token limit.

    Uses character count as a proxy for tokens. Logs a warning so oversized
    chunks are visible and can inform chunking strategy improvements.
    """
    if len(text) <= _MAX_CHARS:
        return text
    logger.warning(
        f"Chunk {chunk_id[:12]}... is {len(text):,} chars "
        f"(>{_MAX_CHARS:,} char limit). Truncating to {_MAX_CHARS:,} chars. "
        "Consider splitting this structural unit during ingestion."
    )
    return text[:_MAX_CHARS]


def embed_chunks(
    chunks: list[Chunk],
    model: str = config.EMBEDDING_MODEL,
) -> list[tuple[Chunk, list[float]]]:
    """Embed a list of Chunks using the OpenAI embeddings API.

    Returns a list of (Chunk, embedding_vector) pairs in the same order
    as the input.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    results: list[tuple[Chunk, list[float]]] = []

    total_chars = sum(len(c.text) for c in chunks)
    estimated_tokens = total_chars / _AVG_CHARS_PER_TOKEN
    price_per_1m = _PRICE_PER_1M.get(model, 0.02)
    estimated_cost = (estimated_tokens / 1_000_000) * price_per_1m

    logger.info(
        f"Embedding {len(chunks)} chunks ({total_chars:,} chars, "
        f"~{estimated_tokens:,.0f} tokens) with {model}. "
        f"Estimated cost: ${estimated_cost:.4f}"
    )

    start = time.time()
    for batch_start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[batch_start : batch_start + _BATCH_SIZE]
        texts = [_truncate(c.text, c.chunk_id) for c in batch]

        response = client.embeddings.create(model=model, input=texts)
        embeddings = [item.embedding for item in response.data]

        results.extend(zip(batch, embeddings))

        logger.debug(
            f"  Batch {batch_start // _BATCH_SIZE + 1}: "
            f"{len(batch)} chunks embedded."
        )

    elapsed = time.time() - start
    logger.info(f"Embedding complete in {elapsed:.1f}s.")
    return results
