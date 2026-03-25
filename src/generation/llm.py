"""LLM generation layer using the Anthropic API."""

import logging
import time
from dataclasses import dataclass, field
from anthropic import RateLimitError

import anthropic

from src import config
from src.generation.prompts import build_user_prompt, get_system_prompt
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

# USD per token (approximate; update if pricing changes)
_INPUT_PRICE_PER_1M = {
    "claude-sonnet-4-5": 3.0,
    "claude-haiku-4-5-20251001": 0.25,
    "claude-opus-4-6": 15.0,
}
_OUTPUT_PRICE_PER_1M = {
    "claude-sonnet-4-5": 15.0,
    "claude-haiku-4-5-20251001": 1.25,
    "claude-opus-4-6": 75.0,
}

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


@dataclass
class LLMResponse:
    answer: str
    query: str
    retrieved_chunks: list[RetrievalResult]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    prompt_version: str = field(default_factory=lambda: config.PROMPT_VERSION)


def generate_response(
    query: str,
    chunks: list[RetrievalResult],
    model: str = config.LLM_MODEL,
    temperature: float = config.LLM_TEMPERATURE,
    max_tokens: int = config.LLM_MAX_TOKENS,
    prompt_version: str = config.PROMPT_VERSION,
) -> LLMResponse:
    """Generate a grounded, cited answer using the Anthropic API.

    Args:
        query: the user's question.
        chunks: retrieved context chunks to inject into the prompt.
        model: Anthropic model ID.
        temperature: sampling temperature (0.1 default for regulatory accuracy).
        max_tokens: maximum output tokens.
        prompt_version: selects the system prompt variant (for E4 experiments).

    Returns:
        LLMResponse with the answer, token usage, latency, and cost.
    """
    system_prompt = get_system_prompt(prompt_version)
    user_prompt = build_user_prompt(query, chunks)

    start = time.time()
    for attempt in range(4):
        try:
            response = _get_client().messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            break
        except RateLimitError:
            wait = 60 * (attempt + 1)  # 60s, 120s, 180s, 240s
            logger.warning(f"Rate limit hit — waiting {wait}s before retry {attempt + 1}/3 ...")
            time.sleep(wait)
    else:
        raise RateLimitError  # re-raise after all retries exhausted
    latency_ms = (time.time() - start) * 1000

    answer = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    cost_usd = (
        input_tokens / 1_000_000 * _INPUT_PRICE_PER_1M.get(model, 3.0)
        + output_tokens / 1_000_000 * _OUTPUT_PRICE_PER_1M.get(model, 15.0)
    )

    logger.info(
        f"LLM response: {input_tokens} in / {output_tokens} out tokens, "
        f"${cost_usd:.5f}, {latency_ms:.0f}ms"
    )

    return LLMResponse(
        answer=answer,
        query=query,
        retrieved_chunks=chunks,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        prompt_version=prompt_version,
    )
