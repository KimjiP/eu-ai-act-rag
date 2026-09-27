"""LLM-as-judge evaluation.

Uses Claude Haiku (not Sonnet) to keep judging costs low during iteration.
All judge prompts output structured JSON so scores can be parsed reliably.

Faithfulness, factual consistency and answer relevance check the answer against
the question and the retrieved passages. Completeness checks it against the
facts in the golden dataset, which come from the Act itself: an answer can be
faithful to the passages it was given and still miss the provision that answers
the question.
"""

import json
import logging
import re
from dataclasses import dataclass, field

import anthropic

from src import config
from eval.judges.prompts import (
    ANSWER_RELEVANCE_JUDGE_PROMPT,
    COMPLETENESS_JUDGE_PROMPT,
    FACTUAL_CONSISTENCY_JUDGE_PROMPT,
    FAITHFULNESS_JUDGE_PROMPT,
)
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-haiku-4-5-20251001"

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


@dataclass
class JudgeResult:
    score: float
    reasoning: str
    judge_model: str
    extra: dict = field(default_factory=dict)  # unsupported_claims, contradictions, etc.


def _call_judge(prompt: str) -> JudgeResult:
    """Call the judge LLM and parse the JSON response."""
    response = _get_client().messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        temperature=0.0,  # deterministic judging
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in response.content if block.type == "text").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON from the response if the model added extra text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {"score": 0.0, "reasoning": raw}

    return JudgeResult(
        score=float(data.get("score", 0.0)),
        reasoning=data.get("reasoning", ""),
        judge_model=JUDGE_MODEL,
        extra={k: v for k, v in data.items() if k not in ("score", "reasoning")},
    )


def _format_context(chunks: list[RetrievalResult]) -> str:
    return "\n\n".join(
        f"[{i+1}] {c.article_number or '(unlabelled)'}\n{c.text}" for i, c in enumerate(chunks)
    )


def judge_answer_relevance(query: str, answer: str) -> JudgeResult:
    prompt = ANSWER_RELEVANCE_JUDGE_PROMPT.format(query=query, answer=answer)
    return _call_judge(prompt)


def judge_faithfulness(answer: str, chunks: list[RetrievalResult]) -> JudgeResult:
    context = _format_context(chunks)
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=answer)
    return _call_judge(prompt)


def judge_factual_consistency(answer: str, chunks: list[RetrievalResult]) -> JudgeResult:
    context = _format_context(chunks)
    prompt = FACTUAL_CONSISTENCY_JUDGE_PROMPT.format(context=context, answer=answer)
    return _call_judge(prompt)


def judge_completeness(query: str, answer: str, expected_elements: list[str]) -> JudgeResult:
    """Fraction of the expected facts the answer states, counted in code from per-fact verdicts."""
    facts = "\n".join(f"{i}. {element}" for i, element in enumerate(expected_elements, 1))
    result = _call_judge(COMPLETENESS_JUDGE_PROMPT.format(query=query, facts=facts, answer=answer))
    covered = result.extra.get("covered", [])
    if len(covered) != len(expected_elements):
        logger.warning(
            f"Completeness judge returned {len(covered)} verdicts for {len(expected_elements)} facts"
        )
    result.score = sum(1 for c in covered[: len(expected_elements)] if c is True) / len(expected_elements)
    return result


def run_all_judges(
    query: str,
    answer: str,
    chunks: list[RetrievalResult],
    expected_elements: list[str] | None = None,
) -> dict[str, JudgeResult]:
    """Run the judges and return results keyed by judge name.

    Completeness runs only when expected answer elements are given.
    """
    results = {
        "answer_relevance": judge_answer_relevance(query, answer),
        "faithfulness": judge_faithfulness(answer, chunks),
        "factual_consistency": judge_factual_consistency(answer, chunks),
    }
    if expected_elements:
        results["completeness"] = judge_completeness(query, answer, expected_elements)
    return results
