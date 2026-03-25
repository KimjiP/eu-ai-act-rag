"""Automated (code-based) response quality metrics.

These are fast, deterministic metrics that do not require an LLM call.
LLM-as-judge metrics (faithfulness, factual consistency, answer relevance)
are in eval/judges/.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from src import config
from src.api.query import QueryResponse, answer_query
from src.generation.citations import detect_bad_framing, extract_citations, verify_citations

logger = logging.getLogger(__name__)

# Rough heuristic: a "factual claim" is a sentence that doesn't start with "Note",
# "Disclaimer", or "⚠️" and is longer than 20 chars.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _count_factual_sentences(text: str) -> int:
    sentences = _SENTENCE_SPLIT.split(text)
    return sum(
        1
        for s in sentences
        if len(s.strip()) > 20
        and not s.strip().startswith(("Note", "Disclaimer", "⚠️", "This is informational"))
    )


@dataclass
class PerResponseResult:
    query_id: str
    query: str
    answered: bool
    citation_accuracy: float
    citation_completeness: float
    bad_framing: bool
    graceful_failure_correct: bool  # True if unanswerable ↔ not answered, or answerable ↔ answered
    cost_usd: float
    latency_ms: float


@dataclass
class ResponseMetrics:
    citation_accuracy: float
    citation_completeness: float
    bad_framing_rate: float
    graceful_failure_rate: float
    avg_cost_usd: float
    avg_latency_ms: float
    n_queries: int
    split: str
    per_query: list[PerResponseResult] = field(default_factory=list)


def evaluate_responses(
    dataset_path: Path = config.GOLDEN_DATASET_PATH,
    split: str = "dev",
    output_path: Path | None = None,
    delay_seconds: float = 25.0,
    top_k: int = 3,
) -> ResponseMetrics:
    """Run all golden dataset queries through the full pipeline and compute
    automated response quality metrics.

    Args:
        dataset_path: path to golden_dataset.json.
        split: "dev" or "test".
        output_path: if provided, saves per-query results as JSON.
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    queries = [q for q in dataset if q.get("split") == split]
    if not queries:
        raise ValueError(f"No queries found with split='{split}' in {dataset_path}")

    logger.info(f"Evaluating responses on {len(queries)} '{split}' queries ...")

    per_query: list[PerResponseResult] = []

    for i, item in enumerate(queries):
        query_id = item["query_id"]
        query = item["query"]
        is_unanswerable = item.get("is_unanswerable", False)

        if i > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)

        response: QueryResponse = answer_query(query, top_k=top_k)

        # Citation accuracy — from the verification already done in query.py
        citation_accuracy = (
            response.citation_verification.accuracy
            if response.citation_verification
            else 1.0
        )

        # Citation completeness: fraction of factual sentences that have a citation
        if response.answer and response.answered:
            factual_sentences = _count_factual_sentences(response.answer)
            citations_in_answer = len(extract_citations(response.answer))
            citation_completeness = (
                min(citations_in_answer / factual_sentences, 1.0)
                if factual_sentences > 0
                else 1.0
            )
        else:
            citation_completeness = 1.0  # unanswered queries don't need citations

        # Graceful failure correctness
        graceful_failure_correct = (is_unanswerable and not response.answered) or (
            not is_unanswerable and response.answered
        )

        per_query.append(
            PerResponseResult(
                query_id=query_id,
                query=query,
                answered=response.answered,
                citation_accuracy=citation_accuracy,
                citation_completeness=citation_completeness,
                bad_framing=response.bad_framing_detected,
                graceful_failure_correct=graceful_failure_correct,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )
        )

    n = len(per_query)
    metrics = ResponseMetrics(
        citation_accuracy=sum(r.citation_accuracy for r in per_query) / n,
        citation_completeness=sum(r.citation_completeness for r in per_query) / n,
        bad_framing_rate=sum(1 for r in per_query if r.bad_framing) / n,
        graceful_failure_rate=sum(1 for r in per_query if r.graceful_failure_correct) / n,
        avg_cost_usd=sum(r.cost_usd for r in per_query) / n,
        avg_latency_ms=sum(r.latency_ms for r in per_query) / n,
        n_queries=n,
        split=split,
        per_query=per_query,
    )

    _print_metrics(metrics)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "split": split,
                    "metrics": {
                        "citation_accuracy": metrics.citation_accuracy,
                        "citation_completeness": metrics.citation_completeness,
                        "bad_framing_rate": metrics.bad_framing_rate,
                        "graceful_failure_rate": metrics.graceful_failure_rate,
                        "avg_cost_usd": metrics.avg_cost_usd,
                        "avg_latency_ms": metrics.avg_latency_ms,
                    },
                    "per_query": [
                        {
                            "query_id": r.query_id,
                            "query": r.query,
                            "answered": r.answered,
                            "citation_accuracy": r.citation_accuracy,
                            "citation_completeness": r.citation_completeness,
                            "bad_framing": r.bad_framing,
                            "graceful_failure_correct": r.graceful_failure_correct,
                            "cost_usd": r.cost_usd,
                            "latency_ms": r.latency_ms,
                        }
                        for r in per_query
                    ],
                },
                f,
                indent=2,
            )
        logger.info(f"Response eval results saved to {output_path}")

    return metrics


def _print_metrics(m: ResponseMetrics) -> None:
    print(f"\n{'='*55}")
    print(f"Response Metrics — split={m.split}, n={m.n_queries}")
    print(f"{'='*55}")
    print(f"  Citation accuracy:     {m.citation_accuracy:.3f}  (target ≥ 0.80)")
    print(f"  Citation completeness: {m.citation_completeness:.3f}  (target ≥ 0.75)")
    print(f"  Bad framing rate:      {m.bad_framing_rate:.3f}  (target ≤ 0.05)")
    print(f"  Graceful failure rate: {m.graceful_failure_rate:.3f}  (target = 1.00)")
    print(f"  Avg cost/query:        ${m.avg_cost_usd:.5f}")
    print(f"  Avg latency:           {m.avg_latency_ms:.0f}ms")
    print(f"{'='*55}\n")
