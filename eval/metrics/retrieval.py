"""Retrieval evaluation metrics computed against the golden dataset.

Metrics:
    Recall@k     — is the correct chunk in the top k results?
    MRR          — mean reciprocal rank of first relevant result
    Precision@k  — what fraction of top k results are relevant?
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src import config
from src.retrieval import search
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class PerQueryResult:
    query_id: str
    query: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    precision_at_5: float
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]


@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    precision_at_5: float
    n_queries: int
    split: str
    per_query: list[PerQueryResult] = field(default_factory=list)


def compute_recall_at_k(
    results: list[RetrievalResult],
    relevant_ids: set[str],
    k: int,
) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0."""
    top_k_ids = {r.chunk_id for r in results[:k]}
    return 1.0 if top_k_ids & relevant_ids else 0.0


def compute_mrr(
    results: list[RetrievalResult],
    relevant_ids: set[str],
) -> float:
    """Reciprocal rank of the first relevant result (0.0 if none found)."""
    for i, result in enumerate(results):
        if result.chunk_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_precision_at_k(
    results: list[RetrievalResult],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r.chunk_id in relevant_ids)
    return hits / len(top_k)


def evaluate_retrieval(
    dataset_path: Path = config.GOLDEN_DATASET_PATH,
    split: str = "dev",
    top_k: int = 5,
) -> RetrievalMetrics:
    """Run all queries from the golden dataset through the retrieval pipeline
    and compute aggregate metrics.

    Args:
        dataset_path: path to golden_dataset.json.
        split: "dev" or "test".
        top_k: number of results to retrieve per query.
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    queries = [q for q in dataset if q.get("split") == split]
    if not queries:
        raise ValueError(f"No queries found with split='{split}' in {dataset_path}")

    logger.info(f"Evaluating retrieval on {len(queries)} '{split}' queries ...")

    per_query_results: list[PerQueryResult] = []

    for item in queries:
        query_id = item["query_id"]
        query = item["query"]
        relevant_ids = set(item.get("relevant_chunk_ids", []))

        results = search(query, top_k=top_k)
        retrieved_ids = [r.chunk_id for r in results]

        pq = PerQueryResult(
            query_id=query_id,
            query=query,
            recall_at_1=compute_recall_at_k(results, relevant_ids, 1),
            recall_at_3=compute_recall_at_k(results, relevant_ids, 3),
            recall_at_5=compute_recall_at_k(results, relevant_ids, 5),
            mrr=compute_mrr(results, relevant_ids),
            precision_at_5=compute_precision_at_k(results, relevant_ids, 5),
            retrieved_chunk_ids=retrieved_ids,
            relevant_chunk_ids=list(relevant_ids),
        )
        per_query_results.append(pq)

    n = len(per_query_results)
    metrics = RetrievalMetrics(
        recall_at_1=sum(r.recall_at_1 for r in per_query_results) / n,
        recall_at_3=sum(r.recall_at_3 for r in per_query_results) / n,
        recall_at_5=sum(r.recall_at_5 for r in per_query_results) / n,
        mrr=sum(r.mrr for r in per_query_results) / n,
        precision_at_5=sum(r.precision_at_5 for r in per_query_results) / n,
        n_queries=n,
        split=split,
        per_query=per_query_results,
    )

    _print_metrics(metrics)
    return metrics


def _print_metrics(m: RetrievalMetrics) -> None:
    print(f"\n{'='*50}")
    print(f"Retrieval Metrics — split={m.split}, n={m.n_queries}")
    print(f"{'='*50}")
    print(f"  Recall@1:     {m.recall_at_1:.3f}  (target ≥ 0.60)")
    print(f"  Recall@3:     {m.recall_at_3:.3f}  (target ≥ 0.85)")
    print(f"  Recall@5:     {m.recall_at_5:.3f}  (target ≥ 0.92)")
    print(f"  MRR:          {m.mrr:.3f}  (target ≥ 0.70)")
    print(f"  Precision@5:  {m.precision_at_5:.3f}  (target ≥ 0.60)")
    print(f"{'='*50}\n")
