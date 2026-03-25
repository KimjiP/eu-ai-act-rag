"""CLI: run retrieval evaluation against the golden dataset.

Usage:
    uv run python eval/metrics/run_retrieval_eval.py
    uv run python eval/metrics/run_retrieval_eval.py --split test
    uv run python eval/metrics/run_retrieval_eval.py --split dev --top-k 3
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src import config
from eval.metrics.retrieval import evaluate_retrieval

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.GOLDEN_DATASET_PATH,
        help="Path to golden_dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save metrics JSON to this path (defaults to experiments/retrieval_<split>.json)",
    )
    args = parser.parse_args()

    output_path = args.output or (
        config.EXPERIMENTS_DIR / f"retrieval_{args.split}_{config.EXPERIMENT_ID}.json"
    )

    metrics = evaluate_retrieval(
        dataset_path=args.dataset,
        split=args.split,
        top_k=args.top_k,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": config.EXPERIMENT_ID,
        "retrieval_strategy": config.RETRIEVAL_STRATEGY,
        "split": args.split,
        "top_k": args.top_k,
        "metrics": {
            "recall_at_1": metrics.recall_at_1,
            "recall_at_3": metrics.recall_at_3,
            "recall_at_5": metrics.recall_at_5,
            "mrr": metrics.mrr,
            "precision_at_5": metrics.precision_at_5,
        },
        "per_query": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "recall_at_1": r.recall_at_1,
                "recall_at_3": r.recall_at_3,
                "recall_at_5": r.recall_at_5,
                "mrr": r.mrr,
                "precision_at_5": r.precision_at_5,
                "retrieved_chunk_ids": r.retrieved_chunk_ids,
                "relevant_chunk_ids": r.relevant_chunk_ids,
            }
            for r in metrics.per_query
        ],
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
