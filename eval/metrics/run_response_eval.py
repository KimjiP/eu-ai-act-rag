"""CLI: run automated response evaluation against the golden dataset.

Usage:
    uv run python eval/metrics/run_response_eval.py
    uv run python eval/metrics/run_response_eval.py --split test
"""

import argparse
import logging
from pathlib import Path

from src import config
from eval.metrics.response import evaluate_responses

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run response quality evaluation")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
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
        help="Save per-query results JSON to this path",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=25.0,
        help="Seconds to wait between LLM calls to avoid rate limits (default: 25.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Chunks to retrieve per query — fewer = smaller prompt = fewer rate limit hits (default: 3)",
    )
    args = parser.parse_args()

    output_path = args.output or (
        config.EXPERIMENTS_DIR / f"response_{args.split}_{config.EXPERIMENT_ID}.json"
    )

    evaluate_responses(
        dataset_path=args.dataset,
        split=args.split,
        output_path=output_path,
        delay_seconds=args.delay,
        top_k=args.top_k,
    )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
