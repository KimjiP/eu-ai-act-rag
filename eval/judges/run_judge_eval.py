"""CLI: run LLM-as-judge evaluation against the golden dataset.

Usage:
    uv run python -m eval.judges.run_judge_eval
    uv run python -m eval.judges.run_judge_eval --split test
"""

import argparse
import json
import logging
import time
from pathlib import Path

from src import config
from src.api.query import answer_query
from eval.judges.llm_judge import run_all_judges

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluation")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.GOLDEN_DATASET_PATH,
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
        default=5.0,
        help="Seconds between queries (default: 5.0 — Haiku is cheaper/faster)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    output_path = args.output or (
        config.EXPERIMENTS_DIR / f"judge_{args.split}_{config.EXPERIMENT_ID}.json"
    )

    with open(args.dataset) as f:
        dataset = json.load(f)

    queries = [q for q in dataset if q.get("split") == args.split and not q.get("is_unanswerable")]
    logger.info(f"Running judge eval on {len(queries)} answerable '{args.split}' queries ...")

    per_query = []

    for i, item in enumerate(queries):
        query_id = item["query_id"]
        query = item["query"]
        logger.info(f"[{i+1}/{len(queries)}] {query_id}: {query[:60]}...")

        if i > 0 and args.delay > 0:
            time.sleep(args.delay)

        response = answer_query(query, top_k=args.top_k)

        if not response.answered or not response.answer:
            logger.warning(f"  {query_id} was not answered — skipping judges")
            continue

        judges = run_all_judges(
            query=query,
            answer=response.answer,
            chunks=response.retrieved_chunks,
        )

        result = {
            "query_id": query_id,
            "query": query,
            "answer_relevance": judges["answer_relevance"].score,
            "faithfulness": judges["faithfulness"].score,
            "factual_consistency": judges["factual_consistency"].score,
            "reasoning": {k: v.reasoning for k, v in judges.items()},
        }
        per_query.append(result)

        print(
            f"  answer_relevance={judges['answer_relevance'].score:.2f}  "
            f"faithfulness={judges['faithfulness'].score:.2f}  "
            f"factual_consistency={judges['factual_consistency'].score:.2f}"
        )

    n = len(per_query)
    if n == 0:
        print("No results to summarise.")
        return

    avg_relevance = sum(r["answer_relevance"] for r in per_query) / n
    avg_faithfulness = sum(r["faithfulness"] for r in per_query) / n
    avg_consistency = sum(r["factual_consistency"] for r in per_query) / n

    print(f"\n{'='*55}")
    print(f"LLM Judge Metrics — split={args.split}, n={n}")
    print(f"{'='*55}")
    print(f"  Answer relevance:    {avg_relevance:.3f}  (target ≥ 0.80)")
    print(f"  Faithfulness:        {avg_faithfulness:.3f}  (target ≥ 0.80)")
    print(f"  Factual consistency: {avg_consistency:.3f}  (target ≥ 0.80)")
    print(f"{'='*55}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "split": args.split,
                "metrics": {
                    "answer_relevance": avg_relevance,
                    "faithfulness": avg_faithfulness,
                    "factual_consistency": avg_consistency,
                },
                "per_query": per_query,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
