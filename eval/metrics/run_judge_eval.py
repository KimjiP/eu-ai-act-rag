"""CLI: run LLM-as-judge evaluation over a saved response results file.

Usage:
    uv run python eval/metrics/run_judge_eval.py --results experiments/response_dev_E0.json
    uv run python eval/metrics/run_judge_eval.py --split test
"""

import argparse
import json
import logging
from pathlib import Path

from src import config
from eval.judges.llm_judge import run_all_judges
from src.api.query import answer_query

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
    )
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    queries = [q for q in dataset if q.get("split") == args.split]
    logger.info(f"Running LLM judges on {len(queries)} '{args.split}' queries ...")

    results = []
    total_relevance = total_faithfulness = total_consistency = 0.0

    for item in queries:
        query = item["query"]
        response = answer_query(query)

        if not response.answered or not response.answer:
            logger.info(f"  Skipping unanswered query: {query[:60]!r}")
            continue

        judge_results = run_all_judges(
            query=query,
            answer=response.answer,
            chunks=response.retrieved_chunks,
        )

        relevance = judge_results["answer_relevance"].score
        faithfulness = judge_results["faithfulness"].score
        consistency = judge_results["factual_consistency"].score

        total_relevance += relevance
        total_faithfulness += faithfulness
        total_consistency += consistency

        results.append(
            {
                "query_id": item["query_id"],
                "query": query,
                "answer_relevance": relevance,
                "faithfulness": faithfulness,
                "factual_consistency": consistency,
                "reasoning": {
                    k: v.reasoning for k, v in judge_results.items()
                },
            }
        )
        logger.info(
            f"  [{item['query_id']}] relevance={relevance:.2f} "
            f"faithfulness={faithfulness:.2f} consistency={consistency:.2f}"
        )

    n = len(results)
    if n > 0:
        print(f"\n{'='*55}")
        print(f"LLM Judge Metrics — split={args.split}, n={n}")
        print(f"{'='*55}")
        print(f"  Answer relevance:     {total_relevance/n:.3f}  (target ≥ 0.85)")
        print(f"  Faithfulness:         {total_faithfulness/n:.3f}  (target ≥ 0.90)")
        print(f"  Factual consistency:  {total_consistency/n:.3f}  (target ≥ 0.90)")
        print(f"{'='*55}\n")

    output_path = args.output or (
        config.EXPERIMENTS_DIR / f"judge_{args.split}_{config.EXPERIMENT_ID}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"split": args.split, "n": n, "per_query": results}, f, indent=2)
    print(f"Judge results saved to {output_path}")


if __name__ == "__main__":
    main()
