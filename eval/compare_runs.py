"""Print a markdown table comparing saved evaluation runs on one split.

Usage:
    uv run python -m eval.compare_runs --split test v1-original v2-parser-only v2-rerank20 v2-final
"""

import argparse
import json

from src import config

ROWS = [
    ("Recall@1", "retrieval", "recall_at_1"),
    ("Recall@3", "retrieval", "recall_at_3"),
    ("Recall@5", "retrieval", "recall_at_5"),
    ("MRR", "retrieval", "mrr"),
    ("Citations: cited text was retrieved", "answers", "citations_grounded"),
    ("Citations: label of a passage whose text is another provision", "answers", "citations_wrong_label"),
    ("Citations: cross-reference named in retrieved text", "answers", "citations_cross_reference"),
    ("Citations: unsupported", "answers", "citations_unsupported"),
    ("Unanswerable questions declined", "answers", "unanswerable_declined"),
    ("Answerable questions declined", "answers", "answerable_declined"),
    ("Completeness vs the Act (judge)", "judges", "completeness"),
    ("Faithfulness to retrieved text (judge)", "judges", "faithfulness"),
    ("Factual consistency (judge)", "judges", "factual_consistency"),
    ("Answer relevance (judge)", "judges", "answer_relevance"),
    ("Avg characters sent to the LLM", "answers", "avg_context_chars"),
    ("Avg cost per question (USD)", "answers", "avg_cost_usd"),
]


def format_value(key: str, value) -> str:
    if value is None:
        return "not run"
    if key == "avg_cost_usd":
        return f"${value:.4f}"
    if value > 100:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved evaluation runs")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()

    summaries = []
    for tag in args.tags:
        with open(config.EXPERIMENTS_DIR / f"eval_{tag}_{args.split}.json") as f:
            summaries.append(json.load(f)["summary"])

    first = summaries[0]
    print(
        f"{args.split} split: {first['n_answerable']} answerable, "
        f"{first['n_unanswerable']} unanswerable questions\n"
    )
    print("| Metric | " + " | ".join(args.tags) + " |")
    print("|---|" + "---|" * len(args.tags))
    for label, section, key in ROWS:
        values = [format_value(key, s.get(section, {}).get(key)) for s in summaries]
        print(f"| {label} | " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
