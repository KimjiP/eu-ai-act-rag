"""Run the evaluation on one split of the golden dataset: retrieval, answers and LLM judges.

Each question is answered once, every metric is computed on that same answer, and
the full record (retrieved passages, answer, citation check, judge verdicts) is
saved so the eval viewer can show why a question passed or failed.

Ground truth never comes from chunk labels. Every passage is mapped to the
provisions its text actually contains (eval/provisions.py), so the same
evaluation can score the original corpus and the corrected one.

Usage:
    uv run python -m eval.run_eval --split test --tag v2
    RAG_CORPUS_VERSION=v1 RAG_RERANK_CANDIDATES=0 uv run python -m eval.run_eval --split test --tag v1-original
    uv run python -m eval.run_eval --split dev --no-judges
"""

import argparse
import json
import logging
import re
import statistics
from datetime import UTC, datetime

from src import config
from src.api.query import answer_query
from src.generation.citations import parse_reference
from src.retrieval import search
from eval.judges.llm_judge import JUDGE_MODEL, run_all_judges
from eval.metrics.response import citation_completeness
from eval.metrics.retrieval import compute_mrr, compute_precision_at_k, compute_recall_at_k
from eval.provisions import contains_any, get_index, refers_to

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

RETRIEVAL_K = 5  # ranks scored for Recall@1/3/5 and MRR
_CONTAINS_SHOWN = 8  # provisions listed per passage in the saved record


def citation_status(citation: str, shown: list[str], context_text: str) -> str:
    """How a citation relates to the passages the model was given.

    retrieved        the cited provision's own text was among the passages
    cross_reference  its text was not, but the passages refer to it by number
                     (e.g. Article 9 mentions "the post-market monitoring system
                     referred to in Article 72")
    unsupported      neither
    """
    if any(refers_to(citation, p) for p in shown):
        return "retrieved"
    ref = parse_reference(citation)
    if ref is not None:
        word = {"article": "Article", "annex": "Annex", "recital": "Recital"}[ref.kind]
        if re.search(rf"\b{word}\s+{ref.number}(?![\dIVXLC])", context_text, re.IGNORECASE):
            return "cross_reference"
    return "unsupported"


def _passage(chunk, provisions: list[str], gold: list[str]) -> dict:
    return {
        "rank": chunk.rank + 1,
        "label": chunk.article_number,
        "contains": provisions[:_CONTAINS_SHOWN],
        "contains_count": len(provisions),
        "relevant": contains_any(provisions, gold),
        "chars": len(chunk.text),
    }


def evaluate_question(item: dict, top_k: int, judges: bool) -> dict:
    index = get_index()
    gold = item["gold_provisions"]
    record = {
        "query_id": item["query_id"],
        "query": item["query"],
        "is_unanswerable": item["is_unanswerable"],
        "gold_provisions": gold,
    }

    # Retrieval, scored on the top RETRIEVAL_K passages
    ranked = search(item["query"], top_k=RETRIEVAL_K)
    ranked_provisions = [index.provisions_in(chunk.text) for chunk in ranked]
    record["retrieved"] = [_passage(c, p, gold) for c, p in zip(ranked, ranked_provisions)]
    if gold:
        hits = [contains_any(provisions, gold) for provisions in ranked_provisions]
        record["retrieval"] = {
            "recall_at_1": compute_recall_at_k(hits, 1),
            "recall_at_3": compute_recall_at_k(hits, 3),
            "recall_at_5": compute_recall_at_k(hits, 5),
            "mrr": compute_mrr(hits),
            "precision_at_5": compute_precision_at_k(hits, 5),
        }

    # Answer, generated from the top_k passages the pipeline passes to the LLM
    response = answer_query(item["query"], top_k=top_k)
    answer = response.answer or ""
    context = [(c, index.provisions_in(c.text)) for c in response.retrieved_chunks]
    shown = [p for _, provisions in context for p in provisions]
    context_text = "\n".join(c.text for c in response.retrieved_chunks)
    citations = []
    for citation in response.citations:
        status = citation_status(citation, shown, context_text)
        citations.append({"citation": citation, "status": status, "grounded": status == "retrieved"})
    record["answer"] = {
        "text": answer,
        "declined": not response.answered,
        "context": [_passage(c, p, gold) for c, p in context],
        "context_chars": sum(len(c.text) for c, _ in context),
        "citations": citations,
        "citation_completeness": citation_completeness(answer) if response.answered else None,
        "bad_framing": response.bad_framing_detected,
        "cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
    }

    # Judges: answerable questions the system answered
    if judges and not item["is_unanswerable"] and response.answered:
        verdicts = run_all_judges(
            item["query"], answer, response.retrieved_chunks, item["expected_answer_elements"]
        )
        record["judges"] = {
            name: {"score": v.score, "reasoning": v.reasoning, **v.extra}
            for name, v in verdicts.items()
        }
    return record


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def summarize(records: list[dict]) -> dict:
    answerable = [r for r in records if not r["is_unanswerable"]]
    unanswerable = [r for r in records if r["is_unanswerable"]]
    answered = [r for r in answerable if not r["answer"]["declined"]]
    all_citations = [c for r in records if not r["answer"]["declined"] for c in r["answer"]["citations"]]
    judged = [r for r in answerable if "judges" in r]

    summary = {
        "n_questions": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "retrieval": {
            metric: _mean([r["retrieval"][metric] for r in answerable])
            for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "precision_at_5")
        },
        "answers": {
            "unanswerable_declined": _mean([float(r["answer"]["declined"]) for r in unanswerable]),
            "answerable_declined": _mean([float(r["answer"]["declined"]) for r in answerable]),
            "citations_grounded": _mean([float(c["grounded"]) for c in all_citations]),
            "citations_cross_reference": _mean(
                [float(c["status"] == "cross_reference") for c in all_citations]
            ),
            "citations_unsupported": _mean([float(c["status"] == "unsupported") for c in all_citations]),
            "n_citations": len(all_citations),
            "citation_completeness": _mean([r["answer"]["citation_completeness"] for r in answered]),
            "bad_framing_rate": _mean([float(r["answer"]["bad_framing"]) for r in records]),
            "avg_context_chars": _mean([r["answer"]["context_chars"] for r in records]),
            "avg_cost_usd": round(statistics.mean(r["answer"]["cost_usd"] for r in records), 5),
            "avg_latency_ms": round(statistics.mean(r["answer"]["latency_ms"] for r in records)),
        },
    }
    if judged:
        summary["judges"] = {
            name: _mean([r["judges"][name]["score"] for r in judged if name in r["judges"]])
            for name in ("completeness", "faithfulness", "factual_consistency", "answer_relevance")
        }
        summary["judges"]["n_judged"] = len(judged)
    return summary


def _print_summary(tag: str, split: str, s: dict) -> None:
    r, a = s["retrieval"], s["answers"]
    print(f"\n{'=' * 64}\n{tag} — split={split}: {s['n_answerable']} answerable, "
          f"{s['n_unanswerable']} unanswerable\n{'=' * 64}")
    print(f"  Recall@1 / @3 / @5       {r['recall_at_1']} / {r['recall_at_3']} / {r['recall_at_5']}")
    print(f"  MRR / Precision@5        {r['mrr']} / {r['precision_at_5']}")
    print(f"  Citations: text retrieved {a['citations_grounded']}, cross-reference "
          f"{a['citations_cross_reference']}, unsupported {a['citations_unsupported']}  (n={a['n_citations']})")
    print(f"  Declined, unanswerable   {a['unanswerable_declined']}  (should be 1.0)")
    print(f"  Declined, answerable     {a['answerable_declined']}  (should be 0.0)")
    print(f"  Bad framing rate         {a['bad_framing_rate']}")
    print(f"  Avg context / cost       {a['avg_context_chars']:.0f} chars / ${a['avg_cost_usd']}")
    for name, value in s.get("judges", {}).items():
        print(f"  Judge: {name:<18}{value}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation on one split")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--tag", required=True, help="Name for this run, used in the output file")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_RETRIEVAL)
    parser.add_argument("--no-judges", action="store_true", help="Skip the LLM judges")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    args = parser.parse_args()

    config.EXPERIMENT_ID = args.tag  # recorded in the query log
    with open(config.GOLDEN_DATASET_PATH) as f:
        questions = [q for q in json.load(f) if q["split"] == args.split][: args.limit]

    logger.info(
        f"Evaluating {len(questions)} '{args.split}' questions: corpus {config.CORPUS_VERSION}, "
        f"rerank candidates {config.RERANK_CANDIDATES}, recital penalty {config.RECITAL_PENALTY}, "
        f"top_k {args.top_k}"
    )
    records = []
    for i, item in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {item['query_id']}: {item['query'][:70]}")
        records.append(evaluate_question(item, top_k=args.top_k, judges=not args.no_judges))

    summary = summarize(records)
    _print_summary(args.tag, args.split, summary)

    output = config.EXPERIMENTS_DIR / f"eval_{args.tag}_{args.split}.json"
    with open(output, "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "split": args.split,
                "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "config": {
                    "corpus_version": config.CORPUS_VERSION,
                    "rerank_candidates": config.RERANK_CANDIDATES,
                    "recital_penalty": config.RECITAL_PENALTY,
                    "top_k": args.top_k,
                    "llm_model": config.LLM_MODEL,
                    "prompt_version": config.PROMPT_VERSION,
                    "embedding_model": config.EMBEDDING_MODEL,
                    "reranker_model": config.RERANKER_MODEL if config.RERANKER_ENABLED else None,
                    "judge_model": None if args.no_judges else JUDGE_MODEL,
                    "golden_dataset": config.GOLDEN_DATASET_PATH.name,
                },
                "summary": summary,
                "per_query": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
