"""Golden dataset generation script.

Feeds chunks (grouped by article) to Claude Sonnet to generate candidate
query-answer pairs across all taxonomy dimensions (type × difficulty × persona).

Run ONCE, then manually review candidates_raw.json and curate golden_dataset.json.

Usage:
    uv run python eval/golden_dataset/generate.py
    uv run python eval/golden_dataset/generate.py --max-articles 10  # quick test
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

import anthropic

from src import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent
CANDIDATES_PATH = OUTPUT_DIR / "candidates_raw.json"

GENERATION_PROMPT = """You are helping build a golden evaluation dataset for an EU AI Act Q&A system.

Below is a passage from the EU AI Act regulatory text. Generate 3-5 evaluation queries that test whether a RAG system can correctly retrieve and answer questions about this passage.

For each query, vary across these dimensions:
- Query type: factual (exact provision), conceptual (regulatory intent), procedural (compliance steps), cross_reference (links to other articles)
- Difficulty: easy (single clear article), medium (2-3 provisions needed), hard (multi-hop or tricky phrasing)
- Persona: compliance_officer (formal, obligation-focused), product_manager (practical, impact-focused), general (informal)

Source passage:
Article/Section: {article_number}
Text: {text}

Respond with ONLY a JSON array. Each item must have these exact fields:
[
  {{
    "query": "<the question a user would ask>",
    "query_type": "factual|conceptual|procedural|cross_reference",
    "difficulty": "easy|medium|hard",
    "persona": "compliance_officer|product_manager|general",
    "expected_answer_elements": ["<key fact 1>", "<key fact 2>"],
    "acceptable_citation_targets": ["Article X(Y)", "Annex Z"],
    "is_unanswerable": false,
    "source_article": "{article_number}"
  }}
]

Generate queries that are genuinely useful for compliance research. Include at least one hard query per passage.
"""


def _parse_json_tolerant(raw: str, article_number: str) -> list | None:
    """Try standard JSON parse first, then fall back to json5-style fixes.

    Handles the most common LLM JSON error: unescaped apostrophes/quotes
    inside string values that come from legal text (e.g. "provider's obligation").
    """
    # Pass 1: standard parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Pass 2: replace curly/smart quotes with straight quotes and retry
    cleaned = raw.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = cleaned.replace("\u201c", '\\"').replace("\u201d", '\\"')
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Pass 3: use a regex to extract individual JSON objects and parse each one
    objects = re.findall(r"\{[^{}]+\}", raw, re.DOTALL)
    candidates = []
    for obj in objects:
        try:
            candidates.append(json.loads(obj))
        except json.JSONDecodeError:
            continue

    if candidates:
        logger.warning(
            f"Used object-level fallback parser for {article_number}: "
            f"recovered {len(candidates)} of {len(objects)} objects."
        )
        return candidates

    logger.warning(f"Failed for {article_number}: could not parse JSON after all fallbacks.")
    return None


def generate_candidates(
    chunks_path: Path = config.CHUNKS_JSON_PATH,
    max_articles: int | None = None,
    seed: int = 42,
    only_articles: list[str] | None = None,
) -> list[dict]:
    """Generate candidate queries from chunks and save to candidates_raw.json.

    Args:
        only_articles: if provided, only generate for these article numbers
                       e.g. ["Article 5", "Article 6", "Article 7"]
    """
    with open(chunks_path) as f:
        chunks = json.load(f)

    # Group chunks by article_number, keep only article chunks (not recitals/annexes for now)
    article_chunks = [
        c for c in chunks
        if (c.get("article_number") or "").startswith("Article")
        and len(c.get("text", "")) > 200
    ]

    if only_articles:
        article_chunks = [c for c in article_chunks if c.get("article_number") in only_articles]
    elif max_articles:
        random.seed(seed)
        article_chunks = random.sample(article_chunks, min(max_articles, len(article_chunks)))

    logger.info(f"Generating queries from {len(article_chunks)} article chunks ...")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    all_candidates: list[dict] = []
    failed = 0

    for i, chunk in enumerate(article_chunks):
        article_number = chunk.get("article_number", "Unknown")
        text = chunk.get("text", "")[:2000]  # cap at 2000 chars to stay within context

        prompt = GENERATION_PROMPT.format(article_number=article_number, text=text)

        try:
            response = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=1024,
                temperature=0.7,  # some diversity in query generation
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Extract JSON array from response
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                logger.warning(f"No JSON array found for {article_number}")
                failed += 1
                continue

            candidates = _parse_json_tolerant(match.group(0), article_number)
            if candidates is None:
                failed += 1
                continue
            # Add a running query_id placeholder (will be overwritten during curation)
            for j, c in enumerate(candidates):
                c["query_id"] = f"auto_{i:03d}_{j:02d}"
                c["split"] = "dev"  # default; adjust during manual review
                c["relevant_chunk_ids"] = []  # must be filled during manual review
                c["reference_answer"] = ""  # optional; fill during manual review
                c["notes"] = ""

            all_candidates.extend(candidates)
            logger.info(
                f"  [{i+1}/{len(article_chunks)}] {article_number}: "
                f"{len(candidates)} candidates generated."
            )

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed for {article_number}: {e}")
            failed += 1

    logger.info(
        f"Generated {len(all_candidates)} candidates from {len(article_chunks)} articles "
        f"({failed} failures)."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # If existing candidates file exists, merge (append new, deduplicate by query text)
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH) as f:
            existing = json.load(f)
        existing_queries = {c["query"] for c in existing}
        new_only = [c for c in all_candidates if c["query"] not in existing_queries]
        all_candidates = existing + new_only
        logger.info(f"Merged with existing file: {len(new_only)} new candidates added.")

    with open(CANDIDATES_PATH, "w") as f:
        json.dump(all_candidates, f, indent=2)

    logger.info(f"Saved {len(all_candidates)} candidates to {CANDIDATES_PATH}")
    logger.info(
        "Next step: manually review candidates_raw.json, fill in relevant_chunk_ids, "
        "assign final query_ids, set splits (70% dev / 30% test), "
        "then save as golden_dataset.json"
    )

    return all_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden dataset candidates")
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Limit to N articles (for quick testing)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--articles",
        nargs="+",
        default=None,
        help='Retry specific articles only, e.g. --articles "Article 5" "Article 6"',
    )
    args = parser.parse_args()

    only_articles = args.articles  # e.g. ["Article 5", "Article 6"]
    generate_candidates(
        max_articles=args.max_articles,
        seed=args.seed,
        only_articles=only_articles,
    )


if __name__ == "__main__":
    main()
