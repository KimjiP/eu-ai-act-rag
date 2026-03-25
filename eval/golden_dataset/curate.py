"""Golden dataset curation helper.

Two modes:

  Basic (fast, no API calls):
    uv run python -m eval.golden_dataset.curate
    Matches chunk IDs by article number, deduplicates, assigns splits.
    Output: pre_curated.json — requires manual verification of relevant_chunk_ids.

  Auto (uses LLM + retrieval, recommended):
    uv run python -m eval.golden_dataset.curate --auto
    For each candidate query, retrieves top-k chunks, asks Claude to judge
    which are genuinely relevant, generates expected_answer_elements, and
    marks unanswerable queries. Costs ~$0.10-0.20 for a full run.
    Output: pre_curated.json — spot-check a sample, then save as golden_dataset.json.

After running either mode:
  1. Open eval/golden_dataset/pre_curated.json
  2. Trim to 40-50 entries (delete low-quality or duplicate queries)
  3. Save as eval/golden_dataset/golden_dataset.json
"""

import argparse
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import anthropic

from src import config
from src.retrieval import search

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CANDIDATES_PATH = Path(__file__).parent / "candidates_raw.json"
PRE_CURATED_PATH = Path(__file__).parent / "pre_curated.json"
GOLDEN_PATH = config.GOLDEN_DATASET_PATH

# LLM judge for relevance verification — use Haiku to keep costs low
_JUDGE_MODEL = "claude-haiku-4-5-20251001"

_RELEVANCE_JUDGE_PROMPT = """You are building a ground-truth evaluation dataset for an EU AI Act Q&A system.

Given a query and a set of retrieved passages, determine:
1. Which passage indices (0-based) are RELEVANT — i.e., contain information needed to answer the query
2. Whether the query is UNANSWERABLE from these passages alone
3. The key factual elements a correct answer must include (2-4 bullet points)

Passages:
{passages}

Query: {query}

Respond with ONLY this JSON (no other text):
{{
  "relevant_indices": [<list of 0-based indices>],
  "is_unanswerable": <true|false>,
  "expected_answer_elements": ["<element 1>", "<element 2>", "<element 3>"]
}}"""


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def build_article_index(chunks: list[dict]) -> dict[str, list[str]]:
    """Map article_number → list of chunk_ids."""
    index: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        article = chunk.get("article_number")
        if article:
            index[article].append(chunk["chunk_id"])
    return dict(index)


def deduplicate(candidates: list[dict]) -> list[dict]:
    """Remove near-duplicate queries by normalised query text."""
    seen: set[str] = set()
    unique = []
    for c in candidates:
        key = c["query"].strip().lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def assign_splits(candidates: list[dict], dev_ratio: float = 0.7, seed: int = 42) -> list[dict]:
    """Assign dev/test splits stratified by query_type and difficulty."""
    import random
    random.seed(seed)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, c in enumerate(candidates):
        groups[(c["query_type"], c["difficulty"])].append(i)
    for indices in groups.values():
        random.shuffle(indices)
        n_dev = max(1, round(len(indices) * dev_ratio))
        for j, idx in enumerate(indices):
            candidates[idx]["split"] = "dev" if j < n_dev else "test"
    return candidates


def print_stats(candidates: list[dict]) -> None:
    print(f"\n{'='*55}")
    print(f"Candidate stats — {len(candidates)} total")
    print(f"{'='*55}")
    print(f"By query type:   {dict(Counter(c['query_type'] for c in candidates))}")
    print(f"By difficulty:   {dict(Counter(c['difficulty'] for c in candidates))}")
    print(f"By split:        {dict(Counter(c['split'] for c in candidates))}")
    no_chunks = [c for c in candidates if not c["relevant_chunk_ids"]]
    unanswerable = [c for c in candidates if c.get("is_unanswerable")]
    print(f"Unanswerable:    {len(unanswerable)}")
    print(f"No chunk match:  {len(no_chunks)}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Basic mode (no LLM calls)
# ---------------------------------------------------------------------------

def run_basic(candidates: list[dict], chunks: list[dict]) -> list[dict]:
    """Populate relevant_chunk_ids by article-number matching only."""
    article_index = build_article_index(chunks)
    logger.info(f"Article index: {len(article_index)} unique article numbers")

    populated = []
    for i, c in enumerate(candidates):
        source_article = c.get("source_article", "")
        populated.append({
            "query_id": f"q{i+1:03d}",
            "query": c.get("query", ""),
            "query_type": c.get("query_type", "factual"),
            "difficulty": c.get("difficulty", "easy"),
            "persona": c.get("persona", "compliance_officer"),
            "split": "dev",
            "relevant_chunk_ids": article_index.get(source_article, []),
            "partially_relevant_chunk_ids": [],
            "expected_answer_elements": c.get("expected_answer_elements", []),
            "acceptable_citation_targets": c.get("acceptable_citation_targets", []),
            "is_unanswerable": c.get("is_unanswerable", False),
            "reference_answer": "",
            "notes": "⚠️ chunk IDs auto-matched by article only — verify manually",
            "source_article": source_article,
        })
    return populated


# ---------------------------------------------------------------------------
# Auto mode (LLM-verified)
# ---------------------------------------------------------------------------

def _call_judge(client: anthropic.Anthropic, prompt: str) -> dict | None:
    """Call the judge LLM and parse JSON response."""
    try:
        response = client.messages.create(
            model=_JUDGE_MODEL,
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
    return None


def _format_passages(results) -> str:
    return "\n\n".join(
        f"[{i}] {r.article_number or '(unlabelled)'}\n{r.text[:600]}"
        for i, r in enumerate(results)
    )


def run_auto(
    candidates: list[dict],
    chunks: list[dict],
    top_k: int = 5,
    max_candidates: int | None = None,
) -> list[dict]:
    """Use LLM + retrieval to verify chunk relevance and generate answer elements."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    if max_candidates:
        candidates = candidates[:max_candidates]

    populated = []
    total_cost = 0.0
    # Haiku pricing: $0.25/1M input, $1.25/1M output (approximate)
    haiku_input_price = 0.25 / 1_000_000
    haiku_output_price = 1.25 / 1_000_000

    logger.info(f"Auto-curating {len(candidates)} candidates with LLM judge ...")

    for i, c in enumerate(candidates):
        query = c.get("query", "")
        source_article = c.get("source_article", "")

        # Retrieve top-k chunks for this query
        results = search(query, top_k=top_k)

        if not results:
            logger.warning(f"  [{i+1}] No retrieval results for: {query[:60]!r}")
            populated.append(_make_entry(i, c, [], True, c.get("expected_answer_elements", []),
                                         "no retrieval results"))
            continue

        # Ask LLM to judge relevance
        passages_text = _format_passages(results)
        prompt = _RELEVANCE_JUDGE_PROMPT.format(passages=passages_text, query=query)
        judgment = _call_judge(client, prompt)

        if judgment is None:
            # Fall back to article-number matching
            relevant_ids = [r.chunk_id for r in results if r.article_number == source_article]
            populated.append(_make_entry(i, c, relevant_ids, False,
                                         c.get("expected_answer_elements", []),
                                         "judge failed — article-match fallback"))
            continue

        relevant_indices = judgment.get("relevant_indices", [])
        is_unanswerable = judgment.get("is_unanswerable", False)
        answer_elements = judgment.get("expected_answer_elements", [])

        relevant_ids = [
            results[idx].chunk_id
            for idx in relevant_indices
            if idx < len(results)
        ]

        # Estimate cost (rough: prompt ~800 tokens, response ~200 tokens)
        total_cost += 800 * haiku_input_price + 200 * haiku_output_price

        logger.info(
            f"  [{i+1}/{len(candidates)}] {source_article}: "
            f"{len(relevant_ids)} relevant chunk(s), "
            f"unanswerable={is_unanswerable}"
        )

        populated.append(_make_entry(
            i, c, relevant_ids, is_unanswerable, answer_elements,
            note="llm-verified" if relevant_ids else "llm-verified: no relevant chunks found",
        ))

        # Brief pause to avoid rate limits
        if (i + 1) % 20 == 0:
            time.sleep(1)

    logger.info(f"Auto-curation complete. Estimated cost: ${total_cost:.4f}")
    return populated


def _make_entry(
    i: int,
    c: dict,
    relevant_ids: list[str],
    is_unanswerable: bool,
    answer_elements: list[str],
    note: str = "",
) -> dict:
    return {
        "query_id": f"q{i+1:03d}",
        "query": c.get("query", ""),
        "query_type": c.get("query_type", "factual"),
        "difficulty": c.get("difficulty", "easy"),
        "persona": c.get("persona", "compliance_officer"),
        "split": "dev",
        "relevant_chunk_ids": relevant_ids,
        "partially_relevant_chunk_ids": [],
        "expected_answer_elements": answer_elements,
        "acceptable_citation_targets": c.get("acceptable_citation_targets", []),
        "is_unanswerable": is_unanswerable,
        "reference_answer": "",
        "notes": note,
        "source_article": c.get("source_article", ""),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Curate golden dataset candidates")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use LLM + retrieval to verify chunk relevance (recommended, costs ~$0.10-0.20)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Chunks to retrieve per query in auto mode (default: 5)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Limit to first N candidates (useful for a quick test run)",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Print stats about existing pre_curated.json and exit",
    )
    args = parser.parse_args()

    # Stats-only mode
    if args.show_stats:
        if PRE_CURATED_PATH.exists():
            with open(PRE_CURATED_PATH) as f:
                print_stats(json.load(f))
        else:
            logger.error("pre_curated.json not found. Run without --show-stats first.")
        return

    # Load candidates
    if not CANDIDATES_PATH.exists():
        logger.error(f"candidates_raw.json not found at {CANDIDATES_PATH}")
        raise SystemExit(1)

    with open(CANDIDATES_PATH) as f:
        candidates = json.load(f)
    logger.info(f"Loaded {len(candidates)} candidates")

    with open(config.CHUNKS_JSON_PATH) as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")

    # Deduplicate first
    before = len(candidates)
    candidates = deduplicate(candidates)
    logger.info(f"Deduplicated: {before} → {len(candidates)} candidates")

    if args.max:
        candidates = candidates[: args.max]
        logger.info(f"Limited to first {args.max} candidates")

    # Run chosen mode
    if args.auto:
        populated = run_auto(candidates, chunks, top_k=args.top_k)
    else:
        populated = run_basic(candidates, chunks)

    # Assign stratified splits
    populated = assign_splits(populated)

    print_stats(populated)

    # Write output
    with open(PRE_CURATED_PATH, "w") as f:
        json.dump(populated, f, indent=2)
    logger.info(f"Wrote {len(populated)} entries to {PRE_CURATED_PATH}")

    print(f"\nNext steps:")
    print(f"  1. Open:  {PRE_CURATED_PATH}")
    print(f"  2. Trim to 40-50 entries — keep diverse coverage across articles,")
    print(f"     query types, and difficulty levels")
    print(f"  3. Save as: {GOLDEN_PATH}")
    print(f"  4. Run:   uv run python -m eval.metrics.run_retrieval_eval --split dev")


if __name__ == "__main__":
    main()
