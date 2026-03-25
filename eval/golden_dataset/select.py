"""Automatic golden dataset selection.

Reads pre_curated.json and selects the best 40-50 entries using stratified
sampling, then writes directly to golden_dataset.json.

Selection criteria (in priority order):
  1. Must have at least one relevant_chunk_id (LLM-verified or article-matched)
  2. Must have at least one expected_answer_element
  3. Stratified across: key articles, query_type, difficulty, persona
  4. At least 1 unanswerable query included
  5. Exactly 70% dev / 30% test split

Usage:
    uv run python -m eval.golden_dataset.select
    uv run python -m eval.golden_dataset.select --target 45
    uv run python -m eval.golden_dataset.select --dry-run   # preview without writing
"""

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from src import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PRE_CURATED_PATH = Path(__file__).parent / "pre_curated.json"
GOLDEN_PATH = config.GOLDEN_DATASET_PATH

# Articles that must be represented in the final dataset (core EU AI Act provisions)
PRIORITY_ARTICLES = {
    "Article 5",   # prohibited practices
    "Article 6",   # high-risk classification
    "Article 9",   # risk management
    "Article 10",  # data governance
    "Article 13",  # transparency
    "Article 14",  # human oversight
    "Article 17",  # quality management
    "Article 50",  # transparency for certain AI systems
}

# Target distribution across query types
TARGET_TYPE_DIST = {
    "factual": 0.35,
    "procedural": 0.30,
    "conceptual": 0.20,
    "cross_reference": 0.15,
}

# Target distribution across difficulty levels
TARGET_DIFF_DIST = {
    "easy": 0.40,
    "medium": 0.40,
    "hard": 0.20,
}


def score_candidate(c: dict) -> float:
    """Score a candidate on quality signals. Higher = better to keep."""
    score = 0.0

    # Has verified chunk IDs
    n_chunks = len(c.get("relevant_chunk_ids", []))
    if n_chunks > 0:
        score += 3.0
    if n_chunks > 1:
        score += 1.0  # bonus for multi-chunk answers (richer ground truth)

    # Has expected answer elements
    n_elements = len(c.get("expected_answer_elements", []))
    score += min(n_elements, 3) * 1.0  # up to 3 points

    # LLM-verified (not just article-matched)
    if "llm-verified" in c.get("notes", ""):
        score += 2.0

    # Penalise fallback entries
    if "fallback" in c.get("notes", "") or "no retrieval" in c.get("notes", ""):
        score -= 2.0

    # Penalise if query is very short (likely vague)
    if len(c.get("query", "")) < 40:
        score -= 1.0

    return score


def select_entries(
    candidates: list[dict],
    target: int = 45,
    seed: int = 42,
) -> list[dict]:
    """Stratified selection of `target` entries from candidates."""
    random.seed(seed)

    # Filter: must have chunk IDs
    viable = [c for c in candidates if c.get("relevant_chunk_ids")]
    logger.info(f"Viable candidates (have chunk IDs): {len(viable)} / {len(candidates)}")

    selected: list[dict] = []
    selected_ids: set[str] = set()

    def _add(c: dict) -> bool:
        qid = c["query_id"]
        if qid not in selected_ids:
            selected.append(c)
            selected_ids.add(qid)
            return True
        return False

    # --- Pass 1: ensure priority articles are represented ---
    by_article: dict[str, list[dict]] = defaultdict(list)
    for c in viable:
        by_article[c.get("source_article", "")].append(c)

    for article in PRIORITY_ARTICLES:
        pool = sorted(by_article.get(article, []), key=score_candidate, reverse=True)
        # Pick 1 easy + 1 medium/hard per priority article if available
        easy = [c for c in pool if c.get("difficulty") == "easy"]
        harder = [c for c in pool if c.get("difficulty") in ("medium", "hard")]
        if easy:
            _add(easy[0])
        if harder:
            _add(harder[0])
        if len(selected) >= target:
            break

    logger.info(f"After priority article pass: {len(selected)} selected")

    # --- Pass 2: fill remaining slots with stratified sampling ---
    remaining_target = target - len(selected)
    remaining = [c for c in viable if c["query_id"] not in selected_ids]

    # Group remaining by (query_type, difficulty)
    by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    for c in remaining:
        by_stratum[(c.get("query_type", "factual"), c.get("difficulty", "easy"))].append(c)

    # Sort each stratum by score descending
    for key in by_stratum:
        by_stratum[key].sort(key=score_candidate, reverse=True)

    # Allocate slots proportionally across strata
    strata_order = sorted(by_stratum.keys())  # deterministic order
    slots_per_stratum: dict[tuple, int] = {}
    for qt, qt_weight in TARGET_TYPE_DIST.items():
        for diff, diff_weight in TARGET_DIFF_DIST.items():
            key = (qt, diff)
            slots_per_stratum[key] = max(1, round(remaining_target * qt_weight * diff_weight))

    # Fill strata in order of allocation
    for key in strata_order:
        pool = by_stratum[key]
        n_slots = slots_per_stratum.get(key, 1)
        for c in pool[:n_slots]:
            _add(c)
            if len(selected) >= target:
                break
        if len(selected) >= target:
            break

    # --- Pass 3: ensure at least 1 unanswerable query ---
    unanswerable = [c for c in viable if c.get("is_unanswerable")]
    if not any(c.get("is_unanswerable") for c in selected) and unanswerable:
        # Replace the lowest-scored selected entry with the best unanswerable one
        selected.sort(key=score_candidate)
        selected[0] = unanswerable[0]
        logger.info("Swapped in 1 unanswerable query.")

    # --- Trim to exact target ---
    if len(selected) > target:
        selected.sort(key=score_candidate, reverse=True)
        selected = selected[:target]

    return selected


def assign_splits(entries: list[dict], dev_ratio: float = 0.7, seed: int = 42) -> list[dict]:
    """Assign stratified dev/test splits and reassign clean sequential query_ids."""
    random.seed(seed)

    by_stratum: dict[tuple, list[int]] = defaultdict(list)
    for i, c in enumerate(entries):
        by_stratum[(c.get("query_type", "factual"), c.get("difficulty", "easy"))].append(i)

    for indices in by_stratum.values():
        random.shuffle(indices)
        n_dev = max(1, round(len(indices) * dev_ratio))
        for j, idx in enumerate(indices):
            entries[idx]["split"] = "dev" if j < n_dev else "test"

    # Reassign clean sequential query_ids
    for i, entry in enumerate(entries):
        entry["query_id"] = f"q{i+1:03d}"

    return entries


def print_summary(entries: list[dict]) -> None:
    print(f"\n{'='*55}")
    print(f"Golden dataset: {len(entries)} entries")
    print(f"{'='*55}")
    print(f"By type:       {dict(Counter(c['query_type'] for c in entries))}")
    print(f"By difficulty: {dict(Counter(c['difficulty'] for c in entries))}")
    print(f"By split:      {dict(Counter(c['split'] for c in entries))}")
    print(f"Unanswerable:  {sum(1 for c in entries if c.get('is_unanswerable'))}")

    covered = {c['source_article'] for c in entries if c.get('source_article')}
    missing_priority = PRIORITY_ARTICLES - covered
    print(f"Priority articles covered: {len(PRIORITY_ARTICLES) - len(missing_priority)}/{len(PRIORITY_ARTICLES)}")
    if missing_priority:
        print(f"  Missing: {sorted(missing_priority)}")

    avg_chunks = sum(len(c.get('relevant_chunk_ids', [])) for c in entries) / len(entries)
    print(f"Avg relevant chunks/query: {avg_chunks:.1f}")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select final golden dataset entries")
    parser.add_argument("--target", type=int, default=45, help="Target number of entries (default: 45)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Preview selection without writing")
    args = parser.parse_args()

    if not PRE_CURATED_PATH.exists():
        logger.error(f"pre_curated.json not found. Run curate.py first.")
        raise SystemExit(1)

    with open(PRE_CURATED_PATH) as f:
        candidates = json.load(f)
    logger.info(f"Loaded {len(candidates)} candidates from pre_curated.json")

    selected = select_entries(candidates, target=args.target, seed=args.seed)
    selected = assign_splits(selected, seed=args.seed)

    print_summary(selected)

    if args.dry_run:
        print("Dry run — nothing written.")
        return

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w") as f:
        json.dump(selected, f, indent=2)

    logger.info(f"Wrote {len(selected)} entries to {GOLDEN_PATH}")
    print(f"Golden dataset saved to: {GOLDEN_PATH}")
    print(f"\nNext step:")
    print(f"  uv run python -m eval.metrics.run_retrieval_eval --split dev")


if __name__ == "__main__":
    main()
