"""Automated (code-based) response quality metrics.

These are fast, deterministic metrics that do not require an LLM call.
LLM-as-judge metrics (faithfulness, factual consistency, answer relevance,
completeness) are in eval/judges/. Citation grounding and decline detection
are computed in eval/run_eval.py.
"""

import re

from src.generation.citations import extract_citations

# Rough heuristic: a "factual claim" is a sentence that doesn't start with "Note",
# "Disclaimer", or "⚠️" and is longer than 20 chars.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _count_factual_sentences(text: str) -> int:
    sentences = _SENTENCE_SPLIT.split(text)
    return sum(
        1
        for s in sentences
        if len(s.strip()) > 20
        and not s.strip().startswith(("Note", "Disclaimer", "⚠️", "This is informational"))
    )


def citation_completeness(answer: str) -> float:
    """Citations per factual sentence, capped at 1.0 (a rough coverage proxy)."""
    factual_sentences = _count_factual_sentences(answer)
    if factual_sentences == 0:
        return 1.0
    return min(len(extract_citations(answer)) / factual_sentences, 1.0)
