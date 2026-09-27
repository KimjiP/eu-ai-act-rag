"""Retrieval metrics.

Each function takes `hits`: one boolean per ranked result, True if that result
contains a gold provision. Deciding what counts as a hit is the caller's job;
eval/run_eval.py maps each passage to the provisions its text contains.

Metrics:
    Recall@k     — is a relevant result in the top k?
    MRR          — reciprocal rank of the first relevant result
    Precision@k  — what fraction of the top k results are relevant?
"""


def compute_recall_at_k(hits: list[bool], k: int) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0."""
    return 1.0 if any(hits[:k]) else 0.0


def compute_mrr(hits: list[bool]) -> float:
    """Reciprocal rank of the first relevant result (0.0 if none found)."""
    for i, hit in enumerate(hits):
        if hit:
            return 1.0 / (i + 1)
    return 0.0


def compute_precision_at_k(hits: list[bool], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    top_k = hits[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / len(top_k)
