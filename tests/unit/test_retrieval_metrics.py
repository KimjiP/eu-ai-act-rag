"""Unit tests for retrieval metric computation."""

import pytest
from src.retrieval.models import RetrievalResult
from eval.metrics.retrieval import compute_mrr, compute_precision_at_k, compute_recall_at_k


def _results(*chunk_ids: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=cid,
            text="text",
            score=1.0 / (i + 1),
            article_number=None,
            section_type="general",
            title="",
            parent_document="",
            rank=i,
        )
        for i, cid in enumerate(chunk_ids)
    ]


class TestRecallAtK:
    def test_relevant_at_rank_1(self):
        results = _results("A", "B", "C")
        assert compute_recall_at_k(results, {"A"}, k=1) == 1.0

    def test_relevant_at_rank_3_not_at_1(self):
        results = _results("X", "Y", "A")
        assert compute_recall_at_k(results, {"A"}, k=1) == 0.0
        assert compute_recall_at_k(results, {"A"}, k=3) == 1.0

    def test_no_relevant_in_results(self):
        results = _results("X", "Y", "Z")
        assert compute_recall_at_k(results, {"A"}, k=5) == 0.0

    def test_multiple_relevant_any_hit_counts(self):
        results = _results("A", "B", "C")
        assert compute_recall_at_k(results, {"B", "D"}, k=2) == 1.0


class TestMRR:
    def test_first_result_relevant(self):
        results = _results("A", "B", "C")
        assert compute_mrr(results, {"A"}) == 1.0

    def test_second_result_relevant(self):
        results = _results("X", "A", "B")
        assert compute_mrr(results, {"A"}) == pytest.approx(1 / 2)

    def test_third_result_relevant(self):
        results = _results("X", "Y", "A")
        assert compute_mrr(results, {"A"}) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        results = _results("X", "Y", "Z")
        assert compute_mrr(results, {"A"}) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        results = _results("A", "B", "C")
        assert compute_precision_at_k(results, {"A", "B", "C"}, k=3) == 1.0

    def test_half_relevant(self):
        results = _results("A", "X", "B", "Y")
        assert compute_precision_at_k(results, {"A", "B"}, k=4) == 0.5

    def test_none_relevant(self):
        results = _results("X", "Y", "Z")
        assert compute_precision_at_k(results, {"A"}, k=3) == 0.0

    def test_k_larger_than_results(self):
        results = _results("A", "B")
        assert compute_precision_at_k(results, {"A"}, k=10) == 0.5
