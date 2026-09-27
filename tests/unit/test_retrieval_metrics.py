"""Unit tests for retrieval metric computation."""

import pytest
from eval.metrics.retrieval import compute_mrr, compute_precision_at_k, compute_recall_at_k
from eval.provisions import contains_any, refers_to


class TestRecallAtK:
    def test_relevant_at_rank_1(self):
        assert compute_recall_at_k([True, False, False], k=1) == 1.0

    def test_relevant_at_rank_3_not_at_1(self):
        hits = [False, False, True]
        assert compute_recall_at_k(hits, k=1) == 0.0
        assert compute_recall_at_k(hits, k=3) == 1.0

    def test_no_relevant_in_results(self):
        assert compute_recall_at_k([False, False, False], k=5) == 0.0


class TestMRR:
    def test_first_result_relevant(self):
        assert compute_mrr([True, False, False]) == 1.0

    def test_second_result_relevant(self):
        assert compute_mrr([False, True, True]) == pytest.approx(1 / 2)

    def test_third_result_relevant(self):
        assert compute_mrr([False, False, True]) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert compute_mrr([False, False, False]) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert compute_precision_at_k([True, True, True], k=3) == 1.0

    def test_half_relevant(self):
        assert compute_precision_at_k([True, False, True, False], k=4) == 0.5

    def test_none_relevant(self):
        assert compute_precision_at_k([False, False, False], k=3) == 0.0

    def test_k_larger_than_results(self):
        assert compute_precision_at_k([True, False], k=10) == 0.5


class TestProvisionMatching:
    def test_definition_matches_whole_article(self):
        assert refers_to("Article 3", "Article 3(56)")
        assert not refers_to("Article 3(12)", "Article 3(56)")

    def test_no_prefix_matching(self):
        assert not refers_to("Article 1", "Article 13")
        assert not refers_to("Annex I", "Annex III")

    def test_non_article_labels_compared_as_text(self):
        assert refers_to("Preamble", "Preamble")
        assert not refers_to("Preamble", "Article 1")

    def test_contains_any(self):
        assert contains_any(["Recital 12", "Article 9"], ["Article 9(2)"])
        assert not contains_any(["Article 90"], ["Article 9"])
