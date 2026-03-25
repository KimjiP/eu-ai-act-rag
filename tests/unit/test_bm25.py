"""Unit tests for BM25 index."""

import json
import tempfile
from pathlib import Path

import pytest
from src.retrieval.bm25 import BM25Index


@pytest.fixture
def sample_chunks_file():
    """Create a temporary chunks JSON file for testing."""
    chunks = [
        {
            "chunk_id": "hash_001",
            "text": "Providers of high-risk AI systems shall establish quality management systems.",
            "article_number": "Article 9",
            "section_type": "obligation",
            "title": "Risk Management",
            "chunk_index": 0,
            "parent_document": "test.pdf",
            "source_url": "",
            "publication_date": "2024-07-12",
            "ingestion_run_id": "test",
            "chunk_hash": "hash_001",
            "cross_references": [],
        },
        {
            "chunk_id": "hash_002",
            "text": "Real-time biometric identification systems in public spaces are prohibited.",
            "article_number": "Article 5",
            "section_type": "prohibition",
            "title": "Prohibited Practices",
            "chunk_index": 1,
            "parent_document": "test.pdf",
            "source_url": "",
            "publication_date": "2024-07-12",
            "ingestion_run_id": "test",
            "chunk_hash": "hash_002",
            "cross_references": [],
        },
        {
            "chunk_id": "hash_003",
            "text": "Transparency obligations require providers to inform users about AI system capabilities.",
            "article_number": "Article 13",
            "section_type": "obligation",
            "title": "Transparency",
            "chunk_index": 2,
            "parent_document": "test.pdf",
            "source_url": "",
            "publication_date": "2024-07-12",
            "ingestion_run_id": "test",
            "chunk_hash": "hash_003",
            "cross_references": [],
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(chunks, f)
        return Path(f.name)


class TestBM25Index:
    def test_builds_from_json(self, sample_chunks_file):
        index = BM25Index(chunks_json_path=sample_chunks_file)
        assert index is not None

    def test_returns_results(self, sample_chunks_file):
        index = BM25Index(chunks_json_path=sample_chunks_file)
        results = index.search("quality management systems", top_k=3)
        assert len(results) > 0

    def test_relevant_result_ranks_high(self, sample_chunks_file):
        index = BM25Index(chunks_json_path=sample_chunks_file)
        results = index.search("biometric identification prohibited", top_k=3)
        top_result = results[0]
        assert top_result.chunk_id == "hash_002"

    def test_respects_top_k(self, sample_chunks_file):
        index = BM25Index(chunks_json_path=sample_chunks_file)
        results = index.search("AI system", top_k=2)
        assert len(results) <= 2

    def test_result_has_article_number(self, sample_chunks_file):
        index = BM25Index(chunks_json_path=sample_chunks_file)
        results = index.search("transparency obligations", top_k=1)
        assert results[0].article_number is not None

    def test_rank_is_zero_indexed(self, sample_chunks_file):
        index = BM25Index(chunks_json_path=sample_chunks_file)
        results = index.search("providers", top_k=3)
        ranks = [r.rank for r in results]
        assert ranks == list(range(len(results)))
