"""Unit tests for the chunking strategies."""

import pytest
from src.ingestion.parser import SectionType, StructuralUnit
from src.ingestion.chunker import Chunk, chunk_fixed_size, chunk_structural


def _make_unit(text: str, article: str = "Article 1", idx: int = 0) -> StructuralUnit:
    return StructuralUnit(
        article_number=article,
        section_type=SectionType.OBLIGATION,
        title="Test Section",
        text=text,
        chunk_index=idx,
        parent_document="test.pdf",
        source_url="https://example.com",
        publication_date="2024-07-12",
    )


class TestChunkStructural:
    def test_one_chunk_per_unit(self):
        units = [_make_unit(f"Text {i}", f"Article {i}", i) for i in range(5)]
        chunks = chunk_structural(units)
        assert len(chunks) == 5

    def test_chunk_preserves_article_number(self):
        units = [_make_unit("Some obligation text.", "Article 17")]
        chunks = chunk_structural(units)
        assert chunks[0].article_number == "Article 17"

    def test_chunk_id_is_content_hash(self):
        units = [_make_unit("Deterministic text.")]
        chunks_1 = chunk_structural(units)
        chunks_2 = chunk_structural(units)
        assert chunks_1[0].chunk_id == chunks_2[0].chunk_id

    def test_different_texts_different_ids(self):
        u1 = _make_unit("First text.", "Article 1")
        u2 = _make_unit("Second text.", "Article 2")
        chunks = chunk_structural([u1, u2])
        assert chunks[0].chunk_id != chunks[1].chunk_id

    def test_empty_units_skipped(self):
        units = [_make_unit(""), _make_unit("  "), _make_unit("Real content.")]
        chunks = chunk_structural(units)
        assert len(chunks) == 1

    def test_ingestion_run_id_stored(self):
        units = [_make_unit("Content.")]
        chunks = chunk_structural(units, ingestion_run_id="test-run-001")
        assert chunks[0].ingestion_run_id == "test-run-001"

    def test_section_type_stored_as_string(self):
        units = [_make_unit("Content.")]
        chunks = chunk_structural(units)
        assert isinstance(chunks[0].section_type, str)
        assert chunks[0].section_type == "obligation"


class TestChunkFixedSize:
    def test_respects_token_limit(self):
        long_text = " ".join([f"word{i}" for i in range(1000)])
        units = [_make_unit(long_text)]
        chunks = chunk_fixed_size(units, token_limit=100, overlap=0)
        for chunk in chunks:
            word_count = len(chunk.text.split())
            assert word_count <= 100

    def test_produces_multiple_chunks_from_long_text(self):
        long_text = " ".join([f"word{i}" for i in range(600)])
        units = [_make_unit(long_text)]
        chunks = chunk_fixed_size(units, token_limit=200, overlap=0)
        assert len(chunks) >= 3

    def test_overlap_creates_extra_chunks(self):
        text = " ".join([f"word{i}" for i in range(200)])
        units = [_make_unit(text)]
        chunks_no_overlap = chunk_fixed_size(units, token_limit=100, overlap=0)
        chunks_with_overlap = chunk_fixed_size(units, token_limit=100, overlap=50)
        assert len(chunks_with_overlap) > len(chunks_no_overlap)

    def test_chunk_inherits_metadata_from_unit(self):
        units = [_make_unit("Content words " * 50, article="Article 9")]
        chunks = chunk_fixed_size(units, token_limit=20, overlap=0)
        assert all(c.article_number == "Article 9" for c in chunks)
