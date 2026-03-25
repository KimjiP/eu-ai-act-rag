"""Unit tests for citation extraction and verification."""

import pytest
from src.generation.citations import (
    CitationVerification,
    detect_bad_framing,
    extract_citations,
    verify_citations,
)
from src.retrieval.models import RetrievalResult


def _make_result(article_number: str, chunk_id: str = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or f"hash_{article_number.replace(' ', '_')}",
        text="Some text.",
        score=0.5,
        article_number=article_number,
        section_type="obligation",
        title="Test",
        parent_document="test.pdf",
        rank=0,
    )


class TestExtractCitations:
    def test_extracts_article(self):
        answer = "Providers must comply with Article 17(1) requirements."
        citations = extract_citations(answer)
        assert "Article 17(1)" in citations

    def test_extracts_annex(self):
        answer = "Systems listed in Annex III are classified as high-risk."
        citations = extract_citations(answer)
        assert any("Annex III" in c for c in citations)

    def test_extracts_recital(self):
        answer = "As stated in Recital 42, the purpose is to ensure safety."
        citations = extract_citations(answer)
        assert any("Recital 42" in c for c in citations)

    def test_deduplicates_citations(self):
        answer = "Article 6 applies here. Article 6 also applies there."
        citations = extract_citations(answer)
        assert citations.count("Article 6") == 1

    def test_extracts_multiple_citations(self):
        answer = "See Article 9(2) and Annex III for details (Recital 42)."
        citations = extract_citations(answer)
        assert len(citations) >= 2

    def test_no_citations_returns_empty(self):
        answer = "This answer has no citations."
        citations = extract_citations(answer)
        assert citations == []


class TestVerifyCitations:
    def test_matched_citation(self):
        citations = ["Article 17"]
        chunks = [_make_result("Article 17")]
        result = verify_citations(citations, chunks)
        assert result.accuracy == 1.0
        assert "Article 17" in result.matched

    def test_missing_citation(self):
        citations = ["Article 50"]
        chunks = [_make_result("Article 17")]
        result = verify_citations(citations, chunks)
        assert result.accuracy == 0.0
        assert "Article 50" in result.missing

    def test_partial_match_article_paragraph(self):
        # "Article 17" should match chunk with "Article 17(1)"
        citations = ["Article 17"]
        chunks = [_make_result("Article 17(1)")]
        result = verify_citations(citations, chunks)
        assert result.accuracy == 1.0

    def test_no_citations_returns_perfect_accuracy(self):
        result = verify_citations([], [_make_result("Article 1")])
        assert result.accuracy == 1.0

    def test_mixed_matched_and_missing(self):
        citations = ["Article 6", "Article 99"]
        chunks = [_make_result("Article 6")]
        result = verify_citations(citations, chunks)
        assert result.accuracy == 0.5
        assert "Article 6" in result.matched
        assert "Article 99" in result.missing


class TestDetectBadFraming:
    def test_detects_bad_framing(self):
        answer = "Based on the documents you provided, Article 17 applies."
        assert detect_bad_framing(answer) is True

    def test_detects_transcript_variant(self):
        answer = "The transcripts you provided indicate that Article 9 requires..."
        assert detect_bad_framing(answer) is True

    def test_clean_answer_not_flagged(self):
        answer = "Providers must comply with Article 17(1). (Article 17(1))"
        assert detect_bad_framing(answer) is False

    def test_disclaimer_not_flagged(self):
        answer = "⚠️ This is informational support only, not legal advice."
        assert detect_bad_framing(answer) is False
