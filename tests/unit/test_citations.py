"""Unit tests for citation extraction and verification."""

import pytest
from src.generation.citations import (
    CitationVerification,
    ProvisionRef,
    detect_bad_framing,
    detect_decline,
    extract_citations,
    parse_reference,
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

    def test_paragraph_citation_matches_whole_article_chunk(self):
        result = verify_citations(["Article 17(1)(a)"], [_make_result("Article 17")])
        assert result.accuracy == 1.0

    @pytest.mark.parametrize(
        "cited, retrieved",
        [
            ("Article 13", "Article 1"),
            ("Article 1", "Article 13"),
            ("Article 52(1)", "Article 5"),
            ("Annex III", "Annex I"),
            ("Recital 42", "Recital 4"),
        ],
    )
    def test_string_prefix_is_not_a_match(self, cited, retrieved):
        # The original check compared strings by prefix and passed all of these.
        result = verify_citations([cited], [_make_result(retrieved)])
        assert result.accuracy == 0.0
        assert result.missing == [cited]

    def test_definition_citations(self):
        chunks = [_make_result("Article 3(56)")]
        assert verify_citations(["Article 3(56)"], chunks).accuracy == 1.0
        assert verify_citations(["Article 3"], chunks).accuracy == 1.0
        assert verify_citations(["Article 3(12)"], chunks).accuracy == 0.0

    def test_unparseable_chunk_labels_ignored(self):
        result = verify_citations(["Article 1"], [_make_result("Preamble")])
        assert result.accuracy == 0.0


class TestParseReference:
    def test_article_with_path(self):
        assert parse_reference("Article 17(1)(a)") == ProvisionRef("article", "17", ("1", "a"))

    def test_annex_with_section(self):
        assert parse_reference("Annex VIII, Section A") == ProvisionRef("annex", "VIII", ("section a",))

    def test_recital(self):
        assert parse_reference("Recital 42") == ProvisionRef("recital", "42")

    def test_not_a_reference(self):
        assert parse_reference("Preamble") is None

    def test_extracts_definition_and_nested_points(self):
        citations = extract_citations("See Article 3(56) and Article 5(1)(h)(iii).")
        assert citations == ["Article 3(56)", "Article 5(1)(h)(iii)"]

    def test_skips_references_to_other_acts(self):
        answer = (
            "The Commission informs the committee referred to in Article 22 of Regulation (EU) "
            "No 1025/2012 (Article 41(2)), based on Article 114 TFEU and Article 39 of the Charter."
        )
        assert extract_citations(answer) == ["Article 41(2)"]


class TestDetectDecline:
    def test_prompt_decline_sentence(self):
        answer = (
            "The provided regulatory text does not contain sufficient information to answer "
            "this question. National implementation acts may contain the answer."
        )
        assert detect_decline(answer) is True

    def test_normal_answer(self):
        answer = "Providers must establish a risk management system (Article 9(1))."
        assert detect_decline(answer) is False


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
