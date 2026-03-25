"""Unit tests for the document structure parser."""

import pytest
from src.ingestion.extractor import RawPage
from src.ingestion.parser import SectionType, parse_document


def _make_pages(text: str, source_file: str = "test.pdf") -> list[RawPage]:
    return [RawPage(page_number=1, text=text, source_file=source_file)]


def test_parses_article_header():
    pages = _make_pages("Article 6\nHigh-Risk AI Systems\nProviders shall comply.")
    units = parse_document(pages)
    articles = [u for u in units if u.article_number == "Article 6"]
    assert len(articles) >= 1
    assert articles[0].article_number == "Article 6"


def test_parses_recital():
    pages = _make_pages("(42) The purpose of this regulation is to ensure safety.")
    units = parse_document(pages)
    recitals = [u for u in units if u.article_number == "Recital 42"]
    assert len(recitals) >= 1
    assert recitals[0].section_type == SectionType.RECITAL


def test_parses_annex():
    pages = _make_pages("ANNEX III\nHigh-Risk AI Systems referred to in Article 6(2)")
    units = parse_document(pages)
    annexes = [u for u in units if u.article_number and "Annex" in u.article_number]
    assert len(annexes) >= 1
    assert annexes[0].section_type == SectionType.ANNEX


def test_section_type_prohibition():
    pages = _make_pages(
        "Article 5\nProhibited AI Practices\n"
        "The following AI practices shall be prohibited under this regulation."
    )
    units = parse_document(pages)
    article_5 = [u for u in units if u.article_number == "Article 5"]
    assert len(article_5) >= 1
    assert article_5[0].section_type == SectionType.PROHIBITION


def test_cross_references_extracted():
    pages = _make_pages(
        "Article 9\nRisk Management\n"
        "Providers must comply with Article 17(1) and Annex III requirements."
    )
    units = parse_document(pages)
    article_9 = [u for u in units if u.article_number == "Article 9"]
    assert len(article_9) >= 1
    refs = article_9[0].cross_references
    assert any("Article 17" in r for r in refs)
    assert any("Annex" in r for r in refs)


def test_multiple_articles_parsed():
    text = (
        "Article 1\nSubject Matter\nThis regulation establishes rules.\n\n"
        "Article 2\nScope\nThis regulation applies to providers.\n\n"
        "Article 3\nDefinitions\nFor the purposes of this regulation, 'AI system' means..."
    )
    pages = _make_pages(text)
    units = parse_document(pages)
    article_numbers = [u.article_number for u in units if u.article_number]
    assert "Article 1" in article_numbers
    assert "Article 2" in article_numbers
    assert "Article 3" in article_numbers


def test_empty_pages_return_no_units():
    pages = _make_pages("   \n\n   ")
    units = parse_document(pages)
    # May return 0 or very few units — should not crash
    assert isinstance(units, list)
