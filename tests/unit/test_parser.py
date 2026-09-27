"""Unit tests for the document structure parser."""

from pathlib import Path

import pytest
from src.ingestion.extractor import RawPage
from src.ingestion.parser import SectionType, parse_document, summarize_structure

PDF_PATH = Path(__file__).parents[2] / "data" / "raw" / "OJ_L_202401689_EN_TXT.pdf"


def _make_pages(*texts: str, source_file: str = "test.pdf") -> list[RawPage]:
    return [RawPage(page_number=i + 1, text=t, source_file=source_file) for i, t in enumerate(texts)]


def _labels(units) -> list[str | None]:
    return [u.article_number for u in units]


def _by_label(units) -> dict:
    return {u.article_number: u for u in units}


# ---------------------------------------------------------------------------
# A condensed document in the Official Journal layout
# ---------------------------------------------------------------------------

OJ_PAGE_1 = """REGULATION (EU) 2024/1689 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL
of 13 June 2024
Having regard to the opinion of the European Economic and Social Committee (1),
Whereas:
(1)
The purpose of this Regulation is to improve the functioning of the internal market.
(2)
In order to address the risks of undue external interference with the right to vote enshrined in
Article 39 of the Charter, AI systems intended to influence elections should be high-risk.
EN
OJ L, 12.7.2024
1/144
ELI: http://data.europa.eu/eli/reg/2024/1689/oj
(1)
OJ C 517, 22.12.2021, p. 56.
"""

OJ_PAGE_2 = """(3)
Divergent national rules may fragment the internal market.
HAVE ADOPTED THIS REGULATION:
CHAPTER I
GENERAL PROVISIONS
Article 1
Subject matter
The purpose of this Regulation is to improve the functioning of the internal market.
Article 2
Scope
This Regulation applies to deployers of AI systems as referred to in
Article 13;
and to providers.
Article 3
Definitions
For the purposes of this Regulation, the following definitions apply:
(1)
‘AI system’ means a machine-based system designed to operate with varying levels of autonomy;
(2)
‘risk’ means the combination of the probability of an occurrence of harm and its severity;
(3) ‘AI literacy’ means skills, knowledge and understanding that allow informed deployment;
CHAPTER II
PROHIBITED AI PRACTICES
Article 4
Prohibited AI practices
The following AI practices shall be prohibited, including those listed in
Article 9
of Directive (EU) 2016/680.
Done at Brussels, 13 June 2024.
For the European Parliament
The President
R. METSOLA
ANNEX I
List of Union harmonisation legislation
Annex I, the market surveillance authority shall be the authority responsible under that legislation.
ANNEX II
Information to be submitted upon the registration of high-risk AI systems in accordance with
Article 49
Section A — Information to be submitted by providers
EN
OJ L, 12.7.2024
2/144
ELI: http://data.europa.eu/eli/reg/2024/1689/oj
(2)
OJ L 119, 4.5.2016, p. 1.
"""


@pytest.fixture
def oj_units():
    return parse_document(_make_pages(OJ_PAGE_1, OJ_PAGE_2))


class TestOfficialJournalLayout:
    def test_units_in_document_order(self, oj_units):
        assert _labels(oj_units) == [
            "Preamble",
            "Recital 1",
            "Recital 2",
            "Recital 3",
            "Article 1",
            "Article 2",
            "Article 3(1)",
            "Article 3(2)",
            "Article 3(3)",
            "Article 4",
            "Annex I",
            "Annex II",
        ]

    def test_cross_reference_at_line_start_stays_in_recital(self, oj_units):
        # The original parser started a fake "Article 39" unit here.
        assert "Article 39 of the Charter" in _by_label(oj_units)["Recital 2"].text

    def test_cross_reference_line_stays_in_article(self, oj_units):
        units = _by_label(oj_units)
        assert "Article 13;" in units["Article 2"].text
        assert "and to providers." in units["Article 2"].text

    def test_standalone_out_of_sequence_article_line_is_body_text(self, oj_units):
        # "Article 9" alone on a line inside Article 4 is a cross-reference, not Article 5.
        assert "Article 9\nof Directive (EU) 2016/680." in _by_label(oj_units)["Article 4"].text

    def test_standalone_article_line_in_annex_stays_in_annex(self, oj_units):
        annex = _by_label(oj_units)["Annex II"]
        assert "Article 49" in annex.text
        assert "Section A" in annex.text

    def test_mixed_case_annex_mention_is_not_a_header(self, oj_units):
        assert "Annex I, the market surveillance authority" in _by_label(oj_units)["Annex I"].text

    def test_definitions_become_one_unit_each(self, oj_units):
        units = _by_label(oj_units)
        literacy = units["Article 3(3)"]
        assert literacy.section_type == SectionType.DEFINITION
        assert literacy.title == "Definitions"
        assert literacy.text.startswith("Article 3\nDefinitions\n(3) ‘AI literacy’ means")
        assert "‘risk’" not in literacy.text

    def test_definitions_are_not_recitals(self, oj_units):
        assert not any(u.section_type == SectionType.RECITAL and "means" in u.text for u in oj_units)

    def test_footnotes_and_page_furniture_removed(self, oj_units):
        all_text = "\n".join(u.text for u in oj_units)
        assert "OJ C 517" not in all_text
        assert "ELI:" not in all_text
        assert "OJ L, 12.7.2024" not in all_text
        assert "1/144" not in all_text

    def test_chapter_headings_and_signature_dropped(self, oj_units):
        all_text = "\n".join(u.text for u in oj_units)
        for dropped in ("CHAPTER I", "GENERAL PROVISIONS", "PROHIBITED AI PRACTICES", "METSOLA"):
            assert dropped not in all_text

    def test_titles_captured(self, oj_units):
        units = _by_label(oj_units)
        assert units["Article 2"].title == "Scope"
        assert units["Annex II"].title.endswith("in accordance with Article 49")

    def test_article_section_type_from_title(self, oj_units):
        assert _by_label(oj_units)["Article 4"].section_type == SectionType.PROHIBITION


# ---------------------------------------------------------------------------
# Lenient mode: short fixtures without the OJ region markers
# ---------------------------------------------------------------------------


def test_parses_article_header():
    units = parse_document(_make_pages("Article 6\nHigh-Risk AI Systems\nProviders shall comply."))
    assert _labels(units) == ["Article 6"]
    assert units[0].title == "High-Risk AI Systems"


def test_parses_recital():
    units = parse_document(_make_pages("(42)\nThe purpose of this regulation is to ensure safety."))
    recitals = [u for u in units if u.article_number == "Recital 42"]
    assert len(recitals) == 1
    assert recitals[0].section_type == SectionType.RECITAL


def test_parses_annex():
    units = parse_document(_make_pages("ANNEX III\nHigh-Risk AI Systems referred to in Article 6(2)"))
    assert _labels(units) == ["Annex III"]
    assert units[0].section_type == SectionType.ANNEX


def test_section_type_prohibition():
    units = parse_document(
        _make_pages(
            "Article 5\nProhibited AI Practices\n"
            "The following AI practices shall be prohibited under this regulation."
        )
    )
    assert _by_label(units)["Article 5"].section_type == SectionType.PROHIBITION


def test_cross_references_extracted():
    units = parse_document(
        _make_pages(
            "Article 9\nRisk Management\n"
            "Providers must comply with Article 17(1) and Annex III requirements."
        )
    )
    refs = _by_label(units)["Article 9"].cross_references
    assert any("Article 17" in r for r in refs)
    assert any("Annex" in r for r in refs)


def test_multiple_articles_parsed():
    text = (
        "Article 1\nSubject Matter\nThis regulation establishes rules.\n\n"
        "Article 2\nScope\nThis regulation applies to providers.\n\n"
        "Article 3\nDefinitions\nFor the purposes of this regulation, 'AI system' means..."
    )
    units = parse_document(_make_pages(text))
    # Article 3 without numbered definitions stays one unit.
    assert _labels(units) == ["Article 1", "Article 2", "Article 3"]


def test_out_of_sequence_header_is_body_text():
    units = parse_document(_make_pages("Article 1\nScope\nSee also\nArticle 7\nfor details."))
    assert _labels(units) == ["Article 1"]
    assert "Article 7" in units[0].text


def test_empty_pages_return_no_units():
    assert parse_document(_make_pages("   \n\n   ")) == []


# ---------------------------------------------------------------------------
# The real Regulation (skipped when the PDF is not downloaded)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PDF_PATH.exists(), reason="EU AI Act PDF not in data/raw/")
def test_real_regulation_structure():
    from src.ingestion.extractor import extract_pdf

    units = parse_document(extract_pdf(PDF_PATH))
    structure = summarize_structure(units)

    assert structure["recital"] == list(range(1, 181))
    assert structure["article"] == [n for n in range(1, 114) if n != 3]
    assert structure["definition"] == list(range(1, 69))
    assert structure["annex"] == list(range(1, 14))
    assert len({u.article_number for u in units}) == len(units)
    assert max(len(u.text) for u in units) < 15_000

    by_label = _by_label(units)
    assert "Article 39 of the Charter" in by_label["Recital 62"].text
    assert by_label["Article 3(56)"].text.startswith("Article 3\nDefinitions\n(56) ‘AI literacy’")
    assert "(d) the adoption of appropriate and targeted risk management measures" in (
        by_label["Article 9"].text.replace("\n", " ")
    )
