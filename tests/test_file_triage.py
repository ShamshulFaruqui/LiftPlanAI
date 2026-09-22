"""
Basic tests for the file triage stage.
Run with: pytest tests/test_file_triage.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.triage.file_triage import fuzzy_contains, check_filename


def test_exact_match():
    assert fuzzy_contains("LIFT SECTION DETAIL", "LIFT") is True


def test_fuzzy_typo_match():
    assert fuzzy_contains("4PLEX LIFT DETALS 00", "DETAILS") is True


def test_no_match():
    assert fuzzy_contains("STRUCTURAL FOUNDATION PLAN", "ELEVATOR") is False


def test_filename_matching():
    fake_path = Path("360-1171-DWG-4PB-AR-401-4PLEX LIFT DETALS-00.pdf")
    matches = check_filename(fake_path)
    assert len(matches) > 0, "Should detect LIFT in the filename despite messy naming"


def test_service_lift_abbreviation():
    assert fuzzy_contains("SL01 SL02 SHAFT DIMENSIONS", "SL") is True


def test_firefighter_lift_code():
    assert fuzzy_contains("FFL01 EN81-72 DEDICATED SHAFT", "EN81-72") is True


def test_electrical_discipline_code_not_falsely_matched():
    # Real bug found via distractor testing: "EL" (elevator abbreviation)
    # collided with "EL" as the standard AEC discipline code for Electrical
    # drawings, e.g. filenames like "360-2018-DWG-EL-318-CABLE ROUTING PLAN".
    assert fuzzy_contains("360-2018-DWG-EL-318-CABLE ROUTING PLAN-00", "EL") is False


def test_plumbing_discipline_code_not_falsely_matched():
    assert fuzzy_contains("360-2013-DWG-PL-313-SANITARY RISER DIAGRAM-00", "PL") is False


def test_elevator_label_with_digits_still_matches():
    assert fuzzy_contains("EL01 EL02 PASSENGER LIFT GROUP", "EL") is True


def test_fire_fighting_plumbing_not_falsely_matched_as_firefighter_lift():
    # Real false positive found via distractor testing: a plumbing drawing
    # titled "FIRE FIGHTING PIPING PLAN" (fire suppression system — sprinklers,
    # hose reels) was wrongly flagged elevator-relevant because the bare "FIRE"
    # keyword matched. Fire-fighting/fire-suppression MEP drawings are a
    # completely distinct, extremely common drawing category from firefighter
    # LIFTS, so "FIRE" alone is excluded from the lexicon (checked below) —
    # here we confirm none of the *actual* firefighter keywords match either.
    from src.triage.file_triage import KEYWORD_LEXICON
    text = "FIRE FIGHTING PIPING PLAN SPRINKLER LAYOUT"
    firefighter_matches = [kw for kw in KEYWORD_LEXICON["firefighter"] if fuzzy_contains(text, kw)]
    assert firefighter_matches == []


def test_finished_floor_not_falsely_matched_as_firefighter_abbreviation():
    # "FF" alone commonly means "Finished Floor" (a level marker) in real
    # architectural drawings — an even more common false-positive risk than
    # "FIRE" — so a drawing mentioning finished floor levels should not be
    # picked up by any firefighter-lift keyword.
    from src.triage.file_triage import KEYWORD_LEXICON
    text = "FF LEVEL 100.500 FINISHED FLOOR DATUM"
    firefighter_matches = [kw for kw in KEYWORD_LEXICON["firefighter"] if fuzzy_contains(text, kw)]
    assert firefighter_matches == []


def test_firefighter_keywords_are_specific_not_generic():
    from src.triage.file_triage import KEYWORD_LEXICON
    firefighter_keywords = KEYWORD_LEXICON["firefighter"]
    assert "FIRE" not in firefighter_keywords, "bare 'FIRE' collides with fire-suppression MEP drawings"
    assert "FF" not in firefighter_keywords, "bare 'FF' collides with 'Finished Floor' level marker"


# --- Regression tests for the silent-OCG-degradation fix -------------------
# Found via a real 41-page A1 CAD-exported drawing set: PyMuPDF hit an
# Optional-Content-Group parsing issue and returned next to no text for
# every page WITHOUT raising a Python exception, which previously read as
# "genuinely no elevator-relevant content" rather than "could not be read".
# These mock PyMuPDF/pdfplumber directly rather than shipping the 83MB real
# file that triggered this, since the failure mode (empty per-page text on
# a multi-page doc) is what matters, not that specific file's exact bytes.

from unittest.mock import patch, MagicMock


def _mock_fitz_doc(page_texts):
    """Build a MagicMock standing in for a fitz.Document with the given
    per-page get_text() return values, indexable like the real object."""
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.get_text.return_value = text
        pages.append(page)
    doc = MagicMock()
    doc.page_count = len(pages)
    doc.__getitem__.side_effect = lambda i: pages[i]
    return doc


def test_pdfplumber_fallback_recovers_a_genuine_match():
    from src.triage.file_triage import check_pdf_text

    mock_doc = _mock_fitz_doc(["", "", ""])  # PyMuPDF: near-empty on every page
    mock_pdfplumber_doc = MagicMock()
    pl_page = MagicMock()
    pl_page.extract_text.return_value = "LIFT SHAFT DETAIL EL01"
    mock_pdfplumber_doc.__enter__.return_value.pages = [pl_page]

    with patch("src.triage.file_triage.fitz") as mock_fitz, \
         patch("pdfplumber.open", return_value=mock_pdfplumber_doc):
        mock_fitz.open.return_value = mock_doc
        matches, page_count, error = check_pdf_text(Path("fake.pdf"))

    assert error == "", "a successful fallback should not report an error"
    assert page_count == 3
    assert matches, "the fallback-recovered text should have matched a keyword"


def test_pdfplumber_fallback_timeout_reports_uncertain_not_irrelevant():
    from src.triage.file_triage import check_pdf_text

    mock_doc = _mock_fitz_doc(["", "", ""])

    def _hang_forever(*args, **kwargs):
        import time
        time.sleep(60)  # longer than the fallback's own alarm — must be interrupted

    with patch("src.triage.file_triage.fitz") as mock_fitz, \
         patch("pdfplumber.open", side_effect=_hang_forever):
        mock_fitz.open.return_value = mock_doc
        matches, page_count, error = check_pdf_text(Path("fake.pdf"))

    assert matches == {}
    assert "possible read failure" in error, (
        "a timed-out fallback must be reported as uncertain, not silently "
        "returned as a confident 'no matches found'"
    )


def test_single_page_empty_text_is_not_treated_as_a_read_failure():
    """A single blank/near-blank page is the ordinary case (e.g. a cover
    sheet) and must NOT trigger the multi-page-degradation fallback path."""
    from src.triage.file_triage import check_pdf_text

    mock_doc = _mock_fitz_doc([""])

    with patch("src.triage.file_triage.fitz") as mock_fitz, \
         patch("pdfplumber.open") as mock_pdfplumber_open:
        mock_fitz.open.return_value = mock_doc
        matches, page_count, error = check_pdf_text(Path("fake.pdf"))

    mock_pdfplumber_open.assert_not_called()
    assert error == ""


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
