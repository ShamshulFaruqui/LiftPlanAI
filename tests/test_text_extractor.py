"""
Tests for the text-layer parameter extractor (Stage 2a).
Run with: pytest tests/test_text_extractor.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser.text_extractor import (
    parse_length_mm, extract_length_field, extract_shaft_dimensions,
    FLOOR_TO_FLOOR_LABELS, TOTAL_TRAVEL_LABELS,
)


# --- parse_length_mm: unit normalisation ---

def test_parse_length_mm_explicit_mm():
    assert parse_length_mm("3700", "mm") == 3700


def test_parse_length_mm_explicit_metres():
    assert parse_length_mm("3.7", "m") == 3700


def test_parse_length_mm_no_unit_decimal_assumed_metres():
    assert parse_length_mm("3.7", None) == 3700


def test_parse_length_mm_no_unit_integer_assumed_mm():
    assert parse_length_mm("3700", None) == 3700


# --- Floor-to-floor: all four real-world label/unit conventions ---

def test_floor_to_floor_format_ff_ht():
    assert extract_length_field("F-F HT: 3700mm", FLOOR_TO_FLOOR_LABELS) == 3700


def test_floor_to_floor_format_floor_to_floor_metres():
    assert extract_length_field("FLOOR TO FLOOR: 3.70m", FLOOR_TO_FLOOR_LABELS) == 3700


def test_floor_to_floor_format_no_unit():
    assert extract_length_field("F/F HEIGHT = 3700", FLOOR_TO_FLOOR_LABELS) == 3700


def test_floor_to_floor_format_typical_floor_height():
    assert extract_length_field("TYPICAL FLOOR HEIGHT: 3700 MM", FLOOR_TO_FLOOR_LABELS) == 3700


# --- Total travel: all four real-world label/unit conventions ---

def test_total_travel_format_total_travel():
    assert extract_length_field("TOTAL TRAVEL: 14.8m", TOTAL_TRAVEL_LABELS) == 14800


def test_total_travel_format_travel_height_mm():
    assert extract_length_field("TRAVEL HEIGHT: 14800mm", TOTAL_TRAVEL_LABELS) == 14800


def test_total_travel_format_total_rise():
    assert extract_length_field("TOTAL RISE: 14.8 M", TOTAL_TRAVEL_LABELS) == 14800


def test_total_travel_format_overall_travel_oat():
    assert extract_length_field("OVERALL TRAVEL (O.A.T.): 14.8m", TOTAL_TRAVEL_LABELS) == 14800


# --- Shaft dimensions: all three real-world formats ---

def test_shaft_dims_format_plain_mm():
    assert extract_shaft_dimensions("2000x2400mm") == (2000, 2400)


def test_shaft_dims_format_spaced_uppercase():
    assert extract_shaft_dimensions("2000 X 2400 MM") == (2000, 2400)


def test_shaft_dims_format_wd_labels():
    assert extract_shaft_dimensions("W:2000 D:2400") == (2000, 2400)


def test_shaft_dims_format_metres_per_number():
    assert extract_shaft_dimensions("2.0m x 2.4m") == (2000, 2400)


def test_shaft_dims_no_match_returns_none():
    assert extract_shaft_dimensions("NO DIMENSIONS HERE") == (None, None)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


def test_rated_load_speed_extraction():
    from src.parser.text_extractor import extract_load_speed
    load, speed = extract_load_speed("690kg / 1.24m/s")
    assert load == 690
    assert speed == 1.24


def test_rated_load_speed_missing_returns_none():
    from src.parser.text_extractor import extract_load_speed
    load, speed = extract_load_speed("no load speed info here")
    assert load is None
    assert speed is None


def test_extract_text_falls_back_to_pdfplumber_on_mupdf_layer_config_error():
    # Real issue found via real (non-synthetic) drawing testing: some
    # CAD-exported PDFs trigger "MuPDF error: format error: No default
    # Layer config" — a known PyMuPDF/MuPDF bug category with OCG/layer
    # handling. pdfplumber uses a different underlying library and should
    # succeed where PyMuPDF fails on this specific error class.
    from unittest.mock import patch
    import src.parser.text_extractor as te

    sample = "data/synthetic/drawings"
    import glob
    paths = glob.glob(f"{sample}/*LIFT*")
    if not paths:
        return  # dataset not generated in this environment; skip silently

    with patch.object(te.fitz, "open", side_effect=Exception("MuPDF error: format error: No default Layer config")):
        text = te.extract_text(paths[0])
    assert len(text) > 0
    assert "SECTION" in text


def test_extract_text_falls_through_to_ocr_when_both_text_methods_fail():
    # Updated for the three-tier design: when PyMuPDF and pdfplumber both
    # fail, the correct behaviour is now to try OCR next, not to give up
    # immediately — this is the real fix for text-flattened drawings
    # (which raise no exception at all, just return empty text, so OCR
    # must be reachable both via exceptions AND via silent empty results).
    from unittest.mock import patch
    import src.parser.text_extractor as te
    import glob

    paths = glob.glob("data/synthetic/drawings/*LIFT*")
    if not paths:
        return

    with patch.object(te.fitz, "open", side_effect=Exception("original MuPDF error")):
        with patch("pdfplumber.open", side_effect=Exception("pdfplumber also failed")):
            # Real synthetic drawings render cleanly, so OCR should
            # succeed here even though the first two tiers were forced
            # to fail — proving OCR is genuinely reachable as tier 3.
            text = te.extract_text(paths[0])
    assert len(text) > 0


def test_extract_text_returns_empty_when_all_three_tiers_fail():
    from unittest.mock import patch
    import src.parser.text_extractor as te
    import glob

    paths = glob.glob("data/synthetic/drawings/*LIFT*")
    if not paths:
        return

    with patch.object(te.fitz, "open", side_effect=Exception("mupdf failed")):
        with patch("pdfplumber.open", side_effect=Exception("pdfplumber failed")):
            with patch.object(te, "ocr_extract_text", side_effect=Exception("ocr failed")):
                try:
                    te.extract_text(paths[0])
                    assert False, "expected an exception when all three tiers fail"
                except Exception as e:
                    assert "ocr failed" in str(e)


def test_extract_text_empty_but_no_exception_still_falls_through():
    # The specific real bug found: PyMuPDF can open a text-flattened
    # drawing WITHOUT raising any exception at all, and simply return an
    # empty string. An earlier version of this function only checked for
    # exceptions to decide whether to try the next tier, so this exact
    # "succeeded but empty" case silently returned nothing instead of
    # falling through to pdfplumber/OCR.
    from unittest.mock import patch, MagicMock
    import src.parser.text_extractor as te
    import glob

    paths = glob.glob("data/synthetic/drawings/*LIFT*")
    if not paths:
        return

    fake_page = MagicMock()
    fake_page.get_text.return_value = ""  # no exception, just nothing
    fake_doc = MagicMock()
    fake_doc.__iter__ = lambda self: iter([fake_page])

    with patch.object(te.fitz, "open", return_value=fake_doc):
        text = te.extract_text(paths[0])  # should fall through to pdfplumber, then OCR
    assert len(text) > 0  # pdfplumber or OCR should recover real text from the actual file


def test_ocr_extract_text_produces_output_on_real_sample():
    # Uses a synthetic drawing (which has a normal text layer) just to
    # confirm the OCR pipeline mechanics work end-to-end (render -> OCR),
    # not to validate OCR accuracy — that was checked manually against a
    # real text-flattened drawing during development (see module docstring).
    from src.parser.text_extractor import ocr_extract_text
    import glob

    paths = glob.glob("data/synthetic/drawings/*LIFT*")
    if not paths:
        return
    text = ocr_extract_text(paths[0])
    assert len(text) > 0


# --- clean_cad_control_codes: real AutoCAD MTEXT codes found in a real drawing ---

def test_alignment_code_stripped():
    from src.parser.text_extractor import clean_cad_control_codes
    assert clean_cad_control_codes(r"\A1;300") == "300"


def test_percent_p_symbol_code_stripped():
    from src.parser.text_extractor import clean_cad_control_codes
    # AutoCAD's legacy "%%P" special-character code (plus/minus symbol)
    # — unrelated to MTEXT's "\P" paragraph break despite the similar
    # letter, and must not be confused with it.
    assert clean_cad_control_codes("%%P0.00") == "0.00"


def test_paragraph_break_separates_words_not_deleted():
    from src.parser.text_extractor import clean_cad_control_codes
    # Real case: "FUTURE\PKIDS POOL" must not become "FUTUREKIDS POOL"
    # (words wrongly joined) — \P becomes a newline, which downstream
    # \s+-based regex matching treats the same as any other whitespace.
    result = clean_cad_control_codes(r"FUTURE\PKIDS POOL")
    assert "FUTURE" in result.split()
    assert "KIDS" in result.split()


def test_nested_formatting_group_fully_cleaned():
    from src.parser.text_extractor import clean_cad_control_codes
    # The real, complete case from the villa drawing: paragraph
    # alignment code, a width-factor code, a mid-word paragraph break,
    # and a formatting brace group, all in one fragment.
    raw = r"\pxqc;{\W1.5;FUTURE\PKIDS POOL}"
    result = clean_cad_control_codes(raw)
    assert "{" not in result and "}" not in result
    assert "\\" not in result
    assert "FUTURE" in result and "KIDS POOL" in result


def test_complex_comma_containing_code_stripped():
    from src.parser.text_extractor import clean_cad_control_codes
    # Real case: paragraph indent/justify/tab codes can contain commas
    # and digits before the terminating semicolon.
    raw = r"\pxi-3,l4,ql,t4;ALL DIMENSIONS ARE IN CENTIMETERS"
    assert clean_cad_control_codes(raw) == "ALL DIMENSIONS ARE IN CENTIMETERS"


def test_coordinate_values_separated_by_paragraph_break():
    from src.parser.text_extractor import clean_cad_control_codes
    # Real case: two coordinate values were jammed together in the raw
    # extracted text by an embedded "\P" — must not merge into one
    # unreadable run-on token.
    raw = r"X = 506489.769\PY = 2792742.805"
    result = clean_cad_control_codes(raw)
    assert "506489.769" in result
    assert "2792742.805" in result
    assert "506489.769Y" not in result  # confirms they didn't merge


def test_ordinary_text_without_control_codes_unaffected():
    from src.parser.text_extractor import clean_cad_control_codes
    assert clean_cad_control_codes("+5.80 FIRST FLOOR F.F.L") == "+5.80 FIRST FLOOR F.F.L"
    assert clean_cad_control_codes("GL=19.53 = as ref +0.00") == "GL=19.53 = as ref +0.00"


def test_cleaning_is_applied_automatically_inside_extract_text():
    # Regression guard: clean_cad_control_codes must actually be wired
    # into extract_text()'s return path, not just exist as an unused
    # standalone function.
    from unittest.mock import patch
    import src.parser.text_extractor as te
    with patch.object(te, "_extract_text_tiered", return_value=r"\A1;300 test"):
        result = te.extract_text("fake_path.pdf")
    assert result == "300 test"
