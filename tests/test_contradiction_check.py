"""
Tests for the contradiction detector.
Run with: pytest tests/test_contradiction_check.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contradiction.contradiction_check import (
    compare_field, TRAVEL_TOLERANCE_M, SHAFT_WIDTH_TOLERANCE_MM,
)


def test_matching_values_do_not_contradict():
    comparison = compare_field("total_travel_m", 24.1, 24.1, TRAVEL_TOLERANCE_M)
    assert comparison.contradicts is False


def test_small_rounding_difference_within_tolerance_does_not_contradict():
    comparison = compare_field("total_travel_m", 24.1, 24.1 + (TRAVEL_TOLERANCE_M - 0.02),
                                TRAVEL_TOLERANCE_M)
    assert comparison.contradicts is False


def test_genuine_difference_beyond_tolerance_contradicts():
    comparison = compare_field("total_travel_m", 24.1, 27.7, TRAVEL_TOLERANCE_M)
    assert comparison.contradicts is True


def test_drawing_value_is_always_authoritative():
    comparison = compare_field("total_travel_m", 24.1, 27.7, TRAVEL_TOLERANCE_M)
    assert comparison.authoritative_value == 24.1  # the drawing's value, not the spec's


def test_shaft_width_contradiction_detected():
    comparison = compare_field("shaft_width_mm", 2100, 2300, SHAFT_WIDTH_TOLERANCE_MM)
    assert comparison.contradicts is True


def test_shaft_width_small_difference_within_tolerance():
    comparison = compare_field("shaft_width_mm", 2100, 2110, SHAFT_WIDTH_TOLERANCE_MM)
    assert comparison.contradicts is False


def test_missing_drawing_value_does_not_crash_and_has_no_verdict():
    comparison = compare_field("total_travel_m", None, 24.1, TRAVEL_TOLERANCE_M)
    assert comparison.contradicts is None
    assert comparison.authoritative_value is None


def test_missing_spec_value_does_not_crash_and_has_no_verdict():
    comparison = compare_field("total_travel_m", 24.1, None, TRAVEL_TOLERANCE_M)
    assert comparison.contradicts is None
    assert comparison.authoritative_value == 24.1  # drawing value still recorded


# --- Regression tests: distinguishing "no contradiction" from "spec could
# not be read at all" ------------------------------------------------------
# Found via testing against a real (non-synthetic) specification document:
# spec_reader.py's patterns only match this project's own synthetic spec
# format, so every field came back None for a real spec sheet — which,
# unguarded, is indistinguishable in the output from a spec that was read
# fine and simply agreed with the drawing.

from unittest.mock import patch, MagicMock
from src.contradiction.contradiction_check import check_contradictions


def _mock_drawing_result(total_travel_m=24.1, shaft_width_mm=1500):
    drawing = MagicMock()
    drawing.total_travel_m = total_travel_m
    group = MagicMock()
    group.shaft_width_mm = shaft_width_mm
    drawing.groups = [group]
    return drawing


def test_spec_extraction_uncertain_when_both_fields_unreadable():
    unreadable_spec = MagicMock(total_travel_m=None, shaft_width_mm=None)
    with patch("src.contradiction.contradiction_check.extract_parameters",
               return_value=_mock_drawing_result()), \
         patch("src.contradiction.contradiction_check.extract_spec_parameters",
               return_value=unreadable_spec):
        result = check_contradictions(Path("fake_drawing.pdf"), Path("fake_spec.pdf"))

    assert result.spec_extraction_uncertain is True
    assert result.has_contradiction is False, "uncertain is not the same claim as a detected contradiction"
    assert any("could not be read" in f for f in result.flags)


def test_spec_extraction_uncertain_false_when_a_field_was_read():
    partially_readable_spec = MagicMock(total_travel_m=24.1, shaft_width_mm=None)
    with patch("src.contradiction.contradiction_check.extract_parameters",
               return_value=_mock_drawing_result()), \
         patch("src.contradiction.contradiction_check.extract_spec_parameters",
               return_value=partially_readable_spec):
        result = check_contradictions(Path("fake_drawing.pdf"), Path("fake_spec.pdf"))

    assert result.spec_extraction_uncertain is False, (
        "one field extracted successfully is enough to not treat this as a total read failure"
    )


def test_clean_synthetic_style_match_is_not_flagged_uncertain():
    clean_spec = MagicMock(total_travel_m=24.1, shaft_width_mm=1500)
    with patch("src.contradiction.contradiction_check.extract_parameters",
               return_value=_mock_drawing_result()), \
         patch("src.contradiction.contradiction_check.extract_spec_parameters",
               return_value=clean_spec):
        result = check_contradictions(Path("fake_drawing.pdf"), Path("fake_spec.pdf"))

    assert result.spec_extraction_uncertain is False
    assert result.has_contradiction is False
    assert result.flags == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
