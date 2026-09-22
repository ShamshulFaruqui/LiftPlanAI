"""
Tests for the merged text+visual cross-validation extractor.
Run with: pytest tests/test_merged_extractor.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser.merged_extractor import compare_extractions, SHAFT_WIDTH_TOLERANCE_MM
from src.parser.text_extractor import ExtractedParameters, ExtractedGroup
from src.parser.visual_extractor import VisualParameters, VisualGroup


def make_text_result(group_widths):
    groups = [ExtractedGroup(group_id=f"G{i}", lift_type="passenger", shaft_width_mm=w)
              for i, w in enumerate(group_widths)]
    return ExtractedParameters(source_file="test", groups=groups)


def make_visual_result(shaft_widths):
    shafts = [VisualGroup(left_px=0, right_px=100, width_px=100, width_mm=w)
              for w in shaft_widths]
    return VisualParameters(source_file="test", shaft_count=len(shafts), shafts=shafts)


def test_matching_group_counts_and_widths_gives_high_confidence():
    text = make_text_result([2100, 2400])
    visual = make_visual_result([2100, 2400])
    result = compare_extractions(text, visual)
    assert result.group_count_agrees is True
    assert result.confidence == "high"
    assert result.flags == []


def test_small_pixel_rounding_noise_within_tolerance_does_not_flag():
    # Mirrors the real, expected gap seen on clean synthetic data: text
    # states an exact value, visual measures a few mm off due to pixel
    # rounding. This should NOT be flagged as a genuine discrepancy.
    text = make_text_result([2100])
    visual = make_visual_result([2100 - (SHAFT_WIDTH_TOLERANCE_MM - 5)])  # just inside tolerance
    result = compare_extractions(text, visual)
    assert result.shaft_comparisons[0].agrees is True
    assert result.confidence == "high"


def test_genuine_width_discrepancy_beyond_tolerance_is_flagged():
    text = make_text_result([2100])
    visual = make_visual_result([2100 - (SHAFT_WIDTH_TOLERANCE_MM + 50)])  # well outside tolerance
    result = compare_extractions(text, visual)
    assert result.shaft_comparisons[0].agrees is False
    assert result.confidence == "review_needed"
    assert any("width mismatch" in f for f in result.flags)


def test_group_count_mismatch_is_flagged():
    text = make_text_result([2100, 2400])   # text found 2 groups
    visual = make_visual_result([2100])      # visual only found 1 shaft
    result = compare_extractions(text, visual)
    assert result.group_count_agrees is False
    assert result.confidence == "review_needed"
    assert any("Group count mismatch" in f for f in result.flags)


def test_group_count_mismatch_does_not_crash_width_comparison():
    # When counts differ, comparison should still run on the overlapping
    # portion without raising an index error.
    text = make_text_result([2100, 2400, 2000])
    visual = make_visual_result([2100])
    result = compare_extractions(text, visual)
    assert len(result.shaft_comparisons) == 1  # only compares as far as both have data
    assert result.confidence == "review_needed"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
