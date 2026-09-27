"""
Tests for real_lift_specification_extractor.py.

The parsing logic (_collect_label_blocks, _values_from_block, and the
merge-identical-groups step) is tested directly against synthetic OCR-text
fixtures, since that's the actual unit of correctness here — OCR quality
itself isn't something a unit test can verify, and the module's own
docstring is explicit that it has been manually verified against one real
drawing (see the docstring and the dissertation's Chapter 6 for that
verification). These tests cover the parsing/grouping logic in isolation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser.real_lift_specification_extractor import (
    FIELD_LABELS,
    _collect_label_blocks,
    _values_from_block,
    extract_lift_group_specs,
)


def _canonical_by_label():
    out = {}
    for canon, labels in FIELD_LABELS.items():
        for lab in labels:
            out[lab] = canon
    return out


def test_collect_label_blocks_groups_multiline_values_under_their_label():
    lines = [
        "LIFT DESIGNATION",
        "PL-01 ~ PL-04",
        "PL-05 ~ PL-07",
        "MU01",
        "USAGE",
        "PASSENGER",
        "MULTIUTILITY",
    ]
    blocks = _collect_label_blocks(lines, _canonical_by_label())
    assert blocks[0] == ("lift_designation", ["PL-01 ~ PL-04", "PL-05 ~ PL-07", "MU01"])
    assert blocks[1] == ("usage", ["PASSENGER", "MULTIUTILITY"])


def test_collect_label_blocks_handles_same_line_label_and_value():
    lines = ["CAPACITY 1350 Kgs / 18 PERSONS 1600 Kgs / 21 PERSONS", "SPEED 3.5 MPS 3.5 MPS"]
    blocks = _collect_label_blocks(lines, _canonical_by_label())
    assert blocks[0][0] == "capacity"
    assert blocks[1][0] == "speed"


def test_values_from_block_uses_shape_pattern_when_whitespace_split_fails():
    # Single OCR line, single space between the two cells (the real case
    # that broke a pure whitespace-gap split during development).
    values = _values_from_block(
        ["1350 Kgs / 18 PERSONS 1600 Kgs / 21 PERSONS"], n_groups=2,
        is_level_field=False, canon="capacity",
    )
    assert values == ["1350 Kgs / 18 PERSONS", "1600 Kgs / 21 PERSONS"]


def test_values_from_block_one_value_per_line():
    values = _values_from_block(
        ["PL-01 ~ PL-04", "PL-05 ~ PL-07", "MU01"], n_groups=3, is_level_field=False,
    )
    assert values == ["PL-01 ~ PL-04", "PL-05 ~ PL-07", "MU01"]


def test_values_from_block_empty_input_returns_empty():
    assert _values_from_block([], n_groups=3, is_level_field=False) == []
    assert _values_from_block(["   ", ""], n_groups=3, is_level_field=False) == []


def test_extract_lift_group_specs_returns_empty_list_when_no_spec_table(tmp_path):
    # A PDF with no "BRIEF SPECIFICATION" heading at all in its text layer.
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "FLOOR PLAN - LEVEL 3")
    pdf_path = tmp_path / "plain_floor_plan.pdf"
    doc.save(str(pdf_path))
    doc.close()

    assert extract_lift_group_specs(pdf_path) == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
