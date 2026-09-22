"""
Tests for the spatial (OCR bounding-box) dimension extractor.
Uses constructed word-box data rather than requiring real OCR, keeping
tests fast and independent of Tesseract/Poppler being installed — the
underlying association LOGIC is what's being tested here, not OCR
accuracy itself (which was verified manually against a real drawing
during development — see module docstring for the confirmed real
result: three independent "SHAFT WIDTH" labels all converging on the
same 2550mm value).

Run with: pytest tests/test_spatial_dimension_extractor.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser.spatial_dimension_extractor import (
    find_label_positions, find_nearest_plausible_number, PLAUSIBLE_RANGES_MM,
)


def word(text, left, top, width=40, height=20):
    return {"text": text, "left": left, "top": top, "width": width, "height": height}


# --- find_label_positions ---

def test_finds_adjacent_multiword_label():
    words = [word("SHAFT", 100, 200), word("WIDTH", 150, 200)]
    positions = find_label_positions(words, ["SHAFT", "WIDTH"])
    assert positions == [(100, 200)]


def test_does_not_match_shaft_without_width_nearby():
    words = [word("SHAFT", 100, 200), word("DEPTH", 150, 200)]
    positions = find_label_positions(words, ["SHAFT", "WIDTH"])
    assert positions == []


def test_finds_multiple_label_occurrences():
    words = [
        word("SHAFT", 100, 200), word("WIDTH", 150, 200),
        word("SHAFT", 500, 800), word("WIDTH", 550, 800),
    ]
    positions = find_label_positions(words, ["SHAFT", "WIDTH"])
    assert len(positions) == 2


# --- find_nearest_plausible_number: the core association logic ---

def test_picks_horizontally_aligned_number_over_closer_but_misaligned_one():
    # A number that's numerically CLOSER in raw distance but poorly
    # X-aligned should lose to one that's further but better X-aligned —
    # this is the real pattern confirmed against the actual drawing
    # (dimension text sits X-aligned with its label, offset vertically).
    words = [
        word("2400", left=300, top=205),  # small total distance, poor X-alignment (200px off)
        word("2550", left=105, top=400),  # larger vertical distance, but only 5px X-offset
    ]
    match = find_nearest_plausible_number(words, label_pos=(100, 200),
                                           plausible_range=PLAUSIBLE_RANGES_MM["shaft_width"])
    assert match["value"] == 2550


def test_rejects_implausible_values_outside_range():
    words = [word("50000", left=105, top=210)]  # way outside plausible shaft width range
    match = find_nearest_plausible_number(words, label_pos=(100, 200),
                                           plausible_range=PLAUSIBLE_RANGES_MM["shaft_width"])
    assert match is None


def test_rejects_numbers_too_far_away():
    words = [word("2550", left=100, top=5000)]  # far outside the search window
    match = find_nearest_plausible_number(words, label_pos=(100, 200),
                                           plausible_range=PLAUSIBLE_RANGES_MM["shaft_width"])
    assert match is None


def test_no_candidates_returns_none():
    words = [word("HALL BUTTON", left=100, top=210)]  # not numeric at all
    match = find_nearest_plausible_number(words, label_pos=(100, 200),
                                           plausible_range=PLAUSIBLE_RANGES_MM["shaft_width"])
    assert match is None


def test_convergent_real_world_case():
    # Mirrors the actual confirmed result: three separate label positions
    # on a real drawing, each independently resolving to the same 2550mm
    # value with small horizontal offsets (67px, 3px, 2px).
    words = [
        word("2550", left=2790, top=2998),
        word("2550", left=3109, top=2998),
        word("2550", left=3427, top=3013),
    ]
    label_positions = [(2857, 2974), (3106, 3037), (3425, 3020)]
    values = [
        find_nearest_plausible_number(words, pos, PLAUSIBLE_RANGES_MM["shaft_width"])["value"]
        for pos in label_positions
    ]
    assert values == [2550, 2550, 2550]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


# --- car_depth: new field added this session, confirmed against the real drawing ---

def test_car_depth_uses_its_own_plausible_range():
    assert "car_depth" in PLAUSIBLE_RANGES_MM
    assert PLAUSIBLE_RANGES_MM["car_depth"] == (1200, 3500)


def test_car_depth_label_matches_car_and_depth_tokens():
    words = [word("CAR", 100, 200), word("DEPTH", 150, 200)]
    positions = find_label_positions(words, ["CAR", "DEPTH"])
    assert positions == [(100, 200)]


def test_car_depth_convergent_real_world_case():
    # Mirrors the actual confirmed result: two separate "CAR DEPTH" label
    # positions on the real drawing both independently resolved to the
    # same 1650mm value (offsets 21px and 20px).
    words = [
        word("1650", left=2702, top=2219),
        word("1650", left=3398, top=2219),
    ]
    label_positions = [(2681, 2243), (3378, 2243)]
    values = [
        find_nearest_plausible_number(words, pos, PLAUSIBLE_RANGES_MM["car_depth"])["value"]
        for pos in label_positions
    ]
    assert values == [1650, 1650]


def test_shaft_width_and_car_depth_do_not_cross_contaminate():
    # A "CAR" + "DEPTH" pair should not be picked up when searching for
    # "SHAFT" + "WIDTH", and vice versa — each label type's tokens are
    # independent.
    words = [word("CAR", 100, 200), word("DEPTH", 150, 200)]
    assert find_label_positions(words, ["SHAFT", "WIDTH"]) == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
