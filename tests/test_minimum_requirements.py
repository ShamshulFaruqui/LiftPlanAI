"""
Tests for the minimum requirements (pit depth / headroom) proposer.
Run with: pytest tests/test_minimum_requirements.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.traffic_engine.minimum_requirements import (
    propose_pit_depth_mm, propose_headroom_mm, propose_minimum_requirements,
    KONE_PIT_DEPTH_BY_SPEED_MM, KONE_HEADROOM_CLEARANCE_MM,
)


# --- Pit depth: exact published points ---

def test_pit_depth_at_exact_published_point_1_0():
    result = propose_pit_depth_mm(1.00)
    assert result["proposed_pit_depth_mm"] == 1050
    assert result["is_proposed_not_extracted"] is True


def test_pit_depth_at_exact_published_point_1_6():
    result = propose_pit_depth_mm(1.60)
    assert result["proposed_pit_depth_mm"] == 1200


def test_pit_depth_at_exact_published_point_1_75():
    result = propose_pit_depth_mm(1.75)
    assert result["proposed_pit_depth_mm"] == 1550


# --- Pit depth: interpolation ---

def test_pit_depth_interpolates_at_midpoint():
    # Exactly halfway between 1.0m/s (1050mm) and 1.6m/s (1200mm)
    # should give exactly the midpoint value.
    result = propose_pit_depth_mm(1.30)
    assert result["proposed_pit_depth_mm"] == 1125
    assert "interpolated" in result["note"].lower()


def test_pit_depth_below_lowest_point_uses_lowest_value_directly():
    result = propose_pit_depth_mm(0.5)
    assert result["proposed_pit_depth_mm"] == 1050
    assert "not interpolated" in result["note"].lower()


def test_pit_depth_above_highest_point_uses_highest_value_directly():
    result = propose_pit_depth_mm(3.0)
    assert result["proposed_pit_depth_mm"] == 1550
    assert "not interpolated" in result["note"].lower()


def test_pit_depth_increases_monotonically_with_speed():
    # A basic sanity check: faster lifts should never need a shallower pit.
    speeds = [0.5, 1.0, 1.2, 1.4, 1.6, 1.7, 1.75, 2.0]
    depths = [propose_pit_depth_mm(s)["proposed_pit_depth_mm"] for s in speeds]
    assert depths == sorted(depths)


# --- Headroom ---

def test_headroom_uses_provided_car_height():
    result = propose_headroom_mm(car_height_mm=2100)
    assert result["proposed_headroom_mm"] == 2100 + KONE_HEADROOM_CLEARANCE_MM
    assert result["car_height_was_assumed"] is False


def test_headroom_falls_back_to_default_when_car_height_unknown():
    result = propose_headroom_mm(car_height_mm=None)
    assert result["car_height_was_assumed"] is True
    assert result["proposed_headroom_mm"] > 0


def test_headroom_note_discloses_the_real_published_range():
    # The proposed single number must not be presented as more precise
    # than the source actually is — the note should disclose the real
    # range (CH+1300 to CH+1580mm) this is an approximation of.
    result = propose_headroom_mm(car_height_mm=2200)
    assert "1300" in result["note"] and "1580" in result["note"]


# --- Combined proposal ---

def test_combined_proposal_includes_both():
    result = propose_minimum_requirements(rated_speed_ms=1.6, car_height_mm=2200)
    assert "pit_depth" in result and "headroom" in result
    assert result["pit_depth"]["proposed_pit_depth_mm"] == 1200
    assert result["headroom"]["proposed_headroom_mm"] == 2200 + KONE_HEADROOM_CLEARANCE_MM


def test_all_proposals_flagged_as_proposed_not_extracted():
    # Every value returned must carry this flag — these are engineering
    # estimates, never to be presented as facts extracted from a
    # specific drawing.
    result = propose_minimum_requirements(rated_speed_ms=1.2)
    assert result["pit_depth"]["is_proposed_not_extracted"] is True
    assert result["headroom"]["is_proposed_not_extracted"] is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
