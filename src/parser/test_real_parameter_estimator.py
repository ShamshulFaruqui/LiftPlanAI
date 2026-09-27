"""
Tests for real_parameter_estimator.py — see its module docstring for the
real drawing (a warehouse project) whose civil-survey contamination
motivated these specific test cases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser.real_parameter_estimator import estimate_building_cluster


def test_clean_evenly_spaced_levels_estimate_correctly():
    elevs = [0, 3500, 7000, 10500, 14000]
    est = estimate_building_cluster(elevs)
    assert est is not None
    assert est.floor_height_m == 3.5
    assert est.num_floors == 5


def test_civil_survey_contamination_is_excluded_not_averaged_in():
    # The exact real-drawing case that motivated this module: a genuine
    # 8-reading office-block cluster near zero, plus a 9-reading (larger!)
    # civil-survey spot-height grid at 26800-28000mm. The larger cluster
    # is the wrong one to pick (see module docstring) - this asserts the
    # smaller, near-zero cluster is chosen instead, and that the result
    # is refused anyway once its own floor height turns out implausible
    # (the office block's readings are sub-floor annotations, not floors).
    elevs = [-300, -200, -50, 150, 250, 350, 500, 900,
             26800, 26900, 27100, 27250, 27350, 27450, 27550, 27650, 28000]
    est = estimate_building_cluster(elevs)
    assert est is None, "an implausible floor height must be refused, not passed through"


def test_too_few_readings_returns_none():
    assert estimate_building_cluster([0, 3500]) is None
    assert estimate_building_cluster([]) is None


def test_implausibly_small_gap_is_refused():
    # Sub-floor annotations (door thresholds, paving levels) a few
    # hundred mm apart - not real floor-to-floor heights.
    elevs = [0, 150, 300, 450, 600]
    assert estimate_building_cluster(elevs) is None


def test_implausibly_large_gap_is_refused():
    elevs = [0, 20000, 40000, 60000]
    assert estimate_building_cluster(elevs) is None


def test_duplicate_readings_are_deduplicated():
    # A level mentioned twice on a drawing (e.g. front and rear openings
    # at the same elevation) should not distort the gap calculation.
    elevs = [0, 0, 3500, 3500, 7000]
    est = estimate_building_cluster(elevs)
    assert est is not None
    assert est.num_floors == 3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
