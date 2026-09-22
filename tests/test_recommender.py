"""
Tests for the rule-based specification recommender (Stage 4b).
Run with: pytest tests/test_recommender.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spec_engine.recommender import (
    recommend_configuration, configuration_cost_proxy, TARGET_CRITERIA,
)


def test_small_residential_building_finds_feasible_configuration():
    rec = recommend_configuration(num_floors=8, population=40, floor_height_m=3.0,
                                   building_type="residential")
    assert rec.feasible is True
    assert rec.num_lifts >= 1


def test_recommended_configuration_actually_meets_its_own_targets():
    # The whole point of the search — confirm the returned configuration's
    # own traffic study result satisfies the criteria it was selected against,
    # not just that *a* configuration was returned.
    rec = recommend_configuration(num_floors=15, population=300, floor_height_m=3.2,
                                   building_type="commercial")
    assert rec.feasible is True
    criteria = TARGET_CRITERIA["commercial"]
    assert rec.traffic_result.handling_capacity_percent >= criteria["min_hc_percent"]
    assert rec.traffic_result.interval_s <= criteria["max_interval_s"]


def test_harder_building_requires_more_lifts_than_easier_one():
    easy = recommend_configuration(num_floors=6, population=60, floor_height_m=3.0,
                                    building_type="residential")
    hard = recommend_configuration(num_floors=20, population=500, floor_height_m=3.9,
                                    building_type="commercial")
    assert easy.feasible and hard.feasible
    assert hard.num_lifts > easy.num_lifts


def test_extreme_building_correctly_reports_infeasible():
    # A very tall, very populated building should exceed what 8 lifts at
    # the top of the search space's speed/load range can serve — the
    # search should say so clearly rather than silently returning a
    # configuration that doesn't actually meet the targets.
    rec = recommend_configuration(num_floors=22, population=2000, floor_height_m=4.0,
                                   building_type="commercial")
    assert rec.feasible is False
    assert rec.num_lifts is None
    assert "split" in rec.message.lower() or "zone" in rec.message.lower()


def test_unknown_building_type_falls_back_without_crashing():
    rec = recommend_configuration(num_floors=10, population=100, floor_height_m=3.2,
                                   building_type="some-unlisted-type")
    # Should fall back to the mixed-use criteria rather than raising.
    assert rec.target_criteria == TARGET_CRITERIA["mixed-use"]


def test_cost_proxy_prefers_fewer_lifts_above_all_else():
    cheap = configuration_cost_proxy(num_lifts=2, speed=1.75, load=1150)  # fewer lifts, but faster/bigger
    expensive = configuration_cost_proxy(num_lifts=3, speed=1.00, load=630)  # more lifts, but slower/smaller
    assert cheap < expensive  # lift count dominates the ordering


def test_cost_proxy_prefers_lower_speed_when_lift_count_ties():
    slower = configuration_cost_proxy(num_lifts=4, speed=1.00, load=1150)
    faster = configuration_cost_proxy(num_lifts=4, speed=1.75, load=1150)
    assert slower < faster


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
