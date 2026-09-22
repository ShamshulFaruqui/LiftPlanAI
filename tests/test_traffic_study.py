"""
Tests for the CIBSE Guide D uppeak traffic study engine (Stage 4a).

No published numeric worked example was available to validate against
(see the module's docstring), so these tests focus on two things instead:
mathematical correctness of each sub-formula against manual calculation,
and directional/monotonicity consistency (e.g. more lifts should always
reduce interval) across the full calculation chain — which would catch
sign errors, swapped terms, or unit-conversion mistakes even without a
single "known-correct" reference number to compare against.

Run with: pytest tests/test_traffic_study.py -v
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.traffic_engine.traffic_study import (
    single_floor_flight_time, probable_stops, highest_reversal_floor,
    passengers_per_trip, round_trip_time, interval_seconds,
    handling_capacity_percent, estimate_population, run_traffic_study,
)


# --- Kinematics ---

def test_flight_time_trapezoidal_case_matches_manual_calculation():
    # Tall floor: car reaches rated speed and cruises briefly.
    # v=2 m/s, a=1 m/s^2, h=10m -> distance to reach speed = v^2/a = 4m,
    # cruise = 10-4 = 6m, cruise_time = 6/2 = 3s, accel+decel time = 2*(2/1) = 4s
    tv = single_floor_flight_time(floor_height_m=10, rated_speed_ms=2, acceleration_ms2=1)
    assert abs(tv - 7.0) < 0.01


def test_flight_time_triangular_case_matches_manual_calculation():
    # Short floor: car never reaches rated speed.
    # v=2 m/s, a=1 m/s^2, h=2m -> distance to reach speed = 4m > h, so triangular.
    # time = 2*sqrt(h/a) = 2*sqrt(2) = 2.828...
    tv = single_floor_flight_time(floor_height_m=2, rated_speed_ms=2, acceleration_ms2=1)
    assert abs(tv - 2 * math.sqrt(2)) < 0.01


def test_flight_time_increases_with_floor_height():
    tv_short = single_floor_flight_time(2.5, 1.75, 1.0)
    tv_tall = single_floor_flight_time(4.5, 1.75, 1.0)
    assert tv_tall > tv_short


def test_flight_time_decreases_with_higher_speed_for_tall_floor():
    # Only true once the floor is tall enough to benefit from higher speed
    # (a very short floor may never reach either speed, in which case
    # higher rated speed doesn't help — tested separately below).
    tv_slow = single_floor_flight_time(6.0, 1.0, 1.0)
    tv_fast = single_floor_flight_time(6.0, 2.5, 1.0)
    assert tv_fast < tv_slow


# --- Probable stops (S) and highest reversal floor (H) ---

def test_probable_stops_zero_when_no_passengers():
    assert probable_stops(15, 0) == 0.0


def test_probable_stops_increases_with_passengers():
    s_low = probable_stops(15, 4)
    s_high = probable_stops(15, 12)
    assert s_high > s_low


def test_probable_stops_never_exceeds_floor_count():
    s = probable_stops(15, 100)  # unrealistically high passenger count
    assert s <= 15


def test_probable_stops_approaches_floor_count_as_passengers_grow():
    # With very many passengers, almost every floor gets called.
    s = probable_stops(10, 200)
    assert s > 9.9


def test_highest_reversal_floor_zero_when_no_passengers():
    assert highest_reversal_floor(15, 0) == 0.0


def test_highest_reversal_floor_increases_with_passengers():
    h_low = highest_reversal_floor(15, 4)
    h_high = highest_reversal_floor(15, 12)
    assert h_high > h_low


def test_highest_reversal_floor_approaches_floor_count_with_many_passengers():
    h = highest_reversal_floor(10, 200)
    assert h > 9.9


def test_highest_reversal_floor_never_exceeds_floor_count():
    h = highest_reversal_floor(15, 100)
    assert h <= 15


# --- Car loading ---

def test_passengers_per_trip_is_80_percent_of_rated_capacity():
    # 1000kg / 75kg-per-person = 13 rated persons; 80% of 13 = 10.4 -> 10
    assert passengers_per_trip(1000) == 10


def test_passengers_per_trip_at_least_one():
    assert passengers_per_trip(50) >= 1  # even a tiny rated load shouldn't give zero


def test_passengers_per_trip_increases_with_rated_load():
    assert passengers_per_trip(1150) > passengers_per_trip(630)


# --- Round trip time, interval, handling capacity ---

def test_round_trip_time_matches_manual_formula():
    # RTT = 2*H*tv + (S+1)*ts + 2*P*tp
    rtt = round_trip_time(highest_reversal_floor_h=10, flight_time_s=2.0,
                           probable_stops_s=6, door_time_per_stop_s=8,
                           passengers=10, transfer_time_per_passenger_s=1.1)
    expected = 2 * 10 * 2.0 + (6 + 1) * 8 + 2 * 10 * 1.1
    assert abs(rtt - expected) < 0.01


def test_round_trip_time_increases_with_more_floors():
    result_low = run_traffic_study(num_floors=5, population=100, num_lifts=2,
                                    rated_speed_ms=1.5, rated_load_kg=1000, floor_height_m=3.2)
    result_high = run_traffic_study(num_floors=25, population=100, num_lifts=2,
                                     rated_speed_ms=1.5, rated_load_kg=1000, floor_height_m=3.2)
    assert result_high.round_trip_time_s > result_low.round_trip_time_s


def test_round_trip_time_decreases_with_higher_speed():
    result_slow = run_traffic_study(num_floors=15, population=200, num_lifts=3,
                                     rated_speed_ms=1.0, rated_load_kg=1000, floor_height_m=3.2)
    result_fast = run_traffic_study(num_floors=15, population=200, num_lifts=3,
                                     rated_speed_ms=2.5, rated_load_kg=1000, floor_height_m=3.2)
    assert result_fast.round_trip_time_s < result_slow.round_trip_time_s


def test_interval_equals_rtt_divided_by_lift_count():
    assert interval_seconds(120.0, 4) == 30.0


def test_interval_decreases_as_lift_count_increases():
    # Same RTT-determining inputs, only lift count changes.
    r3 = run_traffic_study(num_floors=15, population=300, num_lifts=3,
                            rated_speed_ms=1.75, rated_load_kg=1150, floor_height_m=3.2)
    r6 = run_traffic_study(num_floors=15, population=300, num_lifts=6,
                            rated_speed_ms=1.75, rated_load_kg=1150, floor_height_m=3.2)
    r8 = run_traffic_study(num_floors=15, population=300, num_lifts=8,
                            rated_speed_ms=1.75, rated_load_kg=1150, floor_height_m=3.2)
    assert r3.round_trip_time_s == r6.round_trip_time_s == r8.round_trip_time_s  # RTT is per-lift, unaffected by group size
    assert r3.interval_s > r6.interval_s > r8.interval_s
    # A well-elevatored building (8 lifts here) should land in the
    # commonly cited "good service" interval band.
    assert 15 <= r8.interval_s <= 35


def test_handling_capacity_decreases_with_more_population():
    hc_small = handling_capacity_percent(passengers=10, num_lifts=4, rtt_s=120, building_population=150)
    hc_large = handling_capacity_percent(passengers=10, num_lifts=4, rtt_s=120, building_population=600)
    assert hc_large < hc_small


def test_handling_capacity_increases_with_more_lifts():
    hc_few = handling_capacity_percent(passengers=10, num_lifts=2, rtt_s=120, building_population=300)
    hc_many = handling_capacity_percent(passengers=10, num_lifts=6, rtt_s=120, building_population=300)
    assert hc_many > hc_few


def test_handling_capacity_zero_when_no_population():
    assert handling_capacity_percent(10, 4, 120, 0) == 0.0


# --- Population estimator (rough placeholder) ---

def test_estimate_population_is_positive_and_scales_with_floors():
    p10 = estimate_population(10, "residential")
    p20 = estimate_population(20, "residential")
    assert p10 > 0
    assert p20 == 2 * p10


def test_estimate_population_falls_back_for_unknown_building_type():
    # Should not crash or return zero for a type not in the lookup table.
    p = estimate_population(10, "some-unlisted-type")
    assert p > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
