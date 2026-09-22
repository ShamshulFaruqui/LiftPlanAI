"""
Traffic Study Engine — Stage 4a of the LiftPlan AI pipeline.

Implements the conventional up-peak Round Trip Time (RTT) method for
elevator traffic analysis, as described in CIBSE Guide D and the wider
vertical transportation literature (Barney & dos Santos; Strakosch;
Al-Sharif et al.). This is pure engineering calculation — no ML — which
is exactly why it can be built and validated before real drawing data is
available: the physics and probability formulas are fixed and testable
independent of any dataset.

IMPORTANT — validation status: the core formula (RTT = 2H*tv + (S+1)*ts +
2P*tp) is corroborated by an independent peer-reviewed source citing
CIBSE Guide D directly (Y. Jiang et al., 'Calculation of the elevator
round-trip time under destination group control', ScienceDirect, 2019 —
see the project's literature review for the full citation). However, an
exact published NUMERIC worked example to validate this implementation's
output figures against could not be retrieved in this session (the
Scribd and HKU teaching-notes sources found were blocked by bot
protection). This module has therefore been validated through extensive
internal consistency checks (see tests/test_traffic_study.py) rather
than against a known-correct published numeric result. Before relying on
this for a real specification, cross-check a sample calculation against
the actual CIBSE Guide D document or a validated commercial tool.

Simplifications made (documented, not hidden):
- Kinematics use a simple constant-acceleration model (no jerk-limited
  S-curve profile). CIBSE Guide D itself notes the conventional formula
  makes this same simplifying assumption and offers "enhanced" corrected
  formulae for jerk-limited motion — treated here as a stated limitation
  and a natural next step, not attempted in this module.
- Building population is not derived from the extracted drawing
  parameters (floor area / unit counts aren't currently extracted) — see
  estimate_population() for a clearly-labelled rough placeholder that
  should be replaced with real occupancy data before production use.

Usage:
    python traffic_study.py --floors 15 --population 300 --speed 1.75 \\
        --load 1150 --floor-height 3.2
"""

import argparse
import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def single_floor_flight_time(floor_height_m: float, rated_speed_ms: float,
                              acceleration_ms2: float = 1.0) -> float:
    """
    Time for the car to travel exactly one floor at rated acceleration,
    under a simple constant-acceleration (non jerk-limited) model.

    Two cases, from standard kinematics:
    - If the floor is tall enough for the car to reach rated speed and
      still cruise before decelerating (a trapezoidal velocity profile),
      time = 2*(v/a) + (floor_height - v^2/a) / v
    - If the floor is too short to reach rated speed at all (a triangular
      velocity profile, peaking partway up), time = 2 * sqrt(floor_height / a)
    """
    v, a, h = rated_speed_ms, acceleration_ms2, floor_height_m
    distance_to_reach_rated_speed = v ** 2 / a

    if h >= distance_to_reach_rated_speed:
        cruise_distance = h - distance_to_reach_rated_speed
        return 2 * (v / a) + cruise_distance / v
    else:
        return 2 * math.sqrt(h / a)


# ---------------------------------------------------------------------------
# Probability-based estimates: probable stops (S) and highest reversal
# floor (H). Both derive from treating each of P passengers' destination
# floors as an independent uniform random choice among N floors above the
# main terminal — the standard assumption in the conventional method.
# ---------------------------------------------------------------------------

def probable_stops(num_floors: int, passengers_per_trip: int) -> float:
    """
    Expected number of distinct floors called, out of num_floors floors
    above the main terminal, when passengers_per_trip passengers each
    independently choose a destination floor uniformly at random.

    S = N * [1 - (1 - 1/N)^P]
    """
    if num_floors <= 0 or passengers_per_trip <= 0:
        return 0.0
    n, p = num_floors, passengers_per_trip
    return n * (1 - (1 - 1 / n) ** p)


def highest_reversal_floor(num_floors: int, passengers_per_trip: int) -> float:
    """
    Expected highest floor reached before the car reverses, under the
    same uniform-random-destination assumption used for probable_stops.

    H = N - sum_{j=1}^{N-1} (j/N)^P
    """
    if num_floors <= 0 or passengers_per_trip <= 0:
        return 0.0
    n, p = num_floors, passengers_per_trip
    return n - sum((j / n) ** p for j in range(1, n))


# ---------------------------------------------------------------------------
# Car loading
# ---------------------------------------------------------------------------

def passengers_per_trip(rated_load_kg: int, person_weight_kg: float = 75.0,
                         loading_factor: float = 0.8) -> int:
    """
    Number of passengers assumed to board per up-peak trip. Rated capacity
    in persons uses the standard EN81 divisor of 75kg/person; up-peak
    boarding is conventionally taken as ~80% of rated capacity, since cars
    are rarely observed to fill completely in practice.
    """
    rated_persons = math.floor(rated_load_kg / person_weight_kg)
    return max(1, round(rated_persons * loading_factor))


# ---------------------------------------------------------------------------
# Round trip time, interval, and handling capacity
# ---------------------------------------------------------------------------

def round_trip_time(highest_reversal_floor_h: float, flight_time_s: float,
                     probable_stops_s: float, door_time_per_stop_s: float,
                     passengers: int, transfer_time_per_passenger_s: float) -> float:
    """
    RTT = 2*H*tv + (S+1)*ts + 2*P*tp

    The (S+1) term accounts for the door cycle at every intermediate stop
    plus the final return to the main terminal. 2*P*tp accounts for
    loading time at the main terminal and unloading time distributed
    across the trip (the doubling reflects boarding + alighting).
    """
    return (2 * highest_reversal_floor_h * flight_time_s +
            (probable_stops_s + 1) * door_time_per_stop_s +
            2 * passengers * transfer_time_per_passenger_s)


def interval_seconds(rtt_s: float, num_lifts: int) -> float:
    """Average time between successive car arrivals at the main terminal."""
    if num_lifts <= 0:
        return float("inf")
    return rtt_s / num_lifts


def handling_capacity_percent(passengers: int, num_lifts: int, rtt_s: float,
                               building_population: int) -> float:
    """
    Percentage of the building's population that can be moved in a
    5-minute (300s) up-peak period: HC% = (300 * P * L / RTT) / population * 100
    """
    if rtt_s <= 0 or building_population <= 0:
        return 0.0
    passengers_per_5min = (300 * passengers * num_lifts) / rtt_s
    return (passengers_per_5min / building_population) * 100


# ---------------------------------------------------------------------------
# Rough population estimate — a clearly-labelled placeholder, not
# authoritative. Real occupancy should come from architectural area
# schedules or MEP population calculations once the pipeline extracts
# them; this exists purely so the traffic study can run end-to-end on the
# synthetic dataset's existing fields (floor_count, building_type).
# ---------------------------------------------------------------------------

ROUGH_PERSONS_PER_FLOOR = {
    "residential": 4,
    "commercial": 40,
    "hotel": 20,
    "hospital": 15,
    "mixed-use": 25,
    # Mostly storage/racking, not occupied office space — a real
    # warehouse has far fewer people per floor than any office building,
    # reflecting staff/operations headcount, not floor-area occupancy.
    "warehouse": 8,
    # A villa's "population" is a single household, not a scaled
    # per-floor office-style density — 2 persons/floor gives a realistic
    # 4-8 total across the 2-4 floor range villas are generated in.
    "villa": 2,
}


def estimate_population(floor_count: int, building_type: str) -> int:
    """
    ROUGH placeholder estimate only — see module docstring. Replace with
    real occupancy data (from area schedules or MEP calculations) before
    using this for an actual specification.
    """
    per_floor = ROUGH_PERSONS_PER_FLOOR.get(building_type, 20)
    return floor_count * per_floor


@dataclass
class TrafficStudyResult:
    num_floors: int
    population: int
    passengers_per_trip: int
    flight_time_s: float
    probable_stops: float
    highest_reversal_floor: float
    round_trip_time_s: float
    interval_s: float
    handling_capacity_percent: float
    num_lifts: int


def run_traffic_study(num_floors: int, population: int, num_lifts: int,
                       rated_speed_ms: float, rated_load_kg: int,
                       floor_height_m: float, acceleration_ms2: float = 1.0,
                       door_time_per_stop_s: float = 8.0,
                       transfer_time_per_passenger_s: float = 1.1) -> TrafficStudyResult:
    """Run the full up-peak traffic study for one lift group."""
    p = passengers_per_trip(rated_load_kg)
    tv = single_floor_flight_time(floor_height_m, rated_speed_ms, acceleration_ms2)
    s = probable_stops(num_floors, p)
    h = highest_reversal_floor(num_floors, p)
    rtt = round_trip_time(h, tv, s, door_time_per_stop_s, p, transfer_time_per_passenger_s)
    interval = interval_seconds(rtt, num_lifts)
    hc = handling_capacity_percent(p, num_lifts, rtt, population)

    return TrafficStudyResult(
        num_floors=num_floors,
        population=population,
        passengers_per_trip=p,
        flight_time_s=round(tv, 2),
        probable_stops=round(s, 2),
        highest_reversal_floor=round(h, 2),
        round_trip_time_s=round(rtt, 1),
        interval_s=round(interval, 1),
        handling_capacity_percent=round(hc, 1),
        num_lifts=num_lifts,
    )


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Stage 4a: Traffic Study Engine")
    parser.add_argument("--floors", type=int, required=True, help="Floors served above main terminal")
    parser.add_argument("--population", type=int, required=True, help="Building population served")
    parser.add_argument("--lifts", type=int, default=1, help="Number of lifts in the group")
    parser.add_argument("--speed", type=float, required=True, help="Rated speed (m/s)")
    parser.add_argument("--load", type=int, required=True, help="Rated load (kg)")
    parser.add_argument("--floor-height", type=float, required=True, help="Floor-to-floor height (m)")
    args = parser.parse_args()

    result = run_traffic_study(
        num_floors=args.floors, population=args.population, num_lifts=args.lifts,
        rated_speed_ms=args.speed, rated_load_kg=args.load, floor_height_m=args.floor_height,
    )
    for field, value in result.__dict__.items():
        print(f"{field}: {value}")


if __name__ == "__main__":
    main()
