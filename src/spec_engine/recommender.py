"""
Specification Recommender — Stage 4b of the LiftPlan AI pipeline.

Searches for a lift configuration (number of cars, speed, rated load) that
meets published CIBSE Guide D / industry performance targets for a given
building, using the traffic study engine (Stage 4a) as the evaluation
function for each candidate configuration.

WHY THIS IS RULE-BASED SEARCH, NOT THE XGBOOST MODEL THE PROPOSAL
DESCRIBES: the proposal's Methods section specifies an XGBoost
recommender "trained on historical project data." No real historical
specification data exists in this project — only the synthetic dataset,
whose rated_load_kg/rated_speed_ms values were assigned by
generate_dataset.py's pick_load_speed() function using a loose,
somewhat-arbitrary weighting (see that module), not by any genuine
traffic-adequacy criterion. Training a supervised model on those labels
would teach it to reproduce that arbitrary assignment logic, not real
engineering judgement — a misleading result dressed up as machine
learning. A rule-based search grounded in the validated traffic study
engine (Stage 4a) is the honest choice available now, and — usefully —
its outputs constitute genuinely meaningful labels that a future
supervised model COULD be trained on, once enough configurations have
been generated this way (or once real historical data is obtained). This
is a deliberate, documented scope decision, not an oversight.

Usage:
    python recommender.py --floors 15 --population 300 --floor-height 3.2 --building-type commercial
"""

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.traffic_engine.traffic_study import run_traffic_study, TrafficStudyResult


# ---------------------------------------------------------------------------
# Target performance criteria by building type. Sourced from published
# guidance where available; estimated/blended values are marked as such
# and should be treated with proportionally less confidence.
# ---------------------------------------------------------------------------
TARGET_CRITERIA = {
    # ISO 8100-32:2020 Table 2, which corresponds directly to CIBSE Guide D
    # Table 3.4/3.5 (residential design criteria).
    "residential": {"min_hc_percent": 6, "max_interval_s": 60, "source": "ISO 8100-32:2020 / CIBSE Guide D Table 3.4-3.5"},
    # CIBSE Guide D + British Council for Offices (BCO) 2009 guidelines:
    # ~12% handling capacity, 30s average interval for conventional control.
    "commercial": {"min_hc_percent": 12, "max_interval_s": 30, "source": "CIBSE Guide D / BCO 2009 guidelines"},
    # Interval figure per ISO 8100-32:2020's correction to CIBSE Guide D
    # for midrange hotels (60s, not the commonly mis-cited 40s). Handling
    # capacity target not directly found in this session's sources —
    # ESTIMATED by interpolating between residential and office criteria.
    "hotel": {"min_hc_percent": 8, "max_interval_s": 60, "source": "ISO 8100-32:2020 (interval); HC% estimated, not directly sourced"},
    # Not directly found in this session's sources — ESTIMATED, pending a
    # direct CIBSE Guide D citation. Treat with more caution than the
    # residential/commercial figures above.
    "hospital": {"min_hc_percent": 10, "max_interval_s": 45, "source": "ESTIMATED — not directly sourced this session"},
    # Blended estimate between residential and commercial criteria.
    "mixed-use": {"min_hc_percent": 10, "max_interval_s": 40, "source": "ESTIMATED — blended residential/commercial"},
    # Warehouses are not a typical CIBSE Guide D worked example — traffic
    # is light and mostly goods-focused, not passenger up-peak in the
    # conventional office/residential sense. ESTIMATED as a deliberately
    # relaxed target reflecting low, infrequent passenger demand.
    "warehouse": {"min_hc_percent": 5, "max_interval_s": 90, "source": "ESTIMATED — no formal source for this building type"},
    # A villa is a single household, not a formal traffic-engineering
    # design case at all — CIBSE Guide D doesn't address single-family
    # homes. ESTIMATED as a very relaxed target purely so the recommender
    # has something to search against; in practice a villa lift is sized
    # for the specific family's needs, not a population-based calculation.
    "villa": {"min_hc_percent": 3, "max_interval_s": 120, "source": "ESTIMATED — not a formal CIBSE design case; single-household use"},
}

# Search space, grounded in the KONE MonoSpace 500 envelope already used
# for the synthetic dataset (see src/synthetic/generate_dataset.py).
SPEED_OPTIONS_MS = [1.00, 1.25, 1.50, 1.75]
LOAD_OPTIONS_KG = [630, 800, 1000, 1150]
MAX_LIFTS_SEARCHED = 8


@dataclass
class Recommendation:
    feasible: bool
    num_lifts: int = None
    rated_speed_ms: float = None
    rated_load_kg: int = None
    traffic_result: TrafficStudyResult = None
    target_criteria: dict = None
    message: str = ""


def configuration_cost_proxy(num_lifts: int, speed: float, load: int) -> tuple:
    """
    A simple, defensible ordering for preferring one adequate configuration
    over another: fewer lifts first (each additional lift means another
    shaft — the largest cost and space driver), then lower speed, then
    lower load (both cheaper equipment, all else equal). This is a proxy,
    not a real cost model — a genuine cost comparison would need actual
    unit and installation pricing, which is out of scope here.
    """
    return (num_lifts, speed, load)


def recommend_configuration(num_floors: int, population: int, floor_height_m: float,
                             building_type: str, acceleration_ms2: float = 1.0) -> Recommendation:
    """
    Search the (lifts x speed x load) space for the cheapest configuration
    (by the proxy above) that meets the building type's target handling
    capacity and interval. Returns the first-found best configuration, or
    a Recommendation with feasible=False if nothing in the search space
    meets the targets — itself a valid engineering conclusion (the
    building may need to be split into multiple lift zones/groups, which
    this single-group search does not attempt).
    """
    criteria = TARGET_CRITERIA.get(building_type, TARGET_CRITERIA["mixed-use"])

    candidates = []
    for num_lifts in range(1, MAX_LIFTS_SEARCHED + 1):
        for speed, load in itertools.product(SPEED_OPTIONS_MS, LOAD_OPTIONS_KG):
            result = run_traffic_study(
                num_floors=num_floors, population=population, num_lifts=num_lifts,
                rated_speed_ms=speed, rated_load_kg=load, floor_height_m=floor_height_m,
                acceleration_ms2=acceleration_ms2,
            )
            meets_hc = result.handling_capacity_percent >= criteria["min_hc_percent"]
            meets_interval = result.interval_s <= criteria["max_interval_s"]
            if meets_hc and meets_interval:
                candidates.append((configuration_cost_proxy(num_lifts, speed, load),
                                    num_lifts, speed, load, result))

    if not candidates:
        return Recommendation(
            feasible=False,
            target_criteria=criteria,
            message=(
                f"No configuration within the search space (up to {MAX_LIFTS_SEARCHED} lifts, "
                f"speeds {SPEED_OPTIONS_MS}, loads {LOAD_OPTIONS_KG}) meets the target "
                f"({criteria['min_hc_percent']}% handling capacity, {criteria['max_interval_s']}s "
                f"interval) for this building. This may indicate the building needs to be split "
                f"into multiple lift zones/groups rather than served by one group — a single-group "
                f"search cannot resolve that, and this should be flagged for engineer review."
            ),
        )

    candidates.sort(key=lambda c: c[0])
    _, num_lifts, speed, load, result = candidates[0]

    return Recommendation(
        feasible=True,
        num_lifts=num_lifts,
        rated_speed_ms=speed,
        rated_load_kg=load,
        traffic_result=result,
        target_criteria=criteria,
        message=f"Recommended: {num_lifts} lift(s) at {speed}m/s, {load}kg rated load.",
    )


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Stage 4b: Specification Recommender")
    parser.add_argument("--floors", type=int, required=True)
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument("--floor-height", type=float, required=True)
    parser.add_argument("--building-type", type=str, required=True, choices=list(TARGET_CRITERIA.keys()))
    args = parser.parse_args()

    rec = recommend_configuration(args.floors, args.population, args.floor_height, args.building_type)
    print(rec.message)
    if rec.feasible:
        print(f"\nTarget criteria ({rec.target_criteria['source']}):")
        print(f"  Min handling capacity: {rec.target_criteria['min_hc_percent']}%")
        print(f"  Max interval: {rec.target_criteria['max_interval_s']}s")
        print(f"\nAchieved performance:")
        for field, value in rec.traffic_result.__dict__.items():
            print(f"  {field}: {value}")


if __name__ == "__main__":
    main()
