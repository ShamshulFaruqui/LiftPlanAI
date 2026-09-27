"""
Numerical validation of the traffic study engine (supervisor feedback:
"the traffic study engine relies on unverified internal assumptions due to
the lack of a numerical worked example").

Two independent checks:

1. PUBLISHED WORKED EXAMPLE - Khaleel, Al-Sharif & Salahat (2013),
   'Derivation of an Elevator Round Trip Time Formula under Up-peak
   Traffic for the Case of Four Special Conditions', 3rd Symposium on
   Lift and Escalator Technologies, Section 6. A 14-floor office building
   with unequal floor populations and heights, v = 4.0 m/s, a = 1.0 m/s2,
   j = 1.0 m/s3, P = 16.8. The paper reports S = 9.737, tD = 59.05 s,
   tP = 40.32 s, E(d_total) = 58.28 m, tH = 19.57 s and RTT = 176.33 s
   (Monte Carlo 176.377 s). The reference helpers below (general
   unequal-population S, expected travel distance, Peters' jerk-limited
   kinematics, and a Monte Carlo simulator) reproduce every one of these.

2. MONTE CARLO vs ENGINE - having validated the simulator against the
   published result, it is used as an independent oracle for the engine's
   conventional equal-floor equation on typical buildings.
"""

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.traffic_engine.traffic_study import (  # noqa: E402
    run_traffic_study, passengers_per_trip, probable_stops, highest_reversal_floor,
    single_floor_flight_time,
)


# ---------------------------------------------------------------------------
# Reference helpers (independent of the engine)
# ---------------------------------------------------------------------------
def t_jerk(d, v, a, j):
    """Peters (1996) ideal lift kinematics - time for a journey of d metres."""
    if d <= 0:
        return 0.0
    if d >= v * v / a + v * a / j:
        return d / v + v / a + a / j
    if d >= 2 * a ** 3 / j ** 2:
        return a / j + math.sqrt(4 * d / a + a * a / (j * j))
    return (32 * d / j) ** (1 / 3)


def t_const_acc(d, v, a):
    return 2 * math.sqrt(d / a) if d < v * v / a else d / v + v / a


PAPER = dict(
    pops=[150, 150] + [100] * 10 + [50, 50],          # L1..L14
    heights={0: 5.0, **{k: 4.5 for k in range(1, 11)}, **{k: 4.0 for k in range(11, 15)}},
    P=16.8, v=4.0, a=1.0, j=1.0, tdo=2.0, tdc=3.0, tsd=0.5, tpi=1.2, tpo=1.2,
)


def paper_components():
    pops, P, H_of = PAPER["pops"], PAPER["P"], PAPER["heights"]
    N, U = len(pops), sum(pops)
    S = N - sum((1 - u / U) ** P for u in pops)
    cum = [sum(pops[:i]) for i in range(N + 1)]
    Ed = sum(H_of[i] * (1 - (cum[i] / U) ** P) for i in range(N))
    tD = (S + 1) * (PAPER["tdo"] + PAPER["tdc"] + PAPER["tsd"])
    tP = P * (PAPER["tpi"] + PAPER["tpo"])
    tH = t_jerk(Ed, PAPER["v"], PAPER["a"], PAPER["j"])
    return S, Ed, tD, tP, tH


def paper_monte_carlo(runs=20000, seed=0):
    pops, H_of = PAPER["pops"], PAPER["heights"]
    N, U = len(pops), sum(pops)
    lvl = [0.0]
    for k in range(N):
        lvl.append(lvl[-1] + H_of[k])
    v, a, j = PAPER["v"], PAPER["a"], PAPER["j"]
    door = PAPER["tdo"] + PAPER["tdc"] + PAPER["tsd"]
    rng = random.Random(seed)
    w = [p / U for p in pops]
    total = 0.0
    for _ in range(runs):
        p = 17 if rng.random() < 0.8 else 16        # mean 16.8 passengers
        dests = sorted(set(rng.choices(range(1, N + 1), weights=w, k=p)))
        t, pos = 0.0, 0
        for d in dests:
            t += t_jerk(lvl[d] - lvl[pos], v, a, j)
            pos = d
        t += t_jerk(lvl[pos], v, a, j)
        t += (len(dests) + 1) * door + p * (PAPER["tpi"] + PAPER["tpo"])
        total += t
    return total / runs


def equal_floor_monte_carlo(N, df, v, a, p, door, tp, runs=20000, seed=1):
    rng = random.Random(seed)
    total = 0.0
    for _ in range(runs):
        dests = sorted(set(rng.randint(1, N) for _ in range(p)))
        t, pos = 0.0, 0
        for d in dests:
            t += t_const_acc((d - pos) * df, v, a)
            pos = d
        t += t_const_acc(pos * df, v, a)
        t += (len(dests) + 1) * door + 2 * p * tp
        total += t
    return total / runs


# ---------------------------------------------------------------------------
# 1. Published worked example
# ---------------------------------------------------------------------------
def test_worked_example_probable_stops():
    S, *_ = paper_components()
    assert S == pytest.approx(9.737, abs=0.001)


def test_worked_example_door_time():
    _, _, tD, _, _ = paper_components()
    assert tD == pytest.approx(59.05, abs=0.01)


def test_worked_example_passenger_transfer_time():
    _, _, _, tP, _ = paper_components()
    assert tP == pytest.approx(40.32, abs=0.01)


def test_worked_example_expected_travel_distance():
    _, Ed, _, _, _ = paper_components()
    assert Ed == pytest.approx(58.28, abs=0.01)


def test_worked_example_express_return_time():
    *_, tH = paper_components()
    assert tH == pytest.approx(19.57, abs=0.01)


def test_worked_example_monte_carlo_round_trip_time():
    # paper: 176.33 s (formula), 176.377 s (Monte Carlo)
    assert paper_monte_carlo() == pytest.approx(176.35, rel=0.005)


def test_engine_probable_stops_matches_general_formula_for_equal_populations():
    # the engine's equal-population S must equal the general formula with U_i = U/N
    N, P = 14, 17
    general = N - sum((1 - 1 / N) ** P for _ in range(N))
    assert probable_stops(N, P) == pytest.approx(general, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Engine vs validated Monte Carlo (equal floors, constant acceleration)
# ---------------------------------------------------------------------------
CASES = [  # (floors above terminal, floor height m, speed m/s, load kg) - speed reached in one floor
    (10, 3.3, 1.6, 1000),
    (20, 3.3, 1.75, 1000),
    (8, 4.0, 1.0, 630),
    (14, 4.2, 1.6, 1275),
]


@pytest.mark.parametrize("N,df,v,load", CASES)
def test_engine_round_trip_time_matches_monte_carlo(N, df, v, load):
    r = run_traffic_study(num_floors=N, population=1000, num_lifts=1, rated_speed_ms=v,
                          rated_load_kg=load, floor_height_m=df)
    assert r.rated_speed_reached_in_one_floor
    mc = equal_floor_monte_carlo(N, df, v, 1.0, passengers_per_trip(load), 8.0, 1.1)
    assert r.round_trip_time_s == pytest.approx(mc, rel=0.02)


def test_engine_flags_when_rated_speed_not_reached_in_one_floor():
    r = run_traffic_study(num_floors=15, population=1000, num_lifts=1, rated_speed_ms=2.5,
                          rated_load_kg=1275, floor_height_m=3.6)
    assert r.rated_speed_reached_in_one_floor is False


def test_hand_calculated_example_matches_engine():
    """Fully hand-worked example reproduced in the thesis (Section 6.x)."""
    N, df, v, load = 10, 3.3, 1.6, 1000
    P = passengers_per_trip(load)                     # floor(1000/75)=13 -> 0.8*13 = 10.4 -> 10
    assert P == 10
    S = probable_stops(N, P)                          # 10*(1-0.9^10) = 6.513
    H = highest_reversal_floor(N, P)                  # 10 - sum((j/10)^10, j=1..9) = 9.509
    tf1 = single_floor_flight_time(df, v, 1.0)        # 2*1.6 + (3.3-2.56)/1.6 = 3.6625
    tv = df / v                                       # 2.0625
    ts = tf1 - tv + 8.0                               # 9.6
    rtt = 2 * H * tv + (S + 1) * ts + 2 * P * 1.1     # 39.223 + 72.127 + 22.0 = 133.35
    assert S == pytest.approx(6.513, abs=0.001)
    assert H == pytest.approx(9.509, abs=0.001)
    assert tf1 == pytest.approx(3.6625, abs=1e-4)
    assert rtt == pytest.approx(133.35, abs=0.05)
    r = run_traffic_study(num_floors=N, population=1000, num_lifts=1, rated_speed_ms=v,
                          rated_load_kg=load, floor_height_m=df)
    assert r.round_trip_time_s == pytest.approx(rtt, abs=0.05)


# ---------------------------------------------------------------------------
# 3. The ENGINE'S OWN general-case helpers against the published figures
# ---------------------------------------------------------------------------
from src.traffic_engine.traffic_study import (  # noqa: E402
    flight_time_jerk, probable_stops_unequal, expected_up_travel_distance,
)


def test_engine_unequal_population_stops_matches_published():
    assert probable_stops_unequal(PAPER["pops"], PAPER["P"]) == pytest.approx(9.737, abs=0.001)


def test_engine_expected_travel_distance_matches_published():
    heights = [PAPER["heights"][i] for i in range(14)]
    assert expected_up_travel_distance(heights, PAPER["pops"], PAPER["P"]) == pytest.approx(58.28, abs=0.01)


def test_engine_jerk_limited_flight_time_matches_published():
    assert flight_time_jerk(58.28, 4.0, 1.0, 1.0) == pytest.approx(19.57, abs=0.01)


@pytest.mark.parametrize("d", [0.5, 1.9, 2.0, 2.1, 10.0, 19.9, 20.0, 20.1, 60.0])
def test_engine_jerk_flight_time_agrees_with_reference_and_is_continuous(d):
    assert flight_time_jerk(d, 4.0, 1.0, 1.0) == pytest.approx(t_jerk(d, 4.0, 1.0, 1.0), rel=1e-12)
