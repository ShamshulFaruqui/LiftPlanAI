"""
Minimum Requirements Proposer — proposes a minimum pit depth and
headroom (overhead clearance) for a lift configuration when these are
NOT stated in the drawing.

Built because real drawing testing during this project found this to
be the COMMON case, not the exception: none of the three real drawings
tested (an office building, a warehouse, and a villa) stated pit depth
or overhead clearance explicitly anywhere. Rather than leaving these as
simply "not extracted," this module proposes an engineering minimum
from the lift's rated speed, sourced to a real published planning guide.

Sourced from the KONE MonoSpace 500 Planning Guide — the same source
already used for the load/speed envelope in the synthetic generator
(see src/synthetic/generate_dataset.py's KONE_* constants).

HONESTY NOTE, read before treating these numbers as more precise than
they are: the guide gives exact figures at only three speed points
(1.00, 1.60, and a stated maximum around 1.75 m/s / with counterweight
safety gear). Anything in between is LINEARLY INTERPOLATED here — a
reasonable engineering approximation, not an independently verified
value for every possible speed. Headroom is even rougher: the guide
gives a real RANGE (CH+1300 to CH+1580mm) depending on exact ceiling
type, entrance configuration, and frame type, not one number — this
module uses CH+1500mm as a representative minimum within that range.
Every value returned includes a note explaining exactly how it was
derived, and is explicitly flagged as PROPOSED, not extracted from any
drawing — never present this as a fact about a specific installation
without that context.

Usage:
    python minimum_requirements.py --speed 1.6 --car-height 2200
"""

import argparse
import json

# Pit depth (mm) by rated speed, from KONE MonoSpace 500 Planning Guide.
KONE_PIT_DEPTH_BY_SPEED_MM = {
    1.00: 1050,
    1.60: 1200,
    1.75: 1550,  # the guide's stated maximum, for higher speeds / with counterweight safety gear
}

# Headroom = Car Height (CH) + this clearance, in mm.
KONE_HEADROOM_CLEARANCE_MM = 1500

DEFAULT_CAR_HEIGHT_MM = 2200  # a common standard car height, used only when none is otherwise known


def propose_pit_depth_mm(rated_speed_ms: float) -> dict:
    """
    Propose a minimum pit depth for the given rated speed, via linear
    interpolation between KONE MonoSpace 500's published data points.
    """
    speeds = sorted(KONE_PIT_DEPTH_BY_SPEED_MM.keys())

    if rated_speed_ms <= speeds[0]:
        value = KONE_PIT_DEPTH_BY_SPEED_MM[speeds[0]]
        note = (f"At or below the lowest KONE-published speed point ({speeds[0]}m/s) "
                f"— using that value directly, not interpolated.")
    elif rated_speed_ms >= speeds[-1]:
        value = KONE_PIT_DEPTH_BY_SPEED_MM[speeds[-1]]
        note = (f"At or above the highest KONE-published speed point ({speeds[-1]}m/s) "
                f"— using that value directly, not interpolated.")
    else:
        lower_speed = max(s for s in speeds if s <= rated_speed_ms)
        upper_speed = min(s for s in speeds if s >= rated_speed_ms)
        if lower_speed == upper_speed:
            value = KONE_PIT_DEPTH_BY_SPEED_MM[lower_speed]
            note = f"Matches a KONE-published speed point ({lower_speed}m/s) exactly."
        else:
            lower_val = KONE_PIT_DEPTH_BY_SPEED_MM[lower_speed]
            upper_val = KONE_PIT_DEPTH_BY_SPEED_MM[upper_speed]
            fraction = (rated_speed_ms - lower_speed) / (upper_speed - lower_speed)
            value = round(lower_val + fraction * (upper_val - lower_val))
            note = (f"Linearly interpolated between KONE-published points at {lower_speed}m/s "
                    f"({lower_val}mm) and {upper_speed}m/s ({upper_val}mm) — not an "
                    f"independently verified value for this exact speed.")

    return {
        "proposed_pit_depth_mm": value,
        "source": "KONE MonoSpace 500 Planning Guide",
        "note": note,
        "is_proposed_not_extracted": True,
    }


def propose_headroom_mm(car_height_mm: int = None) -> dict:
    """
    Propose a minimum headroom (overhead clearance) as Car Height +
    KONE_HEADROOM_CLEARANCE_MM. If car_height_mm is unknown, a
    representative default is used instead — a weaker assumption than
    the speed-based pit depth figure, since car height varies by car
    capacity/type rather than speed, so proposing a total headroom
    without knowing the real car height is a rougher estimate than the
    pit depth proposal above.
    """
    used_default = car_height_mm is None
    ch = car_height_mm if car_height_mm is not None else DEFAULT_CAR_HEIGHT_MM
    value = ch + KONE_HEADROOM_CLEARANCE_MM

    default_caveat = (
        f" (a default standard value — the actual car height was not known)" if used_default else ""
    )
    note = (
        f"Car Height ({ch}mm{default_caveat}) + {KONE_HEADROOM_CLEARANCE_MM}mm clearance. "
        f"The KONE MonoSpace 500 Planning Guide gives a real range of approximately "
        f"CH+1300 to CH+1580mm depending on exact ceiling type, entrance configuration, "
        f"and frame type — {KONE_HEADROOM_CLEARANCE_MM}mm is a representative minimum "
        f"within that range, not one precisely verified number."
    )

    return {
        "proposed_headroom_mm": value,
        "car_height_used_mm": ch,
        "car_height_was_assumed": used_default,
        "source": "KONE MonoSpace 500 Planning Guide",
        "note": note,
        "is_proposed_not_extracted": True,
    }


def propose_minimum_requirements(rated_speed_ms: float, car_height_mm: int = None) -> dict:
    """Combine pit depth and headroom proposals for a full picture."""
    return {
        "pit_depth": propose_pit_depth_mm(rated_speed_ms),
        "headroom": propose_headroom_mm(car_height_mm),
    }


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Minimum Requirements Proposer")
    parser.add_argument("--speed", type=float, required=True, help="Rated speed (m/s)")
    parser.add_argument("--car-height", type=int, default=None, help="Car height (mm), if known")
    args = parser.parse_args()
    result = propose_minimum_requirements(args.speed, args.car_height)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
