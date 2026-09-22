"""
Real Drawing Lift Identifier — infers building type and lift group
structure from conventions found in REAL drawings, as a complement to
level_extractor.py (which handles floor count / total travel).

Built against findings from a real drawing during this project's
development, which revealed a genuinely different structure from the
synthetic dataset's assumptions:

- Building type isn't stated explicitly anywhere obvious — it has to be
  inferred from the project name/title block (e.g. "ELLINGTON BUKADRA
  OFFICE, DUBAI" implies a commercial/office building).
- Lift cars are labelled individually (e.g. "L01".."L08"), not as a
  single compound group label like the synthetic generator's
  "EL01-EL02". Car count is the number of DISTINCT car labels found.
- There is no explicit "(PASSENGER)"/"(SERVICE)"/"(FIREFIGHTER)" label
  anywhere near a car ID. Type has to be inferred from context — a
  separately-labelled "SERV.LIFT" indicates a service lift; the overall
  drawing/section title indicates duty scope.
- IMPORTANT, non-obvious finding: a drawing titled "PASSENGER AND FIRE
  SERVICE LIFT" does NOT necessarily mean there is a separate, distinct
  firefighter-labelled car. In practice, firefighter (EN81-72) duty is
  very often a DUAL RATING applied to one or more of the regular
  passenger cars, not a physically separate lift with its own label.
  This module reports firefighter-duty as a building-level flag (is
  firefighter duty required/mentioned SOMEWHERE), not as a distinct
  group — asserting a distinct firefighter group here would be guessing
  a structure that may not exist, contradicting the "flag ambiguity,
  don't guess" principle used throughout this project.

Usage:
    python real_lift_identifier.py --input ../../data/real_test/some_drawing.pdf
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.parser.text_extractor import extract_text

BUILDING_TYPE_KEYWORDS = {
    "commercial": ["OFFICE", "COMMERCIAL", "BUSINESS"],
    "residential": ["RESIDENTIAL", "APARTMENT"],
    "hotel": ["HOTEL", "RESORT"],
    "hospital": ["HOSPITAL", "MEDICAL CENTRE", "MEDICAL CENTER", "CLINIC"],
    "warehouse": ["WAREHOUSE", "LOGISTICS", "DISTRIBUTION CENTRE", "DISTRIBUTION CENTER"],
    # Deliberately does NOT include "RESIDENCE" alone or "PRIVATE
    # RESIDENCE" — both would collide with "residential"'s "RESIDENTIAL"
    # keyword as a substring match (e.g. "PRIVATE RESIDENCE" contains
    # "RESIDENCE", and an earlier version of "residential"'s keyword list
    # included bare "RESIDENCE", which matched first due to dict order
    # and misclassified villas as residential — confirmed and fixed).
    "villa": ["VILLA"],
}

# Checked in this order, not dict insertion order — see infer_building_type()
# for why specific building-type names must be checked before the more
# generic, collision-prone "commercial"/"residential" keywords.
PRIORITY_ORDER = ["hospital", "hotel", "warehouse", "villa", "commercial", "residential"]

# Car label pattern: "L" followed by an optional hyphen/space and 1-3
# digits. The 1-3 digit tolerance (rather than a fixed 2) accommodates
# OCR sometimes inserting an extra digit (e.g. "L004" was observed in
# testing where the true label was "L04") — normalise_car_label() below
# handles reconciling this rather than the regex trying to guess it.
# Compact range notation for a group of car IDs, e.g. "PUBLIC / VISITOR
# ELEVATORS [P1~P3]" (bracketed, found in a real hospital drawing) or
# "ELEVATORS - HPL1 ~ HPL4" (dash-separated, found in a real hotel
# drawing — same underlying convention, different punctuation). Matched
# specifically near the word ELEVATOR(S)/LIFT(S) to distinguish it from
# an unrelated FLOOR range mentioned elsewhere in the very same drawing
# using a different letter prefix (both the hospital and this hotel
# separately use "L1~L8"-style ranges to mean FLOOR levels, not car IDs
# — confirmed on both). Takes priority over CAR_LABEL_PATTERN below
# when found, specifically to avoid the real bug this was built to fix.
BRACKET_RANGE_PATTERN = re.compile(
    r'(?:ELEVATORS?|LIFTS?)\s*[-\[]\s*([A-Z]{1,4})(\d+)\s*~\s*[A-Z]{0,4}(\d+)[\]]?', re.IGNORECASE
)

# Real bug found and fixed: the separator between "L" and its digits
# originally allowed ANY whitespace, including a newline — which meant
# level abbreviations ending in "L" (e.g. "S.S.L", "F.F.L", both
# genuinely real and common) immediately followed by their OWN
# elevation value on the next line (e.g. "S.S.L\n8.400 m") were
# misread as a car label "L8". Genuine car labels are always written as
# a single contiguous token on one line (e.g. "L01", "L-02", "L 02" at
# most) — restricting the separator to same-line space/hyphen only
# (never a newline) fixes this without needing a separate lookbehind
# for every possible dotted abbreviation ending in "L".
CAR_LABEL_PATTERN = re.compile(r'\bL[- ]?(\d{1,3})\b')

# Bare "LIFT" label with no car ID at all — found in a real villa
# drawing, where a single-lift building's lift is simply labelled
# "LIFT" (and "LIFT LOBBY" for its adjacent lobby area), with no L01-
# style numbering, since there's nothing to number in a one-lift
# building. Word-boundary matched so it doesn't accidentally match
# inside an unrelated longer word.
BARE_LIFT_PATTERN = re.compile(r'\bLIFT\b', re.IGNORECASE)

# Dimensions embedded directly in the lift's own label — found in a
# real warehouse drawing: "LIFT 2.0X1.6" states the car's plan
# dimensions (width x depth, in METRES — confirmed by physical
# plausibility, a real elevator car scale) directly in the label text,
# rather than as a separate dimension elsewhere on the drawing.
LIFT_WITH_DIMENSIONS_PATTERN = re.compile(
    r'\bLIFT\s+(\d+\.?\d*)\s*[Xx]\s*(\d+\.?\d*)\b'
)

SERVICE_LIFT_KEYWORDS = ["SERV.LIFT", "SERV LIFT", "SERVLIFT", "SERVICE LIFT"]
FIREFIGHTER_KEYWORDS = ["FIRE SERVICE LIFT", "FIREFIGHTER", "FIRE-FIGHTING LIFT", "EN81-72", "EN 81-72"]


@dataclass
class RealLiftIdentification:
    source_file: str
    inferred_building_type: str = None
    car_labels: list = field(default_factory=list)
    car_count: int = None
    car_count_is_uncertain: bool = False
    car_dimensions_m: tuple = None
    labeled_dimensions_mm: dict = field(default_factory=dict)
    service_lift_present: bool = False
    firefighter_duty_mentioned: bool = False
    firefighter_duty_note: str = ""


def infer_building_type(text: str) -> str:
    """
    Search for building-type keywords, typically found in the project
    name / title block rather than stated as an explicit field. Returns
    None if no known keyword is found, rather than guessing — an
    unrecognised project type should be flagged for engineer input, not
    silently assumed.

    Checks categories in PRIORITY_ORDER (most specific/least ambiguous
    first), not raw dict insertion order. Real bug found and fixed via
    a real hospital drawing: its actual project name states "HAMDAN BIN
    RASHID CANCER HOSPITAL" clearly, but the same title block also
    mentions "Design Office" and "Support Office" (consultant ROLE
    titles, describing who did the work, not what the building is).
    With "commercial" checked before "hospital" (plain dict order), the
    generic "OFFICE" substring matched first and the drawing was
    misclassified as commercial — confirmed directly before fixing.
    Specific building-type names (HOSPITAL, HOTEL, WAREHOUSE, VILLA) are
    essentially never used to describe an unrelated consultant role, so
    checking them first is safe; "commercial" and "residential" use
    much more generic, collision-prone keywords (OFFICE, RESIDENTIAL)
    that plausibly appear in unrelated title-block text, so they are
    checked last, only when nothing more specific matched.
    """
    text_upper = text.upper()
    for building_type in PRIORITY_ORDER:
        keywords = BUILDING_TYPE_KEYWORDS[building_type]
        if any(keyword in text_upper for keyword in keywords):
            return building_type
    return None


# Plausible car-number range. No real elevator installation has anywhere
# near this many cars in one group — this exists specifically to reject
# OCR noise matched from unrelated dimension numbers near a stray "L"
# character (observed on the real drawing: "L 400" and "L579", both from
# ordinary millimetre dimension callouts, not genuine car labels).
MIN_PLAUSIBLE_CAR_NUMBER = 1
MAX_PLAUSIBLE_CAR_NUMBER = 30


def normalise_car_label(raw_digits: str):
    """
    Normalise a car label's digit portion to a consistent 2-digit form,
    tolerating OCR occasionally inserting an extra digit (observed on a
    real drawing: "L004" where the true label was "L04"). Strips leading
    zeros then re-pads to 2 digits; a 3-digit input is treated as having
    one spurious leading/embedded zero rather than a genuinely large car
    number, since lift groups this large are not realistic.

    Returns None if the resulting number falls outside a plausible range
    for a real elevator group — this rejects OCR noise matched from
    unrelated dimension numbers (see MAX_PLAUSIBLE_CAR_NUMBER) rather
    than silently accepting an implausible label like "L79".
    """
    stripped = raw_digits.lstrip("0") or "0"
    if len(stripped) > 2:
        stripped = stripped[-2:]  # keep the last two digits as the more reliable ones
    number = int(stripped)
    if not (MIN_PLAUSIBLE_CAR_NUMBER <= number <= MAX_PLAUSIBLE_CAR_NUMBER):
        return None
    return f"L{number:02d}"


# Known labeled dimension fields, each expected as its own line
# immediately followed by a bare numeric value (in mm) on the next
# line — found as a clean, directly-adjacent "LABEL\nVALUE" pattern in
# a real drawing from a different consultant than any tested so far.
# Genuinely different from, and simpler than, the warehouse's "CLEAR
# SHAFT WIDTH" case (spatial_dimension_extractor.py), where the label
# and its value were NOT textually adjacent at all and needed OCR
# bounding-box analysis — here, plain sequential text is enough.
LABELED_DIMENSION_FIELDS = [
    "CABIN WIDTH", "CABIN DEPTH", "SHAFT WIDTH", "SHAFT DEPTH",
    "DOOR WIDTH", "DOOR HEIGHT", "PIT DEPTH",
]


def extract_labeled_dimensions(text: str) -> dict:
    """
    Extract every known dimension label (see LABELED_DIMENSION_FIELDS)
    that's immediately followed by a bare number on the next line.
    Returns a dict of {label: value_mm}. If a label appears more than
    once (e.g. this drawing's ground and first floor plans both state
    "CABIN WIDTH"), the FIRST occurrence is kept — real drawings tested
    so far repeat the same value for the same physical lift across
    multiple plan views, so this is a reasonable, simple choice rather
    than needing to reconcile potentially-conflicting duplicates.
    """
    lines = [line.strip() for line in text.split("\n")]
    found = {}

    for i, line in enumerate(lines):
        line_upper = line.upper()
        if line_upper not in LABELED_DIMENSION_FIELDS:
            continue
        if line_upper in found:
            continue  # keep the first occurrence only — see docstring
        if i + 1 >= len(lines):
            continue
        value_line = lines[i + 1]
        if re.match(r'^\d{1,5}$', value_line):
            found[line_upper] = int(value_line)

    return found


def extract_car_dimensions_from_label(text: str) -> tuple:
    """
    Extract a car's plan dimensions (width, depth in metres) when stated
    directly in the lift's own label — e.g. "LIFT 2.0X1.6" from a real
    warehouse drawing. Returns None if no such embedded-dimension label
    is found (most drawings don't state dimensions this way — the
    office building and villa this project was also tested against
    state car/shaft dimensions elsewhere, not in the label itself).
    """
    match = LIFT_WITH_DIMENSIONS_PATTERN.search(text)
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)))


def detect_bare_lift_label(text: str) -> bool:
    """
    True if the word "LIFT" appears as a standalone term anywhere in the
    text — used as a fallback signal for single-lift buildings with no
    numbered car ID at all (found in a real villa drawing: just "LIFT"
    and "LIFT LOBBY", no "L01"-style label, since there's nothing to
    number in a one-lift building).

    KNOWN LIMITATION, not fixed without real evidence to fix it against:
    CAR_LABEL_PATTERN requires "L" at a word boundary (so "L01" matches
    but "EL01" does not, since "E" and "L" are both word characters with
    no boundary between them). If a real drawing used a letter-prefixed
    numbering convention this pattern doesn't catch, this bare-LIFT
    fallback would incorrectly report car_count=1 instead of the true
    (higher) count. Not addressed here because no real drawing tested
    during this project actually uses such a convention — extending
    CAR_LABEL_PATTERN to guess at unconfirmed conventions risks its own
    false positives on a different real drawing. Extend this if and
    when a real drawing is found that needs it.
    """
    return bool(BARE_LIFT_PATTERN.search(text))


def extract_bracket_range_cars(text: str) -> list:
    """
    Extract car IDs from a compact range notation like "ELEVATORS
    [P1~P3]" or "ELEVATORS - HPL1 ~ HPL4" (both mean an enumerated list
    of cars). Returns an empty list if no such notation is found — most
    drawings don't use it (the office building and villa this project
    was also tested against label each car individually instead).

    Bare "L" is deliberately rejected as a prefix here even though the
    regex would otherwise accept it — confirmed on TWO separate real
    drawings (a hospital and a hotel) that "L" + digit in a "~" range
    means a FLOOR range being served ("ELEVATORS ... L1~L8"), not an
    enumerated car list, even when it directly follows the word
    ELEVATORS. Every genuine car-range prefix seen across all real
    drawings tested so far (P, HPL, RPL) is NOT bare "L".
    """
    match = BRACKET_RANGE_PATTERN.search(text)
    if not match:
        return []
    prefix = match.group(1).upper()
    if prefix == "L":
        return []  # bare "L" range means floor levels, not cars — see docstring
    start, end = int(match.group(2)), int(match.group(3))
    if start > end or end - start > 20:
        return []  # implausible range — likely a malformed match, not a genuine car group
    return [f"{prefix}{n}" for n in range(start, end + 1)]


def extract_car_labels(text: str) -> list:
    """
    Find all distinct car labels in the text, preferring the more
    reliable range notation (see extract_bracket_range_cars) when
    present, before falling back to the generic "L01".."L08"-style
    pattern below.

    REAL BUG FOUND AND FIXED, TWICE: the generic pattern alone cannot
    distinguish a genuine car ID ("L01" on the office building, where
    each car is individually labelled) from an unrelated FLOOR
    reference using the same "L" + digit shape. Confirmed on TWO
    separate real drawings:
    - A hospital using "L1".."L8" to mean floor LEVELS (actual car IDs:
      "P1".."P3") — the generic pattern alone reported 9 "cars" (every
      floor reference) when there are 3.
    - A hotel using the same shape in a longer comma-separated list,
      "L1(G), L3, L4, L5, L6 ~ L16" (actual car IDs: "HPL1".."HPL4").
      A first attempt at fixing this only stripped the ONE "L" token
      directly touching the "~" character, missing every other
      comma-separated item in the same list — confirmed directly
      before switching to the proximity-based approach below, which
      excludes ANY "L" + digit match with a "~" anywhere nearby, not
      just ones immediately adjacent to it.
    """
    bracket_cars = extract_bracket_range_cars(text)
    if bracket_cars:
        return sorted(bracket_cars)

    tilde_positions = [i for i, ch in enumerate(text) if ch == "~"]

    def near_a_tilde(match_start: int, match_end: int, window: int = 80) -> bool:
        return any(match_start - window <= pos <= match_end + window for pos in tilde_positions)

    normalised = set()
    for match in CAR_LABEL_PATTERN.finditer(text):
        if near_a_tilde(match.start(), match.end()):
            continue  # part of a floor-range list, not a genuine car ID — see docstring
        label = normalise_car_label(match.group(1))
        if label:
            normalised.add(label)

    return sorted(normalised)


def detect_service_lift(text: str) -> bool:
    text_upper = text.upper()
    return any(keyword in text_upper for keyword in SERVICE_LIFT_KEYWORDS)


def detect_firefighter_duty(text: str) -> tuple:
    """
    Returns (mentioned, note). Deliberately does NOT claim a distinct
    firefighter-labelled car exists — see module docstring for why this
    would be an unjustified assumption in real buildings, where
    firefighter duty is frequently a dual rating on regular passenger
    cars rather than a separately labelled lift.
    """
    text_upper = text.upper()
    for keyword in FIREFIGHTER_KEYWORDS:
        if keyword in text_upper:
            return True, (
                f"Firefighter/fire-service duty mentioned (matched '{keyword}'), but no "
                f"distinct firefighter-labelled car was assumed — in practice this duty is "
                f"often a dual rating on one or more regular passenger cars rather than a "
                f"physically separate lift. Verify against the drawing which specific car(s) "
                f"carry this rating."
            )
    return False, ""


def identify_real_lifts(text: str, source_file: str = "") -> RealLiftIdentification:
    result = RealLiftIdentification(source_file=source_file)

    result.inferred_building_type = infer_building_type(text)
    result.car_labels = extract_car_labels(text)

    if result.car_labels:
        result.car_count = len(result.car_labels)
    elif detect_bare_lift_label(text):
        # No numbered car IDs found, but the word "LIFT" appears
        # standalone — treat as exactly one unnamed car (a single-lift
        # building, as found in the real villa this fallback was built
        # against) rather than reporting car_count=None when a lift is
        # clearly present just without a number to identify it.
        result.car_labels = ["LIFT"]
        result.car_count = 1
        result.car_count_is_uncertain = True

    result.car_dimensions_m = extract_car_dimensions_from_label(text)
    result.labeled_dimensions_mm = extract_labeled_dimensions(text)
    result.service_lift_present = detect_service_lift(text)
    result.firefighter_duty_mentioned, result.firefighter_duty_note = detect_firefighter_duty(text)

    return result


def identify_from_pdf(pdf_path: Path) -> RealLiftIdentification:
    text = extract_text(pdf_path)
    return identify_real_lifts(text, source_file=str(pdf_path))


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Real Drawing Lift Identifier")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    result = identify_from_pdf(Path(args.input))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
