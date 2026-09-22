"""
Level/Elevation Extractor — a SECOND, independent text-parsing approach
built specifically for how real architectural drawings actually express
floor count and total travel, discovered via real (non-synthetic)
drawing testing during this project.

This project's synthetic generator (and the original text_extractor.py
built against it) assumed drawings state a single "TOTAL TRAVEL: Xm"
value directly. Real drawings tested during this project do not do this
at all — instead, every floor level is listed with its own name and
elevation value, following the common BIM/CAD level-naming convention:

    P3_PODIUM3          12200.000
    P2_PODIUM2           8700.000
    P1_PODIUM1           5200.000
    GR_GROUND FLOOR         0.000
    B1_BASEMENT1        -4450.000
    B2_BASEMENT2        -8050.000
    B3_WATER TANK LEVEL -9200.000
    B3_UNDERGROUND LEVEL -11450.000

Floor count is the number of distinct level labels; total travel is the
difference between the highest and lowest elevation. This module exists
to extract exactly that, from text that may itself be OCR output (see
text_extractor.py's ocr_extract_text) and therefore noisy.

UNIT ASSUMPTION, stated explicitly: raw elevation values are treated as
MILLIMETRES. The one real drawing this was built against contains a note
reading "ALL DIMENSIONS ARE IN MILLIMETERS AND LEVELS IN METERS", which
taken literally would mean these values are already in metres — but the
observed numbers (e.g. differences of ~3500-5200 between adjacent floors)
only make physical sense as millimetres; read as metres they would imply
multi-kilometre floor-to-floor heights. This is standard BIM/Revit
elevation-display convention. If a future real drawing genuinely does
report these values in metres, this assumption would need revisiting —
flagged here rather than silently assumed correct forever.

SIGN INFERENCE: OCR frequently corrupts the minus sign on negative
(below-datum) elevations — confirmed on the real drawing this was built
against ("B2_BASEMENT2" appeared with no sign at all in the OCR output
where a genuine drawing would show one). Sign is inferred from the level
label's semantic content (BASEMENT/UNDERGROUND/WATER TANK/PIT implies
below datum) when no explicit sign is present, rather than trusting OCR's
sign character, which is treated as unreliable.

Usage:
    python level_extractor.py --input ../../data/real_test/some_drawing.pdf
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.parser.text_extractor import extract_text

# A level label: a KNOWN level-code prefix, optionally followed by a
# number, plus a descriptive name, separated by an underscore OR a space
# (OCR frequently drops underscores).
#
# Originally restricted to only the prefixes observed on the one real
# drawing this module was built against (P, B, GR, RF, F) — an earlier,
# more permissive pattern matched essentially any short fragment as a
# level label (see git history / project README for the "TO_BE
# INTERFACE" false-positive case that caused this restriction).
#
# Broadened here to cover other COMMON real-world AEC level-naming
# conventions, based on general domain knowledge rather than a second
# real drawing (only one was available to test against — this is an
# informed extension, not something verified against further real
# files, and is documented as such rather than claimed as proven).
# Each addition was individually considered for false-positive risk:
#   - GF, LG, UG, MZ, PH: two-letter codes, low collision risk
#   - TER: three letters (not bare "T", which risks matching "TRUE
#     NORTH", "TOTAL", etc. — common drawing header/note text)
#   - FL, LVL: require a following number, low collision risk
#   - S: DELIBERATELY requires a digit immediately after (not bare "S"),
#     since bare "S" would collide with "SERVICE", "SUMP", "STAIR" etc.
#     — all real terms seen on the drawing this module was tested
#     against. "L" (Level) is deliberately NOT added here despite being
#     common in practice — it is already used for lift CAR labels in
#     real_lift_identifier.py, and adding it here would create a direct,
#     unresolvable ambiguity between "Level 5" and "Car L05" that this
#     project has no reliable way to disambiguate from text alone.
LEVEL_LABEL_PATTERN = re.compile(
    r'\b(P\d{1,2}|B\d{1,2}|GR|GF|LG|UG|MZ\d{0,1}|PH|TER\d{0,1}|'
    r'RF\d{0,2}|F\d{1,3}|FL\d{1,2}|LVL\d{1,2}|S\d{1,2})[_\s]'
    r'([A-Z][A-Z0-9]*(?:\s[A-Z0-9]+){0,4})\b'
)

# An elevation value: optional (possibly OCR-corrupted) sign, digits,
# decimal point, digits. (?<!\d) ensures matching starts at the true
# beginning of the number, not partway through — an earlier version
# without this capped the integer part at 3 digits with no start
# anchor, silently truncating "12200.000" down to "200.000" by matching
# only the last 3 digits before the decimal point.
ELEVATION_PATTERN = re.compile(r'(?<!\d)([~\-—]?\s?\d{1,6}(?:,\d{3})*\.\d{1,3})')

# Level-indicating keywords for the "value comes FIRST" naming
# convention, e.g. "+5.80 FIRST FLOOR F.F.L" — found in a real villa
# drawing, the exact OPPOSITE order from the "P3_PODIUM3" then
# "12200.000" convention above (name first, value second). Real
# drawings are not consistent about this ordering — a single project
# (this villa) uses the reversed order compared to the office building
# the name-first pattern above was originally built against.
VALUE_FIRST_KEYWORDS = [
    "FLOOR", "ROOF", "LEVEL", "VILLA", "GATE", "PARAPET", "BASEMENT",
    "GROUND", "PODIUM", "TERRACE", "MEZZANINE", "PENTHOUSE",
    "F.F.L", "S.S.L", "FFL", "SSL",
]

# A signed decimal value (2 decimal places specifically — the precision
# actually used in the real villa drawing this is built against) followed
# by a short descriptive name, optionally in parentheses. Deliberately
# permissive on the name boundary — false positives are rejected by the
# VALUE_FIRST_KEYWORDS filter in extract_value_first_levels(), rather
# than trying to perfectly bound the name syntactically here.
#
# Case-insensitive by design: the villa drawing this was built against
# has a native PDF text layer (not OCR), which preserves the drawing's
# actual mixed-case authoring — "Main Villa" and "Gate Level" are
# genuinely mixed case in the source, even though other labels in the
# SAME drawing ("FIRST FLOOR F.F.L") are fully uppercase. An
# uppercase-only pattern found zero matches at all against this real
# text before this was corrected.
VALUE_FIRST_PATTERN = re.compile(
    r'([+-]?\d{1,3}\.\d{2})\s*\(?\s*([A-Za-z][A-Za-z0-9 ./]{2,40})\)?',
    re.IGNORECASE,
)

# Level-label keywords implying a below-datum (negative) elevation, used
# when no explicit sign is present in the source text — see module
# docstring for why OCR's sign character is not trusted on its own.
NEGATIVE_LEVEL_KEYWORDS = ["BASEMENT", "UNDERGROUND", "WATER TANK", "PIT"]

# Level-label keywords that indicate a duplicate/reference to the SAME
# physical level rather than a distinct floor (e.g. a level mentioned
# twice across two different section cuts of the same drawing).
_SEEN_TOLERANCE_MM = 50  # levels within this elevation difference are treated as the same floor


@dataclass
class LevelReading:
    label: str
    elevation_mm: float
    raw_match: str


@dataclass
class LevelExtractionResult:
    source_file: str
    levels: list = field(default_factory=list)
    floor_count: int = None
    total_travel_m: float = None
    lowest_level: str = None
    highest_level: str = None
    anomaly_flags: list = field(default_factory=list)


def detect_gap_anomalies(readings: list, anomaly_multiple: float = 3.0) -> list:
    """
    Compute the elevation gap between each pair of vertically-adjacent
    levels and flag any gap that's dramatically larger than the typical
    (median) gap — this catches two real, distinct problems found via
    testing against a real drawing: (1) a single-digit OCR misread
    (e.g. "72200" read for "12200") that displaces one level far out of
    its true position, and (2) genuinely missing intermediate floor
    labels that OCR failed to pick up at all (dense, repetitive floor
    labels on a real drawing are not always all captured). Either way,
    a large unexplained gap is worth surfacing rather than silently
    trusting every extracted number as correct.
    """
    if len(readings) < 3:
        return []  # not enough data to establish a meaningful "typical" gap

    sorted_readings = sorted(readings, key=lambda r: r.elevation_mm)
    gaps = [sorted_readings[i + 1].elevation_mm - sorted_readings[i].elevation_mm
            for i in range(len(sorted_readings) - 1)]
    gaps_sorted = sorted(gaps)
    median_gap = gaps_sorted[len(gaps_sorted) // 2]

    flags = []
    if median_gap <= 0:
        return flags
    for i, gap in enumerate(gaps):
        if gap > median_gap * anomaly_multiple:
            flags.append(
                f"Large gap ({gap:.0f}mm) between '{sorted_readings[i].label}' and "
                f"'{sorted_readings[i + 1].label}' — typical gap here is ~{median_gap:.0f}mm. "
                f"This may indicate an OCR misread on one of these two values, or "
                f"floor levels missing from extraction. Verify against the drawing directly."
            )
    return flags


def parse_elevation_value(raw: str) -> float:
    """
    Convert a matched elevation string to a signed float, tolerating
    OCR's corruption of the minus sign and comma thousands-separators.
    """
    raw = raw.strip()
    is_negative = raw.startswith(("-", "~", "—"))
    cleaned = re.sub(r'^[~\-—]\s?', '', raw).replace(",", "")
    value = float(cleaned)
    return -value if is_negative else value


def infer_sign(label_name: str, value: float, had_explicit_sign: bool) -> float:
    """
    If no explicit (or OCR-corrupted-but-detected) sign was present,
    infer whether this level should be negative from its label's
    semantic content — e.g. "BASEMENT" implies below datum. Levels that
    already have a detected sign are left as-is; this only fills the gap
    OCR leaves when it drops a minus sign entirely.
    """
    if had_explicit_sign or value == 0:
        return value
    if any(keyword in label_name.upper() for keyword in NEGATIVE_LEVEL_KEYWORDS):
        return -abs(value)
    return value


def extract_level_readings(text: str) -> list:
    """
    Scan text line-by-line for level-label lines, then look at the same
    line and the next two lines for an elevation value — real drawings
    (and OCR output of them) place the label and its elevation on
    separate lines far more often than on the same line.
    """
    lines = text.split("\n")
    readings = []

    for i, line in enumerate(lines):
        label_match = LEVEL_LABEL_PATTERN.search(line)
        if not label_match:
            continue

        code, name = label_match.group(1), label_match.group(2)
        full_label = f"{code}_{name}".strip()

        # Search this line plus the next two for an elevation value.
        search_window = " ".join(lines[i:min(i + 3, len(lines))])
        elev_match = ELEVATION_PATTERN.search(search_window)
        if not elev_match:
            continue

        raw_value = elev_match.group(1)
        had_explicit_sign = raw_value.strip().startswith(("-", "~", "—"))
        value = parse_elevation_value(raw_value)
        value = infer_sign(full_label, value, had_explicit_sign)

        readings.append(LevelReading(label=full_label, elevation_mm=value, raw_match=raw_value))

    return readings


def deduplicate_levels(readings: list) -> list:
    """
    Collapse readings that refer to the same physical level (e.g. the
    same level appearing in two different section-view cuts of one
    drawing) into a single entry, keyed by label text AND elevation
    proximity — two readings with the same label but wildly different
    elevations are treated as distinct (likely a genuine OCR misread
    worth keeping separately rather than silently merging).
    """
    unique = []
    for reading in readings:
        is_duplicate = any(
            reading.label == existing.label and
            abs(reading.elevation_mm - existing.elevation_mm) <= _SEEN_TOLERANCE_MM
            for existing in unique
        )
        if not is_duplicate:
            unique.append(reading)
    return unique


def infer_unit_and_convert_to_mm(value_str: str) -> float:
    """
    Infer whether a captured elevation value is stated in metres or
    millimetres, based on its decimal precision and magnitude, and
    return it normalised to millimetres.

    Two genuinely different real conventions were found across
    different real drawings during this project's development:
    - Office building: elevations to 3 decimal places, magnitude in the
      thousands (e.g. "12200.000") -> millimetres.
    - Villa: elevations to 2 decimal places, small magnitude (e.g.
      "5.80", "10.40") -> METRES, confirmed both by the villa's own
      general note ("ALL LEVELS ARE IN METERS") and by the values only
      making physical sense as metres (5.80mm would be an absurd floor
      height; 5.80m is a very ordinary one).

    Heuristic: 2 decimal places AND magnitude under 1000 -> metres
    (multiply by 1000); otherwise treat as already being in mm. This is
    a reasonable inference from two confirmed real examples, not a
    universal rule — a future real drawing could use a convention this
    gets wrong, which is a genuine limitation to state, not hide.
    """
    value = float(value_str)
    decimal_places = len(value_str.split(".")[-1]) if "." in value_str else 0
    if decimal_places == 2 and abs(value) < 1000:
        return value * 1000
    return value


# Words that indicate a VALUE_FIRST_PATTERN match is a fragment of a
# general prose sentence, not a genuine short level label — found
# necessary via a real hotel drawing, where a general safety note
# ("HEIGHT OF 2.50M ABOVE THE FLOOR OF LOWEST [SERVING FLOOR]") was
# wrongly captured as a level reading, because the note happens to
# contain the word "FLOOR" (one of VALUE_FIRST_KEYWORDS) despite
# describing a screen height requirement, not a building level at all.
# None of the genuine level labels confirmed across every real drawing
# tested (e.g. "FIRST FLOOR F.F.L", "ROOF S.S.L", "Main Villa", "Gate
# Level") contain any of these connector words — a real label is always
# a short, unconnected label, never a grammatical sentence fragment.
PROSE_FRAGMENT_STOPWORDS = ["OF", "ABOVE", "BELOW", "THE", "FROM", "WITH", "SHALL", "WILL", "PROVIDED"]


def is_likely_prose_fragment(name: str) -> bool:
    words = name.upper().split()
    return any(word in PROSE_FRAGMENT_STOPWORDS for word in words)


def extract_value_first_levels(text: str) -> list:
    """
    Extract level readings following the "value comes first" convention
    (e.g. "+5.80 FIRST FLOOR F.F.L"), as a complement to
    extract_level_readings() (which handles the opposite "name comes
    first" convention, e.g. "P3_PODIUM3" then "12200.000" on the next
    line). Sign is taken directly from the matched value — unlike the
    name-first path, no sign-inference-from-keyword fallback is needed
    here, since this convention states the sign explicitly and (in the
    real drawings this was built against) came from a native PDF text
    layer, not OCR, so sign-character corruption wasn't a concern.
    """
    readings = []
    for match in VALUE_FIRST_PATTERN.finditer(text):
        raw_value, name = match.group(1), match.group(2).strip()
        name_upper = name.upper()
        if not any(keyword in name_upper for keyword in VALUE_FIRST_KEYWORDS):
            continue  # not a level — just some other decimal value followed by capitalised text
        if is_likely_prose_fragment(name):
            continue  # a sentence fragment that happens to contain a level keyword, not a genuine label

        elevation_mm = infer_unit_and_convert_to_mm(raw_value)
        readings.append(LevelReading(label=name, elevation_mm=elevation_mm, raw_match=raw_value))
    return readings


# Keywords indicating a level is a civil/site reference or roof-edge
# detail, NOT an occupied floor a passenger lift would actually serve —
# found necessary against a real villa drawing, where "GATE LEVEL" (the
# site entrance elevation reference) and "PARAPET LEVEL" (the roof's
# perimeter edge wall, not an occupied or lift-accessible space) would
# otherwise be counted as floors, badly inflating floor count and total
# travel for a small building (that villa is genuinely G+1 — 2 lift-
# served levels — not 5). "ROAD" added after a second, unrelated real
# drawing (a different consultant, different building) showed the exact
# same pattern with "ROAD LEVEL" — a site civil reference, not a floor.
#
# Deliberately narrow: only these three are excluded, since all three
# are unambiguous in real practice. "ROOF" is NOT excluded, since some
# buildings genuinely do have lift-accessible roof levels — blanket-
# excluding it would risk the opposite mistake (undercounting a real
# lift stop) on limited evidence.
NON_FLOOR_REFERENCE_KEYWORDS = ["GATE", "PARAPET", "ROAD"]


def is_likely_non_floor_reference(label: str) -> bool:
    label_upper = label.upper()
    return any(keyword in label_upper for keyword in NON_FLOOR_REFERENCE_KEYWORDS)


def extract_spelled_out_name_first_levels(text: str) -> list:
    """
    A THIRD level-naming convention, distinct from both patterns above:
    a fully spelled-out name on its own line (e.g. "GROUND FLOOR FFL"),
    with NO short-code prefix at all, followed by its signed decimal
    value on the very next line (e.g. "+0.30 m") — found in a real
    drawing from a different consultant than any tested so far. This
    differs from extract_level_readings (which requires a short code
    like "P3_" or "GR_") and from extract_value_first_levels (where the
    value comes BEFORE the name on the same line) — here the full name
    is alone on one line, and the value is alone on the next.

    Reuses VALUE_FIRST_KEYWORDS to recognise a level-indicating name,
    and the same metres-vs-millimetres inference already used
    elsewhere in this module.
    """
    lines = [line.strip() for line in text.split("\n")]
    readings = []

    value_pattern = re.compile(r'^([+-]?\d{1,3}\.\d{1,2})\s*m?$', re.IGNORECASE)

    for i, line in enumerate(lines):
        if not line or re.search(r'\d', line):
            continue  # a genuine spelled-out name line has no digits in it at all
        line_upper = line.upper()
        if not any(keyword in line_upper for keyword in VALUE_FIRST_KEYWORDS):
            continue
        if is_likely_prose_fragment(line):
            continue  # same prose-sentence guard used by the value-first path

        if i + 1 >= len(lines):
            continue
        value_match = value_pattern.match(lines[i + 1])
        if not value_match:
            continue

        elevation_mm = infer_unit_and_convert_to_mm(value_match.group(1))
        readings.append(LevelReading(label=line, elevation_mm=elevation_mm, raw_match=value_match.group(1)))

    return readings


def extract_levels_from_text(text: str, source_file: str = "") -> LevelExtractionResult:
    # Try all THREE real-world conventions found across different
    # drawings so far — short-code name-first (office building:
    # "P3_PODIUM3" then "12200.000"), value-first (villa: "+5.80 FIRST
    # FLOOR F.F.L"), and spelled-out name-first (a different consultant's
    # drawing: "GROUND FLOOR FFL" alone on one line, "+0.30 m" on the
    # next) — since real drawings are not consistent about which one
    # they use, and a future drawing could plausibly mix more than one
    # within a single document.
    all_readings = (extract_level_readings(text) + extract_value_first_levels(text) +
                     extract_spelled_out_name_first_levels(text))
    all_readings = deduplicate_levels(all_readings)

    result = LevelExtractionResult(source_file=source_file, levels=[asdict(r) for r in all_readings])

    # Floor count and total travel are computed from the subset of
    # readings that are actually likely to be lift-served floors — see
    # NON_FLOOR_REFERENCE_KEYWORDS above. The full, unfiltered list is
    # still reported in result.levels for transparency; only the
    # derived floor_count/total_travel_m exclude the non-floor entries.
    floor_readings = [r for r in all_readings if not is_likely_non_floor_reference(r.label)]

    if floor_readings:
        result.floor_count = len(floor_readings)
        highest = max(floor_readings, key=lambda r: r.elevation_mm)
        lowest = min(floor_readings, key=lambda r: r.elevation_mm)
        result.total_travel_m = round((highest.elevation_mm - lowest.elevation_mm) / 1000, 1)
        result.highest_level = highest.label
        result.lowest_level = lowest.label
        result.anomaly_flags = detect_gap_anomalies(floor_readings)

    return result


def extract_levels(pdf_path: Path) -> LevelExtractionResult:
    text = extract_text(pdf_path)
    return extract_levels_from_text(text, source_file=str(pdf_path))


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Level/Elevation Extractor (real-drawing format)")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    result = extract_levels(Path(args.input))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
