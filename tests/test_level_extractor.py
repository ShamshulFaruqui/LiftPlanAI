"""
Tests for the level/elevation extractor (real-drawing BIM-style format).
Run with: pytest tests/test_level_extractor.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser.level_extractor import (
    extract_level_readings, extract_levels_from_text, parse_elevation_value,
    infer_sign, deduplicate_levels, detect_gap_anomalies, LevelReading,
    extract_value_first_levels, infer_unit_and_convert_to_mm,
    is_likely_non_floor_reference, is_likely_prose_fragment,
    extract_spelled_out_name_first_levels,
)


# --- parse_elevation_value ---

def test_parse_positive_elevation():
    assert parse_elevation_value("5200.000") == 5200.0


def test_parse_explicit_negative_elevation():
    assert parse_elevation_value("-4450.000") == -4450.0


def test_parse_ocr_corrupted_negative_sign():
    # OCR commonly renders "-" as "~" or an em-dash on real drawings.
    assert parse_elevation_value("~4450.000") == -4450.0
    assert parse_elevation_value("—4450.000") == -4450.0


def test_parse_elevation_with_thousands_separator():
    assert parse_elevation_value("12,200.000") == 12200.0


# --- infer_sign ---

def test_infer_sign_basement_defaults_negative_when_no_explicit_sign():
    assert infer_sign("B1_BASEMENT1", 4450.0, had_explicit_sign=False) == -4450.0


def test_infer_sign_podium_stays_positive_when_no_explicit_sign():
    assert infer_sign("P1_PODIUM1", 5200.0, had_explicit_sign=False) == 5200.0


def test_infer_sign_respects_explicit_sign_even_for_podium():
    # If a sign genuinely was detected, don't override it based on label.
    assert infer_sign("P1_PODIUM1", -5200.0, had_explicit_sign=True) == -5200.0


def test_infer_sign_water_tank_is_negative():
    assert infer_sign("B3_WATER TANK LEVEL", 9200.0, had_explicit_sign=False) == -9200.0


# --- extract_level_readings: the two major bugs found on the real drawing ---

def test_large_elevation_values_not_truncated():
    # Real bug: an earlier regex capped the integer part at 3 digits with
    # no start-of-number anchor, silently truncating "12200.000" to
    # "200.000" by matching only the last 3 digits before the decimal.
    text = "P3_PODIUM3\n12200.000\n"
    readings = extract_level_readings(text)
    assert len(readings) == 1
    assert readings[0].elevation_mm == 12200.0


def test_ocr_garbage_not_matched_as_level_label():
    # Real bug: an earlier, overly-permissive label pattern matched OCR
    # noise fragments like "TO_BE INTERFACE", "FFL_H", "UFT_LOBBY" as if
    # they were genuine level codes, because it accepted ANY 1-3 letter
    # prefix rather than only known level-code prefixes.
    text = (
        "TO_BE INTERFACE LIFTLOBBY 5\n120.200\n"
        "FFL_H\n0.000\n"
        "UFT_LOBBY Y MACHINE ROOM\n84.200\n"
    )
    readings = extract_level_readings(text)
    assert len(readings) == 0


def test_genuine_level_labels_still_match():
    text = (
        "P3_PODIUM3\n12200.000\n"
        "GR_GROUND FLOOR\n0.000\n"
        "B1_BASEMENT1\n-4450.000\n"
        "RF1_ROOF LEVEL\n84200.000\n"
    )
    readings = extract_level_readings(text)
    labels = {r.label for r in readings}
    assert "P3_PODIUM3" in labels
    assert "GR_GROUND FLOOR" in labels
    assert "B1_BASEMENT1" in labels
    assert "RF1_ROOF LEVEL" in labels


def test_label_and_elevation_on_separate_lines():
    # Real drawings (and OCR of them) very commonly place the label and
    # its elevation on different lines, not the same line.
    text = "GR_GROUND FLOOR\nsome unrelated text\n0.000\n"
    readings = extract_level_readings(text)
    assert len(readings) == 1
    assert readings[0].elevation_mm == 0.0


# --- deduplicate_levels ---

def test_duplicate_level_across_two_section_cuts_counted_once():
    readings = [
        LevelReading(label="GR_GROUND FLOOR", elevation_mm=0.0, raw_match="0.000"),
        LevelReading(label="GR_GROUND FLOOR", elevation_mm=0.0, raw_match="0.000"),
    ]
    result = deduplicate_levels(readings)
    assert len(result) == 1


def test_same_label_different_elevation_kept_as_distinct():
    # A large elevation difference under the same label is more likely a
    # genuine misread worth keeping visible than a true duplicate.
    readings = [
        LevelReading(label="P1_PODIUM1", elevation_mm=5200.0, raw_match="5200.000"),
        LevelReading(label="P1_PODIUM1", elevation_mm=52000.0, raw_match="52000.000"),
    ]
    result = deduplicate_levels(readings)
    assert len(result) == 2


# --- detect_gap_anomalies ---

def test_large_outlier_gap_is_flagged():
    readings = [
        LevelReading(label="B1_BASEMENT1", elevation_mm=-4450.0, raw_match=""),
        LevelReading(label="GR_GROUND FLOOR", elevation_mm=0.0, raw_match=""),
        LevelReading(label="P1_PODIUM1", elevation_mm=5200.0, raw_match=""),
        LevelReading(label="P2_PODIUM2", elevation_mm=8700.0, raw_match=""),
        LevelReading(label="P3_PODIUM3", elevation_mm=72200.0, raw_match=""),  # the real OCR misread case
        LevelReading(label="P4_PODIUM4", elevation_mm=15700.0, raw_match=""),
    ]
    flags = detect_gap_anomalies(readings)
    assert len(flags) > 0
    assert any("P3_PODIUM3" in f for f in flags)


def test_evenly_spaced_levels_produce_no_anomaly():
    readings = [
        LevelReading(label="GR_GROUND FLOOR", elevation_mm=0.0, raw_match=""),
        LevelReading(label="P1_PODIUM1", elevation_mm=3500.0, raw_match=""),
        LevelReading(label="P2_PODIUM2", elevation_mm=7000.0, raw_match=""),
        LevelReading(label="P3_PODIUM3", elevation_mm=10500.0, raw_match=""),
    ]
    flags = detect_gap_anomalies(readings)
    assert flags == []


def test_too_few_readings_skips_anomaly_check_gracefully():
    readings = [LevelReading(label="GR_GROUND FLOOR", elevation_mm=0.0, raw_match="")]
    assert detect_gap_anomalies(readings) == []


# --- End-to-end: extract_levels_from_text ---

def test_end_to_end_floor_count_and_travel():
    text = (
        "P1_PODIUM1\n5200.000\n"
        "GR_GROUND FLOOR\n0.000\n"
        "B1_BASEMENT1\n-4450.000\n"
    )
    result = extract_levels_from_text(text)
    assert result.floor_count == 3
    assert result.total_travel_m == round((5200.0 - (-4450.0)) / 1000, 1)
    assert result.highest_level == "P1_PODIUM1"
    assert result.lowest_level == "B1_BASEMENT1"


def test_no_levels_found_returns_none_fields_not_crash():
    result = extract_levels_from_text("no level information in this text at all")
    assert result.floor_count is None
    assert result.total_travel_m is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


# --- Broadened level-prefix coverage (informed extension, not verified
# against a second real drawing — see LEVEL_LABEL_PATTERN's docstring) ---

def test_ground_floor_alternative_gf_prefix():
    text = "GF_GROUND FLOOR\n0.000\n"
    readings = extract_level_readings(text)
    assert any(r.label == "GF_GROUND FLOOR" for r in readings)


def test_lower_ground_prefix():
    text = "LG_LOWER GROUND\n-2500.000\n"
    readings = extract_level_readings(text)
    assert any(r.label == "LG_LOWER GROUND" for r in readings)


def test_underground_prefix():
    text = "UG_UNDERGROUND\n-6000.000\n"
    readings = extract_level_readings(text)
    assert any(r.label == "UG_UNDERGROUND" for r in readings)


def test_mezzanine_prefix():
    text = "MZ_MEZZANINE\n3500.000\n"
    readings = extract_level_readings(text)
    assert any(r.label == "MZ_MEZZANINE" for r in readings)


def test_penthouse_prefix():
    text = "PH_PENTHOUSE\n45000.000\n"
    readings = extract_level_readings(text)
    assert any(r.label == "PH_PENTHOUSE" for r in readings)


def test_terrace_prefix():
    text = "TER_ROOF TERRACE\n42000.000\n"
    readings = extract_level_readings(text)
    assert any(r.label == "TER_ROOF TERRACE" for r in readings)


def test_floor_fl_prefix():
    text = "FL05_LEVEL FIVE\n17500.000\n"
    readings = extract_level_readings(text)
    assert any(r.label.startswith("FL05") for r in readings)


def test_lvl_prefix():
    text = "LVL03_THIRD FLOOR\n10500.000\n"
    readings = extract_level_readings(text)
    assert any(r.label.startswith("LVL03") for r in readings)


def test_sub_basement_s_prefix_with_digit():
    text = "S1_SUB BASEMENT\n-15000.000\n"
    readings = extract_level_readings(text)
    assert any(r.label == "S1_SUB BASEMENT" for r in readings)


def test_bare_s_without_digit_still_rejected():
    # Critical safety check: bare "S" was deliberately NOT added (only
    # "S" + digit) specifically because it would collide with real terms
    # seen on the drawing this module was built against — SERVICE,
    # SUMP, STAIR all start with "S" followed by a letter, not a digit.
    text = "SERVICE ROOM\n1200.000\nSUMP PIT\n800.000\nSTAIR LOBBY\n2000.000\n"
    readings = extract_level_readings(text)
    assert readings == []


def test_bare_l_prefix_not_added_to_avoid_car_label_collision():
    # "L" (Level) is deliberately excluded from LEVEL_LABEL_PATTERN — it
    # is already used for lift CAR labels elsewhere in this project
    # (real_lift_identifier.py), and adding it here would create an
    # unresolvable ambiguity between "Level 5" and "Car L05".
    text = "L05_SOME LEVEL\n17500.000\n"
    readings = extract_level_readings(text)
    assert readings == []


def test_broadened_pattern_still_rejects_original_ocr_garbage_case():
    # Regression guard: the broadening must not reopen the original
    # false-positive hole this pattern was built to close.
    text = (
        "TO_BE INTERFACE LIFTLOBBY 5\n120.200\n"
        "FFL_H\n0.000\n"
        "UFT_LOBBY Y MACHINE ROOM\n84.200\n"
    )
    readings = extract_level_readings(text)
    assert readings == []


# --- Value-first level naming (villa: "+5.80 FIRST FLOOR F.F.L") ---

def test_value_first_basic_extraction():
    readings = extract_value_first_levels("+5.80 FIRST FLOOR F.F.L")
    assert len(readings) == 1
    assert readings[0].label == "FIRST FLOOR F.F.L"
    assert readings[0].elevation_mm == 5800.0


def test_value_first_mixed_case_matched():
    # Real bug found: the villa drawing's native PDF text layer (not
    # OCR) preserves genuine mixed case ("Main Villa", "Gate Level"),
    # even though other labels in the SAME drawing are fully uppercase
    # ("FIRST FLOOR F.F.L"). An uppercase-only pattern found zero
    # matches at all against this real text before being fixed.
    readings = extract_value_first_levels("+0.90 (Main Villa)")
    assert len(readings) == 1
    assert readings[0].label == "Main Villa"


def test_value_first_bare_zero_with_no_sign_matched():
    # Real bug found: "Gate Level" (the datum reference, elevation
    # 0.00) has no sign at all after CAD-code cleaning strips "%%P" —
    # a sign-mandatory pattern missed this entirely before being fixed.
    readings = extract_value_first_levels("0.00 (Gate Level)")
    assert len(readings) == 1
    assert readings[0].elevation_mm == 0.0


def test_value_first_rejects_non_level_decimal_values():
    # A decimal number NOT followed by a level-indicating keyword should
    # not be treated as a level reading at all.
    readings = extract_value_first_levels("2.50 Some Random Dimension")
    assert readings == []


def test_value_first_full_real_villa_sequence():
    from src.parser.text_extractor import clean_cad_control_codes
    villa_text = clean_cad_control_codes(
        "+0.90 (Main Villa)\n%%P0.00 (Gate Level)\n+5.80 FIRST FLOOR F.F.L\n"
        "+10.40 ROOF S.S.L\n+11.60 PARAPET LEVEL\n"
    )
    readings = extract_value_first_levels(villa_text)
    labels = {r.label for r in readings}
    assert labels == {"Main Villa", "Gate Level", "FIRST FLOOR F.F.L", "ROOF S.S.L", "PARAPET LEVEL"}


# --- Unit inference: metres vs millimetres ---

def test_two_decimal_small_value_inferred_as_metres():
    # Villa convention: "5.80" means 5.80 METRES (5800mm), confirmed by
    # the villa's own general note and by physical plausibility.
    assert infer_unit_and_convert_to_mm("5.80") == 5800.0


def test_three_decimal_large_value_treated_as_millimetres():
    # Office building convention: "12200.000" is already millimetres.
    assert infer_unit_and_convert_to_mm("12200.000") == 12200.0


def test_two_decimal_large_value_not_wrongly_converted():
    # A large value with 2 decimals shouldn't be multiplied by 1000 —
    # the magnitude check guards against misapplying the metres rule.
    assert infer_unit_and_convert_to_mm("1200.50") == 1200.50


# --- Non-floor reference exclusion (Gate Level, Parapet Level) ---

def test_gate_level_excluded_from_floor_count():
    assert is_likely_non_floor_reference("Gate Level") is True


def test_parapet_level_excluded_from_floor_count():
    assert is_likely_non_floor_reference("PARAPET LEVEL") is True


def test_roof_level_not_excluded():
    # Deliberately NOT excluded — some buildings genuinely have
    # lift-accessible roof levels; blanket-excluding "ROOF" on the
    # evidence of one villa would risk undercounting a real lift stop
    # in a different building.
    assert is_likely_non_floor_reference("ROOF S.S.L") is False


def test_ordinary_floor_not_excluded():
    assert is_likely_non_floor_reference("FIRST FLOOR F.F.L") is False


def test_end_to_end_villa_floor_count_excludes_gate_and_parapet():
    from src.parser.text_extractor import clean_cad_control_codes
    villa_text = clean_cad_control_codes(
        "+0.90 (Main Villa)\n%%P0.00 (Gate Level)\n+5.80 FIRST FLOOR F.F.L\n"
        "+10.40 ROOF S.S.L\n+11.60 PARAPET LEVEL\n"
    )
    result = extract_levels_from_text(villa_text)
    # 5 levels found in total, but only 3 are lift-relevant floors
    # (Gate Level and Parapet Level excluded).
    assert len(result.levels) == 5
    assert result.floor_count == 3
    assert result.lowest_level == "Main Villa"
    assert result.highest_level == "ROOF S.S.L"
    assert result.total_travel_m == 9.5


# --- Prose-fragment rejection (real hotel bug) ---

def test_prose_fragment_with_level_keyword_rejected():
    # Real bug found: a general safety note ("HEIGHT OF 2.50M ABOVE THE
    # FLOOR OF LOWEST SERVING FLOOR") was wrongly captured as a level
    # reading because it contains "FLOOR" — one of VALUE_FIRST_KEYWORDS
    # — despite being a sentence fragment, not a genuine label.
    text = "HEIGHT OF 2.50M  ABOVE THE FLOOR OF LOWEST"
    assert extract_value_first_levels(text) == []


def test_is_likely_prose_fragment_detects_stopwords():
    assert is_likely_prose_fragment("M ABOVE THE FLOOR OF LOWEST") is True
    assert is_likely_prose_fragment("FIRST FLOOR F.F.L") is False
    assert is_likely_prose_fragment("Main Villa") is False
    assert is_likely_prose_fragment("Gate Level") is False


def test_genuine_villa_levels_unaffected_by_prose_filter():
    from src.parser.text_extractor import clean_cad_control_codes
    villa_text = clean_cad_control_codes(
        "+0.90 (Main Villa)\n%%P0.00 (Gate Level)\n+5.80 FIRST FLOOR F.F.L\n"
        "+10.40 ROOF S.S.L\n+11.60 PARAPET LEVEL\n"
    )
    readings = extract_value_first_levels(villa_text)
    assert len(readings) == 5


# --- Spelled-out name-first convention (real Sila Marine drawing) ---

def test_spelled_out_name_first_basic_extraction():
    text = "GROUND FLOOR FFL\n+0.30 m"
    readings = extract_spelled_out_name_first_levels(text)
    assert len(readings) == 1
    assert readings[0].label == "GROUND FLOOR FFL"
    assert readings[0].elevation_mm == 300.0


def test_spelled_out_name_first_full_real_sequence():
    text = (
        "GROUND FLOOR FFL\n+0.30 m\nFIRST FLOOR FFL\n+4.80 m\n"
        "ROOF FLOOR SSL\n+8.40 m\nROAD LEVEL\n+0.00 m\nTOP ROOF\n+10.40 m\n"
    )
    readings = extract_spelled_out_name_first_levels(text)
    labels = {r.label for r in readings}
    assert labels == {"GROUND FLOOR FFL", "FIRST FLOOR FFL", "ROOF FLOOR SSL", "ROAD LEVEL", "TOP ROOF"}


def test_spelled_out_name_first_rejects_lines_with_digits():
    # A genuine name line has no digits at all — this guards against
    # matching an unrelated line that happens to contain both a keyword
    # and a number as if it were a name/value pair split awkwardly.
    readings = extract_spelled_out_name_first_levels("FLOOR 3 DETAIL\n+0.30 m")
    assert readings == []


def test_spelled_out_name_first_requires_value_on_next_line():
    readings = extract_spelled_out_name_first_levels("GROUND FLOOR FFL\nSOME OTHER TEXT")
    assert readings == []


def test_spelled_out_name_first_rejects_prose_fragment():
    readings = extract_spelled_out_name_first_levels("ABOVE THE FLOOR OF LOWEST\n+2.50 m")
    assert readings == []


# --- ROAD added to non-floor exclusion (confirmed on a second real drawing) ---

def test_road_level_excluded_from_floor_count():
    assert is_likely_non_floor_reference("ROAD LEVEL") is True


def test_end_to_end_sila_marine_floor_count():
    text = (
        "GROUND FLOOR FFL\n+0.30 m\nFIRST FLOOR FFL\n+4.80 m\n"
        "ROOF FLOOR SSL\n+8.40 m\nROAD LEVEL\n+0.00 m\nTOP ROOF\n+10.40 m\n"
    )
    result = extract_levels_from_text(text)
    # 5 levels found, but ROAD LEVEL excluded from the lift-relevant count
    assert len(result.levels) == 5
    assert result.floor_count == 4
    assert result.lowest_level == "GROUND FLOOR FFL"
    assert result.highest_level == "TOP ROOF"
