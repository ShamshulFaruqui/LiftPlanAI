"""
Tests for the real-drawing lift identifier (building type, car labels,
service/firefighter duty inference).
Run with: pytest tests/test_real_lift_identifier.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser.real_lift_identifier import (
    infer_building_type, normalise_car_label, extract_car_labels,
    detect_service_lift, detect_firefighter_duty, identify_real_lifts,
)


# --- infer_building_type ---

def test_office_keyword_infers_commercial():
    assert infer_building_type("ELLINGTON BUKADRA OFFICE, DUBAI") == "commercial"


def test_hotel_keyword_infers_hotel():
    assert infer_building_type("GRAND PALM HOTEL PROJECT") == "hotel"


def test_no_keyword_returns_none_not_a_guess():
    assert infer_building_type("SOME GENERIC PROJECT NAME") is None


# --- normalise_car_label: the real false-positive bug found ---

def test_normal_two_digit_label():
    assert normalise_car_label("06") == "L06"


def test_three_digit_label_with_spurious_extra_digit():
    # Real case found in OCR output: "L004" where the true label was "L04".
    assert normalise_car_label("004") == "L04"


def test_implausible_large_number_rejected():
    # Real false positive found: "L579" matched from OCR noise near an
    # unrelated dimension number, not a genuine car label.
    assert normalise_car_label("579") is None


def test_zero_rejected_as_implausible():
    # Real false positive found: "L 400" (a stray "L" near a 400mm
    # dimension) normalised to "L00" before this check was added.
    assert normalise_car_label("400") is None


# --- extract_car_labels: end-to-end with the real noise patterns ---

def test_extract_car_labels_excludes_ocr_noise():
    text = "L01 8 L-02 8 L-03 8 L004 8 SUMP ... 1409 L579 pod ... L 400 L 10950"
    labels = extract_car_labels(text)
    assert labels == ["L01", "L02", "L03", "L04"]
    assert "L79" not in labels
    assert "L00" not in labels


def test_extract_car_labels_deduplicates_repeated_mentions():
    # Real drawings repeat each car label once per floor plan view.
    text = "L06 LIFT LOBBY L06 LIFT LOBBY L06 CAT LADDER"
    labels = extract_car_labels(text)
    assert labels == ["L06"]


# --- detect_service_lift ---

def test_service_lift_detected_from_common_ocr_variants():
    assert detect_service_lift("SERV.LIFT") is True
    assert detect_service_lift("SERVLIFT") is True
    assert detect_service_lift("no mention here") is False


# --- detect_firefighter_duty ---

def test_firefighter_duty_detected_with_honest_caveat():
    mentioned, note = detect_firefighter_duty("PASSENGER AND FIRE SERVICE LIFT")
    assert mentioned is True
    assert "dual rating" in note.lower()


def test_no_firefighter_mention_returns_false():
    mentioned, note = detect_firefighter_duty("PASSENGER LIFT PLAN ONLY")
    assert mentioned is False
    assert note == ""


# --- End-to-end ---

def test_end_to_end_identification():
    text = (
        "ELLINGTON BUKADRA OFFICE, DUBAI\n"
        "PASSENGER AND FIRE SERVICE LIFT\n"
        "L01 L-02 L-03 L04 L05 L-06 L07 L-08\n"
        "SERV.LIFT\n"
    )
    result = identify_real_lifts(text)
    assert result.inferred_building_type == "commercial"
    assert result.car_count == 8
    assert result.service_lift_present is True
    assert result.firefighter_duty_mentioned is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


def test_villa_vs_residential_keyword_collision_fixed():
    # Real bug found and fixed: villa's original keyword list included
    # "PRIVATE RESIDENCE", which contains "RESIDENCE" as a substring —
    # colliding with residential's "RESIDENTIAL"/"RESIDENCE" keywords and
    # causing villas to be misclassified as residential due to dict
    # ordering. Fixed by removing the ambiguous overlapping phrases
    # entirely rather than relying on fragile ordering.
    assert infer_building_type("SMITH FAMILY PRIVATE RESIDENCE, DUBAI") is None
    assert infer_building_type("AL MAKTOUM VILLA PROJECT") == "villa"


def test_warehouse_keyword_inference():
    assert infer_building_type("DUBAI LOGISTICS WAREHOUSE FACILITY") == "warehouse"


def test_original_real_drawing_result_unaffected_by_new_keywords():
    # Regression guard: the one confirmed real-world result (this project's
    # actual test file) must not change due to the keyword additions.
    assert infer_building_type("ELLINGTON BUKADRA OFFICE, DUBAI") == "commercial"


# --- Bare "LIFT" label (villa: no numbered car ID at all) ---

def test_bare_lift_detected_when_no_numbered_labels():
    from src.parser.real_lift_identifier import identify_real_lifts
    text = "LOBBY LIFT VOID TO UP TERRACE"
    result = identify_real_lifts(text)
    assert result.car_labels == ["LIFT"]
    assert result.car_count == 1
    assert result.car_count_is_uncertain is True


def test_bare_lift_lobby_also_triggers_fallback():
    from src.parser.real_lift_identifier import identify_real_lifts
    # "LIFT LOBBY" (the real villa's exact phrasing) should still count
    # as evidence of a lift, not be excluded by the substring match.
    text = "SITTING AREA LIFT LOBBY VOID TO UP"
    result = identify_real_lifts(text)
    assert result.car_labels == ["LIFT"]
    assert result.car_count == 1


def test_numbered_labels_take_priority_over_bare_lift_fallback():
    from src.parser.real_lift_identifier import identify_real_lifts
    # Realistic case matching the actual office building's convention:
    # genuine numbered labels must win, not be silently replaced by the
    # bare-LIFT fallback just because the word "LIFT" also appears
    # nearby (e.g. in "LIFT LOBBY").
    text = "LIFT LOBBY L01 L02 PASSENGER LIFT GROUP"
    result = identify_real_lifts(text)
    assert result.car_labels == ["L01", "L02"]
    assert result.car_count == 2
    assert result.car_count_is_uncertain is False


def test_no_lift_mentioned_at_all_gives_no_car_count():
    from src.parser.real_lift_identifier import identify_real_lifts
    result = identify_real_lifts("STAIRCASE CORRIDOR RECEPTION")
    assert result.car_labels == []
    assert result.car_count is None


# --- Embedded car dimensions in the label (warehouse: "LIFT 2.0X1.6") ---

def test_embedded_dimensions_extracted_from_label():
    from src.parser.real_lift_identifier import extract_car_dimensions_from_label
    assert extract_car_dimensions_from_label("LIFT 2.0X1.6") == (2.0, 1.6)


def test_embedded_dimensions_case_insensitive_x():
    from src.parser.real_lift_identifier import extract_car_dimensions_from_label
    assert extract_car_dimensions_from_label("LIFT 2.0x1.6") == (2.0, 1.6)


def test_no_embedded_dimensions_returns_none():
    from src.parser.real_lift_identifier import extract_car_dimensions_from_label
    assert extract_car_dimensions_from_label("LIFT LOBBY VOID TO UP") is None


def test_embedded_dimensions_work_alongside_bare_lift_fallback():
    from src.parser.real_lift_identifier import identify_real_lifts
    # The real warehouse case: bare LIFT label with embedded dimensions,
    # both signals should be picked up together.
    text = "RECEPTION LOBBY LIFT 2.0X1.6 RECEPTION"
    result = identify_real_lifts(text)
    assert result.car_count == 1
    assert result.car_dimensions_m == (2.0, 1.6)


# --- Building type priority ordering (real hospital bug) ---

def test_hospital_correctly_identified_despite_generic_office_mentions():
    from src.parser.real_lift_identifier import infer_building_type
    # Real bug found and fixed: a hospital drawing's title block
    # mentions "Design Office" and "Support Office" (consultant role
    # titles), which used to be misclassified as "commercial" because
    # the generic "OFFICE" keyword was checked before "HOSPITAL" in
    # plain dict order.
    text = (
        "HAMDAN BIN RASHID CANCER HOSPITAL\n"
        "Local Support Office\n"
        "Lead Design Office\n"
    )
    assert infer_building_type(text) == "hospital"


def test_specific_types_checked_before_generic_commercial():
    from src.parser.real_lift_identifier import infer_building_type
    for building_type, keyword in [("hotel", "HOTEL"), ("warehouse", "WAREHOUSE"), ("villa", "VILLA")]:
        text = f"{keyword} PROJECT with a Design Office and Support Office"
        assert infer_building_type(text) == building_type


def test_commercial_still_correctly_identified_when_nothing_more_specific_present():
    from src.parser.real_lift_identifier import infer_building_type
    assert infer_building_type("ELLINGTON BUKADRA OFFICE, DUBAI") == "commercial"


# --- Bracket-range car ID notation (real hospital bug) ---

def test_bracket_range_extracts_correct_cars():
    from src.parser.real_lift_identifier import extract_bracket_range_cars
    text = "PUBLIC / VISITOR ELEVATORS [P1~P3] FLOOR PLAN"
    assert extract_bracket_range_cars(text) == ["P1", "P2", "P3"]


def test_bracket_range_takes_priority_over_floor_references():
    from src.parser.real_lift_identifier import extract_car_labels
    # The real, complete bug case: this exact text contains both the
    # genuine car range (P1~P3) AND unrelated floor-level references
    # using "L" + digit (L1, L3, L5, L7) that must NOT be counted as
    # additional cars. The generic pattern alone reported 9 "cars"
    # (every floor reference) against the real drawing before this fix.
    text = "PUBLIC / VISITOR ELEVATORS [P1~P3] FLOOR PLAN @ G, L1~L3, L5~L7"
    labels = extract_car_labels(text)
    assert labels == ["P1", "P2", "P3"]
    assert "L1" not in labels and "L3" not in labels


def test_no_bracket_range_falls_back_to_generic_pattern():
    from src.parser.real_lift_identifier import extract_car_labels
    # The office building's actual convention (individually labelled
    # cars, no bracket notation) must still work exactly as before.
    text = "L01 L-02 L-03 L004"
    assert extract_car_labels(text) == ["L01", "L02", "L03", "L04"]


def test_implausible_bracket_range_rejected():
    from src.parser.real_lift_identifier import extract_bracket_range_cars
    # A malformed or implausible range (e.g. reversed, or absurdly wide)
    # should be rejected rather than producing a nonsensical car list.
    assert extract_bracket_range_cars("ELEVATORS [P9~P1]") == []
    assert extract_bracket_range_cars("ELEVATORS [P1~P99]") == []


def test_end_to_end_hospital_identification():
    from src.parser.real_lift_identifier import identify_real_lifts
    text = (
        "HAMDAN BIN RASHID CANCER HOSPITAL\n"
        "Local Support Office\nLead Design Office\n"
        "PUBLIC / VISITOR ELEVATORS [P1~P3] FLOOR PLAN @ G, L1~L3, L5~L7\n"
    )
    result = identify_real_lifts(text)
    assert result.inferred_building_type == "hospital"
    assert result.car_labels == ["P1", "P2", "P3"]
    assert result.car_count == 3


# --- Dash-separated range notation and tilde-proximity floor exclusion (real hotel bugs) ---

def test_dash_separated_range_notation():
    from src.parser.real_lift_identifier import extract_bracket_range_cars
    # Real hotel convention: dash instead of brackets.
    assert extract_bracket_range_cars("ELEVATORS - HPL1 ~ HPL4") == ["HPL1", "HPL2", "HPL3", "HPL4"]


def test_bare_l_prefix_range_rejected_even_with_elevators_context():
    from src.parser.real_lift_identifier import extract_bracket_range_cars
    # Confirmed on two real drawings: bare "L" in a range always means
    # floor levels, never car IDs, even directly following "ELEVATORS".
    assert extract_bracket_range_cars("ELEVATORS - L1 ~ L8") == []


def test_comma_separated_floor_list_not_counted_as_cars():
    from src.parser.real_lift_identifier import extract_car_labels
    # Real bug found: a first fix only excluded the ONE "L" token
    # directly touching "~", missing every other comma-separated item
    # in the same floor list ("L1(G), L3, L4, L5" all survived
    # untouched before the proximity-based fix).
    text = "HPL1 & HPL4 - L1(G), L3, L4, L5, L6 ~ L16"
    labels = extract_car_labels(text)
    assert labels == []  # no valid car-range pattern in this isolated fragment
    assert "L1" not in labels and "L3" not in labels and "L16" not in labels


def test_full_hotel_pattern_gives_correct_cars_and_excludes_floor_list():
    from src.parser.real_lift_identifier import extract_car_labels
    text = (
        "HPL1 & HPL4 - L1(G), L3, L4, L5, L6 ~ L16\n"
        "ELEVATORS - HPL1 ~ HPL4\n"
    )
    assert extract_car_labels(text) == ["HPL1", "HPL2", "HPL3", "HPL4"]


def test_office_building_pattern_unaffected_by_tilde_proximity_check():
    from src.parser.real_lift_identifier import extract_car_labels
    # Regression guard: individually-labelled cars with no "~" anywhere
    # nearby must still work exactly as before.
    assert extract_car_labels("L01 L-02 L-03 L004") == ["L01", "L02", "L03", "L04"]


# --- Newline-crossing car ID false positive (real Sila Marine bug) ---

def test_ssl_and_ffl_trailing_l_not_matched_as_car_id():
    from src.parser.real_lift_identifier import extract_car_labels
    # Real bug found: "S.S.L\n8.400 m" and "F.F.L\n4.800 m" were
    # misread as car labels "L8" and "L4", since the original pattern's
    # separator allowed a newline between "L" and its following digit.
    text = "S.S.L\n8.400 m\nF.F.L\n4.800 m\nF.F.L\n0.300 m"
    assert extract_car_labels(text) == []


def test_car_label_still_matches_on_same_line_with_space_or_hyphen():
    from src.parser.real_lift_identifier import extract_car_labels
    # Regression guard: the fix must not break genuine same-line labels.
    assert extract_car_labels("L01 L-02 L 03") == ["L01", "L02", "L03"]


# --- Labeled dimension extraction (real Sila Marine drawing) ---

def test_labeled_dimensions_extracted_correctly():
    from src.parser.real_lift_identifier import extract_labeled_dimensions
    text = "CABIN WIDTH\n1500\nCABIN DEPTH\n1200\nSHAFT WIDTH\n2300\nPIT DEPTH\n2000"
    result = extract_labeled_dimensions(text)
    assert result == {
        "CABIN WIDTH": 1500, "CABIN DEPTH": 1200,
        "SHAFT WIDTH": 2300, "PIT DEPTH": 2000,
    }


def test_labeled_dimensions_different_lifts_get_different_values():
    from src.parser.real_lift_identifier import extract_labeled_dimensions
    # Confirmed real case: two lifts in the same project have the same
    # cabin size but genuinely different shaft width and pit depth.
    lift1_text = "SHAFT WIDTH\n2300\nPIT DEPTH\n2000"
    lift2_text = "SHAFT WIDTH\n1800\nPIT DEPTH\n1000"
    assert extract_labeled_dimensions(lift1_text)["SHAFT WIDTH"] == 2300
    assert extract_labeled_dimensions(lift2_text)["SHAFT WIDTH"] == 1800
    assert extract_labeled_dimensions(lift1_text)["PIT DEPTH"] == 2000
    assert extract_labeled_dimensions(lift2_text)["PIT DEPTH"] == 1000


def test_labeled_dimensions_keeps_first_occurrence_of_duplicates():
    from src.parser.real_lift_identifier import extract_labeled_dimensions
    # Real drawings repeat the same label across multiple plan views —
    # the first occurrence should be kept, not overwritten.
    text = "CABIN WIDTH\n1500\nOTHER TEXT\nCABIN WIDTH\n1500"
    assert extract_labeled_dimensions(text)["CABIN WIDTH"] == 1500


def test_labeled_dimensions_ignores_unknown_labels():
    from src.parser.real_lift_identifier import extract_labeled_dimensions
    assert extract_labeled_dimensions("RANDOM LABEL\n1500") == {}


def test_labeled_dimensions_ignores_non_numeric_next_line():
    from src.parser.real_lift_identifier import extract_labeled_dimensions
    assert extract_labeled_dimensions("CABIN WIDTH\nNOT A NUMBER") == {}
