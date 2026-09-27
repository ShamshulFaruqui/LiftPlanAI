"""
Tests for the rule-based lift group classifier (Stage 3).
Run with: pytest tests/test_rule_based_classifier.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.rule_based_classifier import classify_group_id


# --- Correct classification on known conventions (matches synthetic generator) ---

def test_firefighter_prefix():
    assert classify_group_id("FFL01") == "firefighter"


def test_service_prefix():
    assert classify_group_id("SL01") == "service"


def test_passenger_prefix():
    assert classify_group_id("EL01") == "passenger"


def test_goods_prefix():
    assert classify_group_id("GL01") == "goods"


def test_compound_group_id_uses_first_token():
    assert classify_group_id("EL01-EL02") == "passenger"


def test_truncated_compound_group_id_with_ellipsis():
    assert classify_group_id("EL01-EL02...") == "passenger"


def test_case_insensitive():
    assert classify_group_id("ffl01") == "firefighter"
    assert classify_group_id("fFl01") == "firefighter"


# --- Safe failure on ambiguous or unrecognised conventions ---
# These matter as much as the correct cases: a wrong guess is worse than
# admitting uncertainty, since it could silently mislead an engineer.
# "P" is deliberately NOT mapped to passenger, mirroring the earlier
# lesson from the triage keyword lexicon where "PL"/"EL" collided with
# unrelated discipline codes — "P" alone is even more ambiguous (could
# be Parking, Plant, or Passenger) and is left unclassified rather than
# guessed.

def test_ambiguous_bare_p_prefix_not_guessed():
    assert classify_group_id("P01") is None


def test_generic_bare_l_prefix_not_guessed():
    # "L" alone doesn't specify a lift type in real drawing conventions.
    assert classify_group_id("L1") is None


def test_bare_numeric_id_not_guessed():
    assert classify_group_id("1") is None


def test_spelled_out_label_not_guessed():
    # This classifier works on abbreviated prefixes, not full words —
    # a spelled-out ID like this would need the text-label channel instead.
    assert classify_group_id("LIFT-01") is None


def test_ambiguous_fire_service_abbreviation_not_guessed():
    # Could plausibly mean either "Fire Service" or something else;
    # not a recognised prefix, so it should not be silently guessed.
    assert classify_group_id("FS01") is None


def test_empty_string_returns_none():
    assert classify_group_id("") is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
