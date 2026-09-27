"""Tests for the realistic generator (v2), layout-aware extractor and
evaluation harness added in the dataset revision."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic.generate_realistic import generate  # noqa: E402
from src.parser.layout_extractor import (  # noqa: E402
    extract_layout, canonical_label, find_labels, parse_levels, TextLine,
)
from src.evaluation.evaluate import PRF, close, run  # noqa: E402


@pytest.fixture(scope="module")
def small_set(tmp_path_factory):
    out = tmp_path_factory.mktemp("v2")
    recs = generate("dev", 6, 2, out, seed=11, scan_rate=0.0)
    return out, recs


# ---------------- generator ----------------
def test_generator_writes_all_sheets_and_ground_truth(small_set):
    out, recs = small_set
    assert len(recs) == 8
    assert sum(r["is_lift_drawing"] for r in recs) == 6
    assert len(list((out / "dev" / "drawings").glob("*.pdf"))) == 8


def test_generator_is_deterministic_per_sheet(tmp_path):
    a = generate("dev", 3, 0, tmp_path / "a", seed=5, scan_rate=0.0)
    b = generate("dev", 3, 0, tmp_path / "b", seed=5, scan_rate=0.0)
    assert [r["lifts"] for r in a] == [r["lifts"] for r in b]


def test_generator_is_resumable(tmp_path):
    assert generate("dev", 4, 0, tmp_path, seed=3, scan_rate=0.0, stop=2) is None
    recs = generate("dev", 4, 0, tmp_path, seed=3, scan_rate=0.0)
    assert len(recs) == 4


def test_lifts_of_one_type_share_one_duty(small_set):
    _, recs = small_set
    for r in recs:
        by_type = {}
        for l in r["lifts"]:
            by_type.setdefault(l["lift_type"], set()).add((l["rated_load_kg"], l["rated_speed_ms"], l["car_w"]))
        assert all(len(v) == 1 for v in by_type.values())


def test_yolo_labels_are_normalised(small_set):
    out, _ = small_set
    for f in (out / "dev" / "labels").glob("*.txt"):
        for line in f.read_text().splitlines():
            c, *vals = line.split()
            assert c in ("0", "1", "2")
            assert all(0.0 <= float(v) <= 1.0 for v in vals)


# ---------------- layout extractor ----------------
def test_extractor_recovers_every_lift_on_vector_sheets(small_set):
    out, recs = small_set
    for r in recs:
        res = extract_layout(out / "dev" / "drawings" / r["filename"])
        got = {canonical_label(l.label) for l in res.lifts}
        want = {canonical_label(l["label"]) for l in r["lifts"]}
        assert got == want, r["drawing_id"]


def test_extractor_recovers_travel_and_stops(small_set):
    out, recs = small_set
    for r in [x for x in recs if x["is_lift_drawing"]]:
        res = extract_layout(out / "dev" / "drawings" / r["filename"])
        assert res.num_stops == len(r["levels"])
        assert res.total_travel_m == pytest.approx(r["total_travel_m"], abs=0.02)


def test_distractors_yield_no_lifts(small_set):
    out, recs = small_set
    for r in [x for x in recs if not x["is_lift_drawing"]]:
        assert extract_layout(out / "dev" / "drawings" / r["filename"]).lifts == []


@pytest.mark.parametrize("text,label,ltype", [
    ("PASSENGER LIFT 3", "PASSENGER LIFT 3", "passenger"),
    ("FFL-02", "FFL-02", "firefighter"),
    ("LIFT No.4 (SERVICE)", "LIFT NO.4 (SERVICE)", "service"),
    ("FREIGHT ELEVATOR E2", "FREIGHT ELEVATOR E2", "goods"),
])
def test_label_patterns(text, label, ltype):
    found = find_labels([TextLine(text, 0, 0, 100, 10)])
    assert len(found) == 1
    assert canonical_label(found[0].label) == canonical_label(label)
    assert found[0].lift_type == ltype


@pytest.mark.parametrize("text", ["LIFT LOBBY", "LIFT POWER ISOLATOR BY MEP", "FFL +3900", "LIFT PIT SUMP"])
def test_non_labels_are_not_lifts(text):
    assert find_labels([TextLine(text, 0, 0, 100, 10)]) == []


def test_unknown_abbreviation_is_not_guessed():
    # 'PAX' is not in the frozen vocabulary: the extractor must abstain
    assert find_labels([TextLine("PAX-1", 0, 0, 50, 10)]) == []


def test_level_parsing_handles_metres_and_millimetres():
    lines = [TextLine("F.F.L. +3.90", 500, 100, 560, 108), TextLine("FFL +7800", 500, 50, 560, 58),
             TextLine("FFL +0", 500, 150, 560, 158)]
    assert [v for v, _ in parse_levels(lines)] == [0.0, 3.9, 7.8]


# ---------------- evaluator ----------------
def test_prf_arithmetic():
    s = PRF()
    s.add(tp=8, fp=2, fn=2)
    assert (s.p, s.r) == (0.8, 0.8)
    assert s.f1 == pytest.approx(0.8)


def test_close_tolerances():
    assert close("total_travel_m", 33.61, 33.6)
    assert not close("total_travel_m", 33.7, 33.6)
    assert not close("car_w", 1400, 1350)


def test_harness_scores_perfect_on_clean_vector_set(small_set):
    out, _ = small_set
    rep = run(out, "dev", out_dir=None)
    assert rep["all"]["T2_lift_detection"]["f1"] == 1.0
    assert rep["all"]["T1_triage"]["f1"] == 1.0
