"""Regression tests for the dashboard's layout mode, written after testing
the dashboard on generated test sheet TEST-0471 (hotel, 70.8 m travel,
unseen 'PAX-n'/'SVC-n' labels):

  * unrecognised lift-like labels must be reported (never counted or typed);
  * an undimensioned shaft depth must stay None, not become a number;
  * no up-peak traffic study for service/goods/firefighter groups;
  * pit/headroom are proposed only when the drawing does not state them and
    the speed is within the planning data's range (<= 1.75 m/s);
  * the firefighter-lift warning mentions unrecognised labels.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic.generate_realistic import generate  # noqa: E402
from src.parser.layout_extractor import (  # noqa: E402
    TextLine, LayoutResult, LiftPrediction, find_unrecognised_labels, find_labels,
)
from src.dashboard.pipeline import (  # noqa: E402
    PipelineResult, _run_layout_mode, run_pipeline, PIT_PROPOSAL_MAX_SPEED_MS,
)


# ---------------------------------------------------------------- unit level
def _car(label, x=200.0, y=200.0):
    """A label placed like a car label: width above, depth rotated to its left."""
    return [TextLine(label, x - 15, y - 4, x + 15, y + 4),
            TextLine("1600", x - 10, y - 40, x + 10, y - 32),
            TextLine("1400", x - 45, y - 10, x - 37, y + 10, vertical=True)]


def test_unrecognised_label_inside_a_dimensioned_car_is_reported():
    assert find_labels([TextLine("PAX-1", 0, 0, 30, 8)]) == []
    assert find_unrecognised_labels(_car("PAX-1"), []) == ["PAX-1"]


def test_isolated_code_like_token_is_not_reported():
    # e.g. a distribution board reference on an electrical sheet
    assert find_unrecognised_labels([TextLine("DB-01", 100, 100, 130, 108)], []) == []


def test_unrecognised_schedule_row_is_reported():
    row = [TextLine("FRT-2", 700, 800, 730, 808), TextLine("2000 KG", 850, 800, 890, 808),
           TextLine("1.0 M/S", 930, 800, 960, 808)]
    assert find_unrecognised_labels(row, []) == ["FRT-2"]


def test_recognised_labels_are_never_reported():
    lines = _car("PL-01")
    recognised = find_labels(lines)
    assert [l.label for l in recognised] == ["PL-01"]
    assert find_unrecognised_labels(lines, recognised) == []


# ------------------------------------------------------------ pipeline level
def _layout(lift_type="passenger", speed=1.0, pit=None, overhead=None, unrecognised=()):
    lift = LiftPrediction(label="L-1", lift_type=lift_type, x=0, y=0, car_w=1600, car_d=1400,
                          door_w=900, shaft_w=2100, shaft_d=None, rated_load_kg=1000,
                          rated_speed_ms=speed)
    return LayoutResult(source="x.pdf", lifts=[lift], levels_m=[0.0, 3.5, 7.0, 10.5, 14.0],
                        num_stops=5, total_travel_m=14.0, typical_floor_to_floor_mm=3500,
                        pit_depth_mm=pit, overhead_mm=overhead, building_type="commercial",
                        ffl_required=False, ffl_present=False, is_lift_drawing=True,
                        unrecognised_labels=list(unrecognised))


def test_passenger_group_gets_traffic_study_and_recommendation():
    r = _run_layout_mode(PipelineResult(drawing_file="x.pdf"), _layout(), None)
    g = r.groups[0]
    assert g.traffic_study is not None and g.traffic_study_note is None
    assert g.recommendation is not None


@pytest.mark.parametrize("ltype", ["service", "goods", "firefighter"])
def test_non_passenger_groups_get_no_up_peak_study(ltype):
    g = _run_layout_mode(PipelineResult(drawing_file="x.pdf"), _layout(lift_type=ltype), None).groups[0]
    assert g.traffic_study is None
    assert g.traffic_study_note.startswith("Not applicable")


def test_undimensioned_shaft_depth_stays_none():
    g = _run_layout_mode(PipelineResult(drawing_file="x.pdf"), _layout(), None).groups[0]
    assert g.shaft_depth_mm is None and g.lifts[0]["shaft_d"] is None


def test_pit_proposed_only_when_drawing_is_silent_and_speed_in_range():
    in_range = _run_layout_mode(PipelineResult(drawing_file="x.pdf"), _layout(speed=1.0), None).groups[0]
    assert in_range.proposed_minimum_requirements is not None

    fast = _run_layout_mode(PipelineResult(drawing_file="x.pdf"),
                            _layout(speed=PIT_PROPOSAL_MAX_SPEED_MS + 1.75), None).groups[0]
    assert fast.proposed_minimum_requirements is None
    assert "above" in fast.minimum_requirements_note

    stated = _run_layout_mode(PipelineResult(drawing_file="x.pdf"),
                              _layout(pit=1500, overhead=4000), None)
    assert stated.groups[0].proposed_minimum_requirements is None
    assert stated.layout_pit_depth_mm == 1500 and stated.layout_overhead_mm == 4000


def test_unrecognised_labels_are_flagged_in_notes():
    r = _run_layout_mode(PipelineResult(drawing_file="x.pdf"), _layout(unrecognised=["PAX-1"]), None)
    assert r.unrecognised_labels == ["PAX-1"]
    assert any("PAX-1" in f for f in r.extraction_flags)


# ------------------------------------------- end to end: sheet TEST-0471
@pytest.fixture(scope="module")
def sheet_0471(tmp_path_factory):
    out = tmp_path_factory.mktemp("t0471")
    # Same arguments as the CLI's test split, so the sheet is identical to
    # the one the dashboard was tested on.
    generate("test", 500, 100, out, seed=2, start=471, stop=471)
    return next((out / "test" / "drawings").glob("*TEST-0471.pdf"))


def test_sheet_0471_end_to_end(sheet_0471):
    r = run_pipeline(sheet_0471)
    assert r.extraction_mode == "layout"
    assert r.building_type == "hotel" and r.total_travel_m == pytest.approx(70.8)
    assert r.layout_pit_depth_mm == 2550 and r.layout_overhead_mm == 5200
    assert r.unrecognised_labels == ["PAX-1", "PAX-2", "PAX-3", "PAX-4"]
    assert [g.lift_type_from_label for g in r.groups] == ["service"]
    svc = r.groups[0]
    assert svc.traffic_study is None and svc.shaft_depth_mm is None
    assert svc.rated_load_kg == 1000 and svc.rated_speed_ms == 3.5
    assert r.ffl_required and not r.ffl_present
    assert any("required but none was detected" in f and "PAX-1" in f for f in r.compliance_flags)
