"""
Pipeline Orchestration — wires together every working stage (triage,
merged text+visual extraction, rule-based classification, traffic study,
specification recommendation, and contradiction detection) into a single
per-drawing result, ready for the dashboard to render.

This is deliberately separated from the Streamlit UI (app.py) so the
orchestration logic itself can be unit tested without needing a running
web server — the same pattern used throughout this project (pure logic
functions with a thin CLI/UI wrapper on top).

Usage:
    python pipeline.py --drawing ../../data/synthetic/drawings/X.pdf --spec ../../data/synthetic/specs/Y_SPEC.pdf
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.parser.text_extractor import extract_parameters, extract_text
from src.parser.visual_extractor import extract_visual_parameters
from src.parser.merged_extractor import compare_extractions
from src.parser.spec_reader import extract_spec_parameters
from src.parser.level_extractor import extract_levels_from_text
from src.parser.real_lift_identifier import identify_real_lifts
from src.traffic_engine.minimum_requirements import propose_minimum_requirements
from src.classifier.rule_based_classifier import classify_group_id
from src.traffic_engine.traffic_study import run_traffic_study, estimate_population
from src.spec_engine.recommender import recommend_configuration
from src.parser.real_parameter_estimator import estimate_building_cluster
from src.contradiction.contradiction_check import check_contradictions

# Dubai Building Code firefighter-lift threshold. See the project
# proposal and README for the honest caveat on this figure: it comes from
# professional knowledge shared during development, not an independently
# verified code citation — verify the exact section reference before
# relying on it formally.
FFL_REQUIRED_TRAVEL_M = 23.0


@dataclass
class GroupSummary:
    group_id: str
    lift_type_from_label: str = None
    lift_type_from_id: str = None
    type_sources_agree: bool = None
    shaft_width_mm: int = None
    shaft_depth_mm: int = None
    rated_load_kg: int = None
    rated_speed_ms: float = None
    dedicated_shaft: bool = False
    traffic_study: dict = None
    recommendation: dict = None
    proposed_minimum_requirements: dict = None


@dataclass
class PipelineResult:
    drawing_file: str
    spec_file: str = None
    extraction_mode: str = "synthetic-format"  # or "real-drawing-format" — see run_pipeline
    building_type: str = None
    floor_to_floor_mm: int = None
    total_travel_m: float = None
    total_levels_visual: int = None
    estimated_population: int = None
    group_count_agrees: bool = None
    groups: list = field(default_factory=list)
    ffl_required: bool = False
    ffl_present: bool = False
    compliance_flags: list = field(default_factory=list)
    contradiction_flags: list = field(default_factory=list)
    extraction_flags: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    # Real-drawing-format fields, populated only when the synthetic-format
    # extraction (above) finds nothing — see run_pipeline's fallback logic.
    real_car_labels: list = field(default_factory=list)
    real_car_count: int = None
    real_car_count_is_uncertain: bool = False
    real_car_dimensions_m: tuple = None
    real_labeled_dimensions_mm: dict = field(default_factory=dict)
    real_service_lift_present: bool = False
    real_firefighter_duty_mentioned: bool = False
    real_level_readings: list = field(default_factory=list)
    real_level_anomaly_flags: list = field(default_factory=list)
    # Per-lift-group specification details (capacity, speed, stops,
    # travel, power, etc.) recovered from a "brief specification" table
    # embedded in the drawing itself, where one exists — see
    # real_lift_specification_extractor.py for why this is OCR-based and
    # what it's actually been verified against.
    real_lift_group_specs: list = field(default_factory=list)
    # A computed (not extracted) recommendation for real-drawing-format
    # results that had no specification table of their own — see
    # real_parameter_estimator.py for how, and its documented limits.
    real_computed_recommendation: dict = None
    # Proposed (not extracted) minimum pit depth / headroom — populated
    # only when a rated speed is available to base the proposal on, since
    # real drawing testing found these dimensions are typically not
    # stated in the drawing at all (see minimum_requirements.py).
    proposed_pit_depth: dict = None
    proposed_headroom: dict = None


def estimate_car_count(group_id: str) -> tuple:
    """
    Best-effort count of cars in a group from its ID, e.g. "EL01-EL02" -> 2.
    Real drawings sometimes truncate the label when there are more than two
    cars (e.g. "EL01-EL02..." meaning "at least these two, possibly more"),
    which this project's own synthetic generator does — so this returns
    (count, is_uncertain) rather than pretending truncated IDs give an
    exact answer. Downstream traffic study results for uncertain counts
    should be treated as a lower-bound estimate, not a precise figure.
    """
    if not group_id:
        return 1, True
    is_uncertain = "..." in group_id
    tokens = [t for t in group_id.replace("...", "").split("-") if t.strip()]
    count = max(1, len(tokens))
    return count, is_uncertain


def run_pipeline(drawing_path: Path, spec_path: Path = None) -> PipelineResult:
    result = PipelineResult(drawing_file=str(drawing_path),
                             spec_file=str(spec_path) if spec_path else None)

    # Text and visual extraction are attempted INDEPENDENTLY. They are two
    # deliberately separate channels (see the merged extractor's design
    # rationale) — a failure in one (e.g. Poppler missing, which only the
    # visual channel needs) must not discard results the other channel
    # already obtained successfully. An earlier version of this function
    # wrapped both in a single try/except, so a visual-channel failure
    # silently discarded successful text extraction too — found via real
    # (non-synthetic) drawing testing, where this actually happened.
    text_result = None
    visual_result = None
    raw_text = None

    try:
        raw_text = extract_text(drawing_path)
        text_result = extract_parameters(drawing_path, text=raw_text)
    except Exception as e:
        result.errors.append(f"Text extraction failed: {e}")

    try:
        visual_result = extract_visual_parameters(drawing_path)
    except Exception as e:
        result.errors.append(f"Visual extraction failed: {e}")

    if text_result is None and visual_result is None:
        return result  # both channels failed — nothing further to compute

    if text_result is not None:
        result.building_type = text_result.building_type
        result.floor_to_floor_mm = text_result.floor_to_floor_mm
        result.total_travel_m = text_result.total_travel_m

    # Fallback to the real-drawing-format extractors (level/elevation
    # listing + car-label/building-type inference) when the synthetic-
    # format extraction above found essentially nothing. This is the
    # real structural finding from testing against an actual (non-
    # synthetic) drawing: real drawings do not state a single "TOTAL
    # TRAVEL: Xm" value or use "(TYPE)" labels next to lift IDs at all —
    # see level_extractor.py and real_lift_identifier.py for the full
    # rationale. Both attempts run on the SAME already-extracted text
    # (raw_text), not a second OCR pass.
    synthetic_format_found_nothing = (
        result.building_type is None and result.total_travel_m is None and
        (text_result is None or not text_result.groups)
    )
    if synthetic_format_found_nothing and raw_text:
        try:
            level_result = extract_levels_from_text(raw_text, source_file=str(drawing_path))
            lift_id_result = identify_real_lifts(raw_text, source_file=str(drawing_path))

            if level_result.floor_count or lift_id_result.car_count:
                result.extraction_mode = "real-drawing-format"
                result.building_type = result.building_type or lift_id_result.inferred_building_type
                result.total_travel_m = result.total_travel_m or level_result.total_travel_m
                result.real_car_labels = lift_id_result.car_labels
                result.real_car_count = lift_id_result.car_count
                result.real_car_count_is_uncertain = lift_id_result.car_count_is_uncertain
                result.real_car_dimensions_m = lift_id_result.car_dimensions_m
                result.real_labeled_dimensions_mm = lift_id_result.labeled_dimensions_mm
                result.real_service_lift_present = lift_id_result.service_lift_present
                result.real_firefighter_duty_mentioned = lift_id_result.firefighter_duty_mentioned
                result.real_level_readings = level_result.levels
                result.real_level_anomaly_flags = level_result.anomaly_flags
                result.extraction_flags.append(
                    "Synthetic-format extraction found nothing — used the real-drawing-format "
                    "fallback (level/elevation listing + car-label inference) instead. See "
                    "real_level_readings and real_car_labels for what was found, and "
                    "real_level_anomaly_flags for any values worth manually verifying."
                )
                if lift_id_result.firefighter_duty_note:
                    result.extraction_flags.append(lift_id_result.firefighter_duty_note)
        except Exception as e:
            result.errors.append(f"Real-drawing-format fallback failed: {e}")

        # Separate try/except: this is OCR-based (see
        # real_lift_specification_extractor.py) and slower/less certain
        # than the text-layer extraction above — a failure here should
        # never take down the rest of a working real-drawing-format
        # result, only the specification-detail section of it.
        try:
            from src.parser.real_lift_specification_extractor import extract_lift_group_specs
            spec_groups = extract_lift_group_specs(drawing_path)
            if spec_groups:
                result.real_lift_group_specs = [
                    {"lift_ids": g.lift_ids, "fields": g.fields} for g in spec_groups
                ]
        except Exception as e:
            result.errors.append(f"Lift specification table extraction failed: {e}")

        # Fallback: no specification table was found (or found nothing
        # usable) in this drawing. Rather than show nothing, attempt a
        # computed recommendation from whatever building parameters were
        # reliably extracted — clearly a different, lower-confidence kind
        # of result than the OCR-read table above, and only attempted
        # when the level data available passes a plausibility check (see
        # real_parameter_estimator.py). This is expected to refuse more
        # often than it succeeds on real drawings tested so far; refusing
        # is the correct behaviour when the available data doesn't
        # support a confident estimate, not a failure of this fallback.
        if not result.real_lift_group_specs and result.building_type and result.real_level_readings:
            try:
                elevations = [r["elevation_mm"] for r in result.real_level_readings]
                estimate = estimate_building_cluster(elevations)
                if estimate is not None:
                    est_population = estimate_population(
                        floor_count=estimate.num_floors, building_type=result.building_type
                    )
                    rec = recommend_configuration(
                        num_floors=estimate.num_floors,
                        population=est_population,
                        floor_height_m=estimate.floor_height_m,
                        building_type=result.building_type,
                    )
                    result.real_computed_recommendation = {
                        "feasible": rec.feasible,
                        "message": rec.message,
                        "num_lifts": rec.num_lifts,
                        "rated_speed_ms": rec.rated_speed_ms,
                        "rated_load_kg": rec.rated_load_kg,
                        "estimated_floor_height_m": estimate.floor_height_m,
                        "estimated_num_floors": estimate.num_floors,
                        "estimation_warning": estimate.warning,
                    }
            except Exception as e:
                result.errors.append(f"Computed recommendation fallback failed: {e}")

    if visual_result is not None:
        result.total_levels_visual = visual_result.implied_total_levels

    if text_result is not None and visual_result is not None:
        merged = compare_extractions(text_result, visual_result, source_file=str(drawing_path))
        result.group_count_agrees = merged.group_count_agrees
        result.extraction_flags.extend(merged.flags)
    elif text_result is not None:
        result.extraction_flags.append(
            "Visual channel unavailable — proceeding with text-only extraction. "
            "Cross-validation was not possible."
        )
    elif visual_result is not None:
        result.extraction_flags.append(
            "Text channel unavailable — proceeding with visual-only extraction. "
            "Cross-validation was not possible."
        )

    if result.total_travel_m is not None and result.total_travel_m > FFL_REQUIRED_TRAVEL_M:
        result.ffl_required = True

    # Population estimation. In real-drawing-format mode, total_levels_visual
    # is NOT a reliable floor count to use — the visual/CV channel was
    # calibrated against this project's synthetic renderings and has never
    # been validated against real drawings (see the visual channel's known
    # limitations). Use the validated level_extractor floor count instead
    # when available.
    population_floor_count = result.total_levels_visual
    if result.extraction_mode == "real-drawing-format" and result.real_level_readings:
        population_floor_count = len(result.real_level_readings)

    if result.building_type and population_floor_count:
        result.estimated_population = estimate_population(
            floor_count=population_floor_count, building_type=result.building_type
        )

    for group in (text_result.groups if text_result is not None else []):
        summary = GroupSummary(
            group_id=group.group_id,
            lift_type_from_label=group.lift_type,
            lift_type_from_id=classify_group_id(group.group_id),
            shaft_width_mm=group.shaft_width_mm,
            shaft_depth_mm=group.shaft_depth_mm,
            rated_load_kg=group.rated_load_kg,
            rated_speed_ms=group.rated_speed_ms,
            dedicated_shaft=group.dedicated_shaft,
        )
        summary.type_sources_agree = (
            summary.lift_type_from_id is None or
            summary.lift_type_from_id == summary.lift_type_from_label
        )
        if not summary.type_sources_agree:
            result.extraction_flags.append(
                f"{group.group_id}: label says '{summary.lift_type_from_label}' but "
                f"group ID prefix implies '{summary.lift_type_from_id}' — flagged for review."
            )

        if group.lift_type == "firefighter" and group.dedicated_shaft:
            result.ffl_present = True

        # Traffic study — only possible where we have everything the
        # calculation needs (speed, load, floor geometry, population).
        if (summary.rated_speed_ms and summary.rated_load_kg and
                result.floor_to_floor_mm and result.total_levels_visual and
                result.estimated_population):
            car_count, count_uncertain = estimate_car_count(group.group_id)
            if count_uncertain:
                result.extraction_flags.append(
                    f"{group.group_id}: car count truncated in drawing label — "
                    f"using {car_count} as a lower-bound estimate for the traffic study; "
                    f"actual count may be higher. Verify against the drawing directly."
                )
            try:
                traffic = run_traffic_study(
                    num_floors=result.total_levels_visual,
                    population=result.estimated_population,
                    num_lifts=car_count,
                    rated_speed_ms=summary.rated_speed_ms,
                    rated_load_kg=summary.rated_load_kg,
                    floor_height_m=result.floor_to_floor_mm / 1000,
                )
                summary.traffic_study = asdict(traffic)
            except Exception as e:
                result.errors.append(f"Traffic study failed for {group.group_id}: {e}")

        # Proposed minimum pit depth / headroom — only needs a known
        # rated speed, unlike the full traffic study above (which also
        # needs floor count, population, and floor height). Real drawing
        # testing found pit depth and headroom are typically NOT stated
        # in the drawing at all — see minimum_requirements.py.
        if summary.rated_speed_ms:
            try:
                summary.proposed_minimum_requirements = propose_minimum_requirements(
                    rated_speed_ms=summary.rated_speed_ms
                )
            except Exception as e:
                result.errors.append(f"Minimum requirements proposal failed for {group.group_id}: {e}")

        result.groups.append(summary)

    # Real-drawing-format mode has no `groups` with an explicit dedicated-
    # shaft flag to check (see real_lift_identifier's design rationale —
    # asserting a distinct firefighter-labelled car would be guessing a
    # structure that may not exist). Firefighter duty being MENTIONED
    # somewhere in the drawing is treated as satisfying this check, with
    # its own caveat already attached via firefighter_duty_note.
    if result.extraction_mode == "real-drawing-format" and result.real_firefighter_duty_mentioned:
        result.ffl_present = True

    # Compliance check — FFL required vs present.
    if result.ffl_required and not result.ffl_present:
        result.compliance_flags.append(
            f"Dubai Building Code: total travel {result.total_travel_m}m exceeds "
            f"{FFL_REQUIRED_TRAVEL_M}m — a Firefighter Lift is typically required but none "
            f"was detected in this drawing. Verify with authority if an exemption applies."
        )
    elif result.ffl_required and result.ffl_present:
        result.compliance_flags.append(
            "Dubai Building Code: Firefighter Lift requirement (travel > 23m) — satisfied."
        )

    # Specification recommendation for the passenger group, if we have
    # enough information to run the search.
    passenger_group = next((g for g in result.groups if g.lift_type_from_label == "passenger"), None)
    if passenger_group and result.total_levels_visual and result.estimated_population and result.floor_to_floor_mm:
        try:
            rec = recommend_configuration(
                num_floors=result.total_levels_visual,
                population=result.estimated_population,
                floor_height_m=result.floor_to_floor_mm / 1000,
                building_type=result.building_type or "mixed-use",
            )
            passenger_group.recommendation = {
                "feasible": rec.feasible,
                "message": rec.message,
                "num_lifts": rec.num_lifts,
                "rated_speed_ms": rec.rated_speed_ms,
                "rated_load_kg": rec.rated_load_kg,
            }
        except Exception as e:
            result.errors.append(f"Recommendation failed: {e}")

    # Contradiction detection, if a spec document was provided.
    if spec_path is not None:
        try:
            contradiction_result = check_contradictions(drawing_path, spec_path)
            result.contradiction_flags = contradiction_result.flags
        except Exception as e:
            result.errors.append(f"Contradiction check failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Full Pipeline Orchestration")
    parser.add_argument("--drawing", type=str, required=True)
    parser.add_argument("--spec", type=str, default=None)
    args = parser.parse_args()

    result = run_pipeline(Path(args.drawing), Path(args.spec) if args.spec else None)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
