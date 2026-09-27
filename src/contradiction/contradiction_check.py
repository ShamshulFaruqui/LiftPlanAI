"""
Contradiction Detector — Stage: cross-references client specification
documents against architectural drawings, flagging discrepancies for
engineer review. The drawing is always treated as the authoritative
source, matching established industry practice (and the proposal's
Methods section: "drawings treated as the authoritative source
throughout").

SCOPE — read before relying on this. Only two fields are compared:
total_travel_m and the primary (first) group's shaft_width_mm. Floor
count comparison is deliberately NOT attempted here, despite being the
most common contradiction type in this project's synthetic test data
(3 of 5 cases) — because a genuine ambiguity was found rather than
papered over: the drawing generator's own visual floor-labelling scheme
relabels the top numbered floor as "RF" (roof), so counting floor labels
in the drawing text undercounts by one relative to what the spec
document means by "number of floors." Resolving this needs either a
consistent floor-counting convention added to the drawing text extractor
or a dedicated floor-count field, neither of which exists yet — a
genuine limitation, stated here rather than hidden behind a fragile
guess.

Usage:
    python contradiction_check.py --drawing ../../data/synthetic/drawings/X.pdf --spec ../../data/synthetic/specs/Y_SPEC.pdf
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.parser.text_extractor import extract_parameters
from src.parser.spec_reader import extract_spec_parameters

# Tolerances absorb expected measurement/rounding noise (e.g. a spec
# document stating "8.30m" for a drawing value of 8.295m), consistent
# with the same principle applied in the merged text+visual extractor —
# only flag discrepancies large enough to plausibly be a genuine error,
# not floating-point or unit-rounding artefacts.
TRAVEL_TOLERANCE_M = 0.15
SHAFT_WIDTH_TOLERANCE_MM = 25


@dataclass
class FieldComparison:
    field: str
    drawing_value: float = None
    spec_value: float = None
    contradicts: bool = None
    authoritative_value: float = None  # always the drawing's value, per industry practice


@dataclass
class ContradictionResult:
    drawing_file: str
    spec_file: str
    comparisons: list = field(default_factory=list)
    has_contradiction: bool = False
    flags: list = field(default_factory=list)
    spec_extraction_uncertain: bool = False


def compare_field(name: str, drawing_value, spec_value, tolerance: float) -> FieldComparison:
    comparison = FieldComparison(field=name, drawing_value=drawing_value, spec_value=spec_value)
    if drawing_value is not None:
        comparison.authoritative_value = drawing_value  # drawing wins, always

    if drawing_value is not None and spec_value is not None:
        comparison.contradicts = abs(drawing_value - spec_value) > tolerance

    return comparison


def check_contradictions(drawing_path: Path, spec_path: Path) -> ContradictionResult:
    drawing = extract_parameters(drawing_path)
    spec = extract_spec_parameters(spec_path)

    result = ContradictionResult(drawing_file=str(drawing_path), spec_file=str(spec_path))

    drawing_shaft_width = drawing.groups[0].shaft_width_mm if drawing.groups else None

    travel_comparison = compare_field("total_travel_m", drawing.total_travel_m,
                                       spec.total_travel_m, TRAVEL_TOLERANCE_M)
    width_comparison = compare_field("shaft_width_mm", drawing_shaft_width,
                                      spec.shaft_width_mm, SHAFT_WIDTH_TOLERANCE_MM)

    result.comparisons = [travel_comparison, width_comparison]

    # Found via testing against a real (non-synthetic) specification
    # document: spec_reader.py's patterns are built for this project's own
    # synthetic spec format ("Total travel: X m", etc.) and matched nothing
    # at all in a real, table-formatted specification sheet — every field
    # came back None. Without this check, that reads identically to "spec
    # data confirms no contradiction" (has_contradiction=False, flags=[]),
    # when in truth nothing was actually compared. A real-spec-format
    # reader, symmetric to the real-drawing-format fallback already built
    # for drawings, would be needed to genuinely check a real specification
    # document — not attempted here; this only ensures the failure is
    # reported rather than mistaken for a clean result.
    if spec.total_travel_m is None and spec.shaft_width_mm is None:
        result.spec_extraction_uncertain = True
        result.flags.append(
            "Specification document could not be read in the expected format "
            "(total travel and shaft width were not found in it) — no "
            "meaningful contradiction check was performed against this "
            "document. Verify the specification manually."
        )

    for comparison in result.comparisons:
        if comparison.contradicts:
            result.has_contradiction = True
            result.flags.append(
                f"{comparison.field}: drawing states {comparison.drawing_value}, "
                f"specification states {comparison.spec_value} — drawing is authoritative; "
                f"flagged for engineer review."
            )

    return result


def evaluate_against_ground_truth(ground_truth_path: Path, verbose: bool = True) -> dict:
    """
    Confirm the detector catches the genuine contradictions it's scoped
    to catch (total_travel_m, shaft_width_mm), and does NOT raise a false
    alarm on projects with no contradiction, or on the field(s) it isn't
    scoped to check (e.g. a floor_count-only contradiction should produce
    no flags here, since that field isn't compared).
    """
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    metrics = {"total": 0, "correctly_flagged": 0, "correctly_clean": 0,
               "false_positives": 0, "false_negatives": 0, "out_of_scope_correctly_ignored": 0}

    for project in ground_truth:
        drawing_path = Path(project["drawing_path"])
        spec_path = Path(project["spec_path"])
        if not drawing_path.exists() or not spec_path.exists():
            continue

        metrics["total"] += 1
        result = check_contradictions(drawing_path, spec_path)

        contradiction = project.get("spec_contradiction")
        in_scope_fields = {"total_travel_m", "shaft_width_mm"}

        if contradiction is None:
            if result.has_contradiction:
                metrics["false_positives"] += 1
                if verbose:
                    print(f"[FALSE POS] {project['project_id']}: {result.flags}")
            else:
                metrics["correctly_clean"] += 1
        elif contradiction["field"] in in_scope_fields:
            if result.has_contradiction:
                metrics["correctly_flagged"] += 1
            else:
                metrics["false_negatives"] += 1
                if verbose:
                    print(f"[MISSED] {project['project_id']}: expected contradiction in "
                          f"{contradiction['field']}")
        else:
            # Out-of-scope contradiction (e.g. floor_count) — correct
            # behaviour is to NOT flag it, since it's not compared here.
            if not result.has_contradiction:
                metrics["out_of_scope_correctly_ignored"] += 1
            else:
                # Flagged something, but for the wrong reason — still log it.
                metrics["false_positives"] += 1

    print("=" * 60)
    print("CONTRADICTION DETECTOR — VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total projects checked:          {metrics['total']}")
    print(f"Correctly flagged (in-scope):     {metrics['correctly_flagged']}")
    print(f"Correctly clean (no contradiction): {metrics['correctly_clean']}")
    print(f"Correctly ignored (out-of-scope):  {metrics['out_of_scope_correctly_ignored']}")
    print(f"False positives:                  {metrics['false_positives']}")
    print(f"False negatives (missed, in-scope): {metrics['false_negatives']}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Contradiction Detector")
    parser.add_argument("--drawing", type=str, help="Path to the drawing PDF")
    parser.add_argument("--spec", type=str, help="Path to the specification PDF")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--ground-truth", type=str, default="data/annotations/synthetic_ground_truth.json")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_against_ground_truth(Path(args.ground_truth))
    elif args.drawing and args.spec:
        result = check_contradictions(Path(args.drawing), Path(args.spec))
        print(json.dumps(asdict(result), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
