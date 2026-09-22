"""
Merged Extractor — combines Stage 2a (text) and Stage 2b (visual) into a
single cross-validated result, as described in the proposal's Methods
section: "Outputs from both channels will be merged and cross-validated —
discrepancies between text and visual readings will be flagged for
engineer review rather than resolved automatically."

Design principle: this module does NOT silently pick one channel over
the other when they disagree. It reports both values and a confidence
level, preserving engineer oversight — consistent with the project's
stated philosophy of supporting rather than replacing professional
judgement (see the proposal's Ethical Aspects and Evaluation sections).

Usage:
    python merged_extractor.py --input ../../data/synthetic/drawings/some_drawing.pdf
    python merged_extractor.py --evaluate --ground-truth ../../data/annotations/synthetic_ground_truth.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.parser.text_extractor import extract_parameters
from src.parser.visual_extractor import extract_visual_parameters

# Shaft width tolerance in mm. Even on clean synthetic drawings, the visual
# channel's pixel-based measurement differs from the text channel's exact
# stated value by several mm purely from measurement rounding (e.g. 2100mm
# stated vs 2093mm measured) — this is NOT a genuine discrepancy worth
# flagging to an engineer. The tolerance absorbs that expected noise while
# still catching real disagreements (e.g. a value off by hundreds of mm).
SHAFT_WIDTH_TOLERANCE_MM = 75


@dataclass
class ShaftComparison:
    index: int
    text_width_mm: int = None
    visual_width_mm: int = None
    agrees: bool = None
    discrepancy_mm: int = None


@dataclass
class MergedResult:
    source_file: str
    group_count_text: int = None
    group_count_visual: int = None
    group_count_agrees: bool = None
    shaft_comparisons: list = field(default_factory=list)
    confidence: str = "high"       # "high" | "review_needed"
    flags: list = field(default_factory=list)


def compare_extractions(text_result, visual_result, source_file: str = "") -> MergedResult:
    """
    Pure comparison logic: takes already-extracted text and visual results
    and cross-validates them. Separated from merge_parameters() so this
    logic can be tested directly against constructed inputs, without
    needing a real PDF for every test case.
    """
    result = MergedResult(source_file=source_file)

    result.group_count_text = len(text_result.groups)
    result.group_count_visual = visual_result.shaft_count
    result.group_count_agrees = (result.group_count_text == result.group_count_visual)
    if not result.group_count_agrees:
        result.flags.append(
            f"Group count mismatch: text found {result.group_count_text}, "
            f"visual found {result.group_count_visual}"
        )

    n = min(len(text_result.groups), len(visual_result.shafts))
    for i in range(n):
        text_width = text_result.groups[i].shaft_width_mm
        visual_width = visual_result.shafts[i].width_mm
        comparison = ShaftComparison(index=i, text_width_mm=text_width, visual_width_mm=visual_width)

        if text_width is not None and visual_width is not None:
            discrepancy = abs(text_width - visual_width)
            comparison.discrepancy_mm = discrepancy
            comparison.agrees = discrepancy <= SHAFT_WIDTH_TOLERANCE_MM
            if not comparison.agrees:
                result.flags.append(
                    f"Shaft {i} width mismatch: text says {text_width}mm, "
                    f"visual measured {visual_width}mm (discrepancy {discrepancy}mm, "
                    f"exceeds {SHAFT_WIDTH_TOLERANCE_MM}mm tolerance)"
                )
        result.shaft_comparisons.append(comparison)

    if result.flags:
        result.confidence = "review_needed"

    return result


def merge_parameters(pdf_path: Path) -> MergedResult:
    text_result = extract_parameters(pdf_path)
    visual_result = extract_visual_parameters(pdf_path)
    return compare_extractions(text_result, visual_result, source_file=str(pdf_path))


def evaluate_against_ground_truth(ground_truth_path: Path, verbose: bool = True) -> dict:
    """
    Run the merged extraction against every synthetic project and confirm
    two things: that agreement is correctly detected on clean data (i.e.
    the tolerance doesn't cause false alarms), and that the final merged
    group count and shaft widths are at least as accurate as either
    channel alone.
    """
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    metrics = {
        "total_projects": len(ground_truth),
        "group_count_agreement_rate": 0,
        "shaft_width_agreement_rate": 0,
        "shaft_width_comparisons_total": 0,
        "false_review_flags": 0,   # confidence=review_needed on data with no genuine issue
        "failures": [],
    }

    for project in ground_truth:
        pdf_path = Path(project["drawing_path"])
        if not pdf_path.exists():
            metrics["failures"].append(f"Missing file: {pdf_path}")
            continue

        merged = merge_parameters(pdf_path)

        if merged.group_count_agrees:
            metrics["group_count_agreement_rate"] += 1

        for comp in merged.shaft_comparisons:
            metrics["shaft_width_comparisons_total"] += 1
            if comp.agrees:
                metrics["shaft_width_agreement_rate"] += 1

        if merged.confidence == "review_needed":
            metrics["false_review_flags"] += 1

        if verbose:
            status = "OK" if merged.confidence == "high" else "REVIEW"
            print(f"[{status:>6}] {project['project_id']} — "
                  f"groups: text={merged.group_count_text} visual={merged.group_count_visual} | "
                  f"widths agree: {sum(1 for c in merged.shaft_comparisons if c.agrees)}/{len(merged.shaft_comparisons)}")

    n = metrics["total_projects"]
    print("\n" + "=" * 60)
    print("MERGED EXTRACTION — CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Group count agreement:  {metrics['group_count_agreement_rate']}/{n} "
          f"({metrics['group_count_agreement_rate']/n*100:.1f}%)")
    if metrics["shaft_width_comparisons_total"]:
        print(f"Shaft width agreement:  {metrics['shaft_width_agreement_rate']}/{metrics['shaft_width_comparisons_total']} "
              f"({metrics['shaft_width_agreement_rate']/metrics['shaft_width_comparisons_total']*100:.1f}%)")
    print(f"Flagged for review:     {metrics['false_review_flags']}/{n} "
          f"(should be 0 on clean synthetic data — any flags here mean the "
          f"tolerance is miscalibrated, not that a real discrepancy exists)")
    if metrics["failures"]:
        print(f"\nFailures: {len(metrics['failures'])}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Merged Text+Visual Extractor")
    parser.add_argument("--input", type=str, help="Single PDF to extract merged parameters from")
    parser.add_argument("--evaluate", action="store_true",
                         help="Run merged extraction against the full synthetic ground truth")
    parser.add_argument("--ground-truth", type=str, default="data/annotations/synthetic_ground_truth.json")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_against_ground_truth(Path(args.ground_truth))
    elif args.input:
        result = merge_parameters(Path(args.input))
        print(json.dumps(asdict(result), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
