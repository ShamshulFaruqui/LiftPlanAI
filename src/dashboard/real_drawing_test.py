"""
Real Drawing Batch Test — runs the full pipeline against a folder of
REAL (non-synthetic) drawings and produces a summary report, WITHOUT
requiring ground truth (unlike every --evaluate flag elsewhere in this
project, which compares against the synthetic dataset's known-correct
values). This is the tool for the moment real drawings become available.

This is not a pass/fail test. Its job is to surface exactly what the
system extracted (or failed to extract) from drawings it has never seen
a format like before, so failures can be inspected and documented — this
IS the real-world validation this project has been missing, and finding
genuine failure modes here is a valuable dissertation result in itself,
not a bad outcome to hide.

Usage:
    python real_drawing_test.py --input ../../data/real_test --output ../../data/processed/real_test_report.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.triage.file_triage import triage_file
from src.dashboard.pipeline import run_pipeline


def run_batch_test(input_dir: Path, output_path: Path):
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        print("Drop your anonymised real drawings there and re-run this script.")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}\n")

    rows = []
    for pdf_path in pdf_files:
        print(f"--- {pdf_path.name} ---")

        # Stage 1: triage first, exactly like a real project folder would be processed.
        triage_result = triage_file(pdf_path)
        if not triage_result.is_relevant:
            if triage_result.error:
                # The file couldn't even be read — being marked "not relevant"
                # here does NOT mean it was checked and found irrelevant, it
                # means triage couldn't check it at all. Distinguishing this
                # from a genuine "not relevant" result matters: a real MuPDF
                # layer-config error was found doing exactly this during
                # development, and without this flag it looked identical to
                # a correctly-skipped structural/MEP drawing.
                print(f"  SKIPPED by triage — but NOT because it's irrelevant: "
                      f"the file could not be read ({triage_result.error}). "
                      f"This drawing was never actually checked for elevator content.")
            else:
                print("  SKIPPED by triage (not flagged elevator-relevant) — "
                      "if this SHOULD have been flagged, that's a real finding: "
                      "note which keywords this drawing uses that the lexicon doesn't cover yet.")
            rows.append({
                "file": pdf_path.name, "triage_relevant": False,
                "building_type": "", "floor_to_floor_mm": "", "total_travel_m": "",
                "group_count": "", "errors": triage_result.error, "flags": "",
            })
            continue

        result = run_pipeline(pdf_path)

        print(f"  Building type:     {result.building_type or 'NOT EXTRACTED'}")
        print(f"  Floor-to-floor:    {result.floor_to_floor_mm or 'NOT EXTRACTED'}")
        print(f"  Total travel:      {result.total_travel_m or 'NOT EXTRACTED'}")
        print(f"  Groups detected:   {len(result.groups)}")
        for g in result.groups:
            print(f"    - {g.group_id}: type={g.lift_type_from_label or '?'}, "
                  f"shaft={g.shaft_width_mm}x{g.shaft_depth_mm}mm, "
                  f"load/speed={g.rated_load_kg}kg/{g.rated_speed_ms}m/s")
        if result.extraction_flags:
            print(f"  Extraction flags:  {result.extraction_flags}")
        if result.errors:
            print(f"  ERRORS:            {result.errors}")

        rows.append({
            "file": pdf_path.name,
            "triage_relevant": True,
            "building_type": result.building_type or "",
            "floor_to_floor_mm": result.floor_to_floor_mm or "",
            "total_travel_m": result.total_travel_m or "",
            "group_count": len(result.groups),
            "errors": "; ".join(result.errors),
            "flags": "; ".join(result.extraction_flags),
        })
        print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    extracted_something = sum(1 for r in rows if r["building_type"] or r["group_count"])
    print("=" * 60)
    print(f"Processed {len(rows)} files — {extracted_something} yielded at least partial extraction.")
    print(f"Full report: {output_path}")
    print("=" * 60)
    print(
        "\nNEXT STEP: open the report and manually check each extracted value "
        "against the actual drawing. For your dissertation's evaluation chapter, "
        "record: (1) what extracted correctly, (2) what extracted wrong or not at "
        "all, and (3) WHY — a new label format, a scanned page with no text layer, "
        "a font issue, etc. That comparison IS your real-world validation result, "
        "whether the numbers look good or not."
    )


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Real Drawing Batch Test (no ground truth needed)")
    parser.add_argument("--input", type=str, default="data/real_test",
                         help="Folder containing real (anonymised) drawing PDFs")
    parser.add_argument("--output", type=str, default="data/processed/real_test_report.csv")
    args = parser.parse_args()

    run_batch_test(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
