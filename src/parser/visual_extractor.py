"""
Visual Parameter Extractor — Stage 2b of the LiftPlan AI pipeline.

Extracts shaft count, shaft widths, and floor count directly from the
RENDERED IMAGE of a drawing page using OpenCV geometry detection, rather
than the embedded text layer. This is the fallback/complementary channel
for drawings where the text layer is missing, unreliable, or the page is
a scanned image with no text layer at all.

IMPORTANT CALIBRATION NOTE: the pixel-to-millimetre conversion used here
is calibrated specifically to how src/synthetic/generate_dataset.py
renders drawings (a fixed, arbitrary points-per-mm scale, not a real
architectural drawing scale). Real drawings would need proper scale-bar
detection or a known reference dimension to calibrate this conversion —
this is a documented limitation, not something this module solves.

Also note: this synthetic generator does not preserve real proportional
floor-to-floor height in the vertical spacing of floor lines (it always
evenly distributes lines across a fixed vertical span regardless of the
true height) — so floor-to-floor height in mm cannot be recovered from
line spacing here, only the floor COUNT (number of lines) can be. Shaft
DEPTH is also not extractable visually, since a section view only shows
width, not depth — this mirrors how real section drawings work too.

Usage:
    python visual_extractor.py --input ../../data/synthetic/drawings/some_drawing.pdf
    python visual_extractor.py --evaluate --ground-truth ../../data/annotations/synthetic_ground_truth.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

try:
    from pdf2image import convert_from_path
except ImportError:
    sys.exit("Missing dependency: run `pip install -r requirements.txt` first (needs pdf2image).")

try:
    import cv2
except ImportError:
    sys.exit("Missing dependency: run `pip install -r requirements.txt` first (needs opencv-python).")


RENDER_DPI = 150

# Calibration constants matching src/synthetic/generate_dataset.py's
# rendering scale: shaft_w_pts = shaft_width_mm / 40, and pixels-per-point
# = DPI / 72. Combining these gives mm = pixels * (72 / DPI) * 40.
MM_PER_PIXEL = (72 / RENDER_DPI) * 40

# Pixel intensity thresholds (0=black, 255=white), calibrated by direct
# inspection of a rendered sample: shaft/text = near 0, floor lines =
# exactly 128 (the (0.5, 0.5, 0.5) draw colour), background = 255.
SHAFT_EDGE_THRESHOLD = 50
FLOOR_LINE_MIN = 100
FLOOR_LINE_MAX = 160

# Minimum shaft width in pixels to filter out noise/text fragments that
# might otherwise register as a thin dark region.
MIN_SHAFT_WIDTH_PX = 15
# Minimum aspect ratio (height / width) for a dark region to be considered
# a shaft outline rather than the title block (which is wide and short).
MIN_SHAFT_ASPECT_RATIO = 3.0


@dataclass
class VisualGroup:
    left_px: int
    right_px: int
    width_px: int
    width_mm: int


@dataclass
class VisualParameters:
    source_file: str
    image_width_px: int = 0
    image_height_px: int = 0
    shaft_count: int = 0
    shafts: list = field(default_factory=list)
    floor_line_count: int = 0
    implied_total_levels: int = None  # floor_line_count - 1


def render_page_to_grayscale(pdf_path: Path, dpi: int = RENDER_DPI) -> np.ndarray:
    """Render the first page of a PDF to a grayscale numpy array."""
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    image = np.array(pages[0].convert("L"))
    return image


def cluster_columns(columns: np.ndarray, gap_threshold: int = 3) -> list:
    """
    Group individual dark pixel column indices into clusters, merging any
    columns within `gap_threshold` pixels of each other. This collapses a
    1-2px anti-aliased line into a single detected edge position (the
    cluster's midpoint), rather than counting the same edge multiple times.
    """
    if len(columns) == 0:
        return []
    clusters = []
    current = [columns[0]]
    for col in columns[1:]:
        if col - current[-1] <= gap_threshold:
            current.append(col)
        else:
            clusters.append(int(np.mean(current)))
            current = [col]
    clusters.append(int(np.mean(current)))
    return clusters


def detect_shaft_rectangles(image: np.ndarray) -> list:
    """
    Detect vertical shaft outline edges by scanning a horizontal row safely
    within the shaft body's vertical extent (avoiding header labels and
    footer dimension text / title block), then pairing up detected edges
    left-to-right into (left, right) shaft boundaries.
    """
    h, w = image.shape
    scan_row = int(h * 0.45)  # comfortably within the shaft body, away from text
    row = image[scan_row, :]

    dark_cols = np.where(row < SHAFT_EDGE_THRESHOLD)[0]
    edges = cluster_columns(dark_cols)

    # Pair edges left-to-right: (edge0, edge1) = shaft 1, (edge2, edge3) = shaft 2, ...
    shafts = []
    for i in range(0, len(edges) - 1, 2):
        left, right = edges[i], edges[i + 1]
        width_px = right - left
        if width_px < MIN_SHAFT_WIDTH_PX:
            continue
        width_mm = int(round(width_px * MM_PER_PIXEL))
        shafts.append(VisualGroup(left_px=left, right_px=right, width_px=width_px, width_mm=width_mm))

    return shafts


def detect_shaft_vertical_extent(image: np.ndarray, left_px: int, right_px: int) -> tuple:
    """
    Find the top and bottom row of a shaft outline by looking for rows that
    are dark across most of the shaft's full width — a genuine horizontal
    border line spans the whole shaft, whereas nearby text (group labels
    above the shaft, dimension/load-speed text below it) only darkens a
    narrow column range and is rejected by the width-coverage check below.
    A single-column scan through the shaft centre was tried first and
    found to pick up exactly this kind of text contamination, extending
    the perceived shaft height well beyond its real border.
    """
    shaft_width = right_px - left_px
    min_coverage = 0.7  # a genuine border row is dark across most of the shaft width

    candidate_rows = []
    for row in range(image.shape[0]):
        strip = image[row, left_px:right_px]
        dark_fraction = np.mean(strip < SHAFT_EDGE_THRESHOLD)
        if dark_fraction >= min_coverage:
            candidate_rows.append(row)

    if not candidate_rows:
        return None, None

    # Cluster candidate rows into distinct horizontal lines, and take the
    # first two specifically as the shaft's top and bottom border. Other
    # elements below the shaft — notably the title block's own border,
    # which spans a similarly wide x-range — can also pass the width-
    # coverage check, so taking the *last* match (rather than the first
    # two) would incorrectly pick up the title block instead of the
    # shaft's actual bottom edge.
    clusters = cluster_columns(np.array(candidate_rows), gap_threshold=5)
    if len(clusters) < 2:
        return None, None
    return clusters[0], clusters[1]


def detect_floor_line_count(image: np.ndarray, shaft_top: int = None, shaft_bottom: int = None) -> int:
    """
    Count floor lines by scanning a vertical column positioned in the gap
    between the floor labels and the first shaft (where floor lines are
    visible but not overlapped by any shaft rectangle), looking for the
    specific gray intensity used for floor lines.

    If shaft_top/shaft_bottom are provided (from detect_shaft_vertical_extent),
    the scan is restricted to that vertical range, which reliably excludes
    header/footer text that would otherwise produce spurious matches at the
    same gray intensity — this is more robust than filtering by spacing
    pattern alone, which can be fooled by coincidental spacing matches or
    dominated by many closely-spaced spurious matches when there are only
    a few genuine floor lines.
    """
    h, w = image.shape
    scan_col = int(w * 0.11)  # calibrated to fall in the label/shaft gap
    column = image[:, scan_col]

    gray_rows = np.where((column >= FLOOR_LINE_MIN) & (column <= FLOOR_LINE_MAX))[0]

    if shaft_top is not None and shaft_bottom is not None:
        margin = 5  # avoid the shaft's own border pixels at the exact boundary
        gray_rows = gray_rows[(gray_rows >= shaft_top - margin) & (gray_rows <= shaft_bottom + margin)]

    lines = cluster_columns(gray_rows, gap_threshold=3)
    return len(lines)


def extract_visual_parameters(pdf_path: Path) -> VisualParameters:
    image = render_page_to_grayscale(pdf_path)
    h, w = image.shape

    shafts = detect_shaft_rectangles(image)

    shaft_top, shaft_bottom = None, None
    if shafts:
        shaft_top, shaft_bottom = detect_shaft_vertical_extent(image, shafts[0].left_px, shafts[0].right_px)

    floor_line_count = detect_floor_line_count(image, shaft_top, shaft_bottom)

    return VisualParameters(
        source_file=str(pdf_path),
        image_width_px=w,
        image_height_px=h,
        shaft_count=len(shafts),
        shafts=shafts,
        floor_line_count=floor_line_count,
        implied_total_levels=floor_line_count - 1 if floor_line_count > 0 else None,
    )


def evaluate_against_ground_truth(ground_truth_path: Path, verbose: bool = True) -> dict:
    """
    Run visual extraction against every project in the synthetic ground
    truth and compute accuracy metrics, mirroring the text extractor's
    evaluation approach for direct comparability between the two channels.
    """
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    metrics = {
        "total_projects": len(ground_truth),
        "shaft_count_correct": 0,
        "floor_count_correct": 0,
        "shaft_width_correct": 0,
        "shaft_width_total": 0,
        "failures": [],
    }

    for project in ground_truth:
        pdf_path = Path(project["drawing_path"])
        if not pdf_path.exists():
            metrics["failures"].append(f"Missing file: {pdf_path}")
            continue

        extracted = extract_visual_parameters(pdf_path)
        gt_group_count = len(project["groups"])
        gt_total_levels = project["floor_count"] + project["basement_count"]

        if extracted.shaft_count == gt_group_count:
            metrics["shaft_count_correct"] += 1

        if extracted.implied_total_levels == gt_total_levels:
            metrics["floor_count_correct"] += 1

        for i, gt_group in enumerate(project["groups"]):
            metrics["shaft_width_total"] += 1
            if i < len(extracted.shafts):
                if abs(extracted.shafts[i].width_mm - gt_group["shaft_width_mm"]) <= 50:
                    metrics["shaft_width_correct"] += 1

        if verbose:
            ok = (extracted.shaft_count == gt_group_count and
                  extracted.implied_total_levels == gt_total_levels)
            status = "OK" if ok else "MISMATCH"
            widths = [s.width_mm for s in extracted.shafts]
            gt_widths = [g["shaft_width_mm"] for g in project["groups"]]
            print(f"[{status:>8}] {project['project_id']} — "
                  f"shafts: {extracted.shaft_count} (gt {gt_group_count}), "
                  f"levels: {extracted.implied_total_levels} (gt {gt_total_levels}), "
                  f"widths: {widths} (gt {gt_widths})")

    n = metrics["total_projects"]
    print("\n" + "=" * 60)
    print("VISUAL EXTRACTION ACCURACY (against synthetic ground truth)")
    print("=" * 60)
    print(f"Shaft count:      {metrics['shaft_count_correct']}/{n} "
          f"({metrics['shaft_count_correct']/n*100:.1f}%)")
    print(f"Floor count:      {metrics['floor_count_correct']}/{n} "
          f"({metrics['floor_count_correct']/n*100:.1f}%)")
    if metrics["shaft_width_total"]:
        print(f"Shaft width acc.: {metrics['shaft_width_correct']}/{metrics['shaft_width_total']} "
              f"({metrics['shaft_width_correct']/metrics['shaft_width_total']*100:.1f}%) "
              f"[within 50mm tolerance]")
    if metrics["failures"]:
        print(f"\nFailures: {len(metrics['failures'])}")
        for f in metrics["failures"][:5]:
            print(f"  - {f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Stage 2b: Visual Parameter Extractor")
    parser.add_argument("--input", type=str, help="Single PDF to extract visual parameters from")
    parser.add_argument("--evaluate", action="store_true",
                         help="Run extraction against the full synthetic ground truth and report accuracy")
    parser.add_argument("--ground-truth", type=str, default="data/annotations/synthetic_ground_truth.json",
                         help="Path to ground truth JSON (used with --evaluate)")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_against_ground_truth(Path(args.ground_truth))
    elif args.input:
        result = extract_visual_parameters(Path(args.input))
        print(json.dumps(asdict(result), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
