"""
Spatial Dimension Extractor — a THIRD extraction approach, built for a
problem plain-text OCR cannot solve: dimension VALUES (e.g. shaft width)
are connected to their LABELS ("CLEAR SHAFT WIDTH") visually, via
dimension lines and arrows on the drawing — not by textual adjacency.
Confirmed via real drawing testing: level_extractor.py and
real_lift_identifier.py work because their labels and values genuinely
sit next to each other in the OCR text stream; shaft width does not,
because the drawing conveys it through line-and-arrow geometry that
plain OCR text discards entirely.

This module uses pytesseract's word-level bounding-box output
(image_to_data) instead of plain text, and associates each label with
the nearest PLAUSIBLE numeric token by pixel proximity — specifically
prioritising horizontal (X-axis) alignment, since dimension text is
typically positioned directly above/below its dimension line, sharing
the line's X position while sitting some vertical distance away.

Confirmed against a real drawing during development: two separate
"SHAFT" label occurrences both had "2550" as their closest X-aligned
numeric neighbour (29px and 6px horizontal offset respectively) — a
highly plausible elevator shaft width, and consistent across both
occurrences on the same drawing.

Usage:
    python spatial_dimension_extractor.py --input ../../data/real_test/some_drawing.pdf
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Plausible ranges per dimension type, in mm, used to reject numeric
# neighbours that are clearly something else (a floor level, a room
# dimension, a drawing scale reference) even if they happen to be the
# closest number to a label. Elevator car depth uses the same broad
# range as shaft width — both are cabin/shaft cross-section dimensions
# of similar real-world scale (1.2m-3.5m) — this is a reasonable
# starting assumption, not independently verified per dimension type.
PLAUSIBLE_RANGES_MM = {
    "shaft_width": (1200, 3500),
    "car_depth": (1200, 3500),
}

# Search window, in pixels at the DPI used for OCR (see extract_labelled_dimensions),
# within which a numeric token is considered a candidate for a given label.
MAX_SEARCH_DISTANCE_X = 500
MAX_SEARCH_DISTANCE_Y = 350

LABEL_TARGETS = {
    "shaft_width": ["SHAFT", "WIDTH"],
    "car_depth": ["CAR", "DEPTH"],
}


@dataclass
class LabelledDimension:
    label: str
    value_mm: int
    label_position: tuple
    value_position: tuple
    horizontal_offset_px: int
    vertical_offset_px: int


@dataclass
class SpatialExtractionResult:
    source_file: str
    dimensions: list = field(default_factory=list)


def get_word_boxes_from_image(image) -> list:
    """
    Return word-level OCR results with bounding boxes for an already-
    rendered page image: a list of dicts with keys text, left, top,
    width, height.
    """
    try:
        import pytesseract
    except ImportError:
        raise RuntimeError("Spatial extraction requires pytesseract and the Tesseract OCR engine.")

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:
            words.append({
                "text": text, "left": data["left"][i], "top": data["top"][i],
                "width": data["width"][i], "height": data["height"][i],
            })
    return words


def find_label_positions(words: list, label_tokens: list) -> list:
    """
    Find positions where ALL of label_tokens appear as separate word
    tokens within a small proximity of each other (handles a multi-word
    label like "SHAFT" + "WIDTH" being recognised as separate OCR words,
    which is the normal case, not the exception).
    """
    positions = []
    first_token = label_tokens[0].upper()
    for i, w in enumerate(words):
        if w["text"].upper() != first_token:
            continue
        # Check that the remaining tokens appear within the next few words nearby.
        nearby = words[i:i + 6]
        nearby_texts = [nw["text"].upper() for nw in nearby]
        if all(tok in nearby_texts for tok in label_tokens[1:]):
            positions.append((w["left"], w["top"]))
    return positions


def find_nearest_plausible_number(words: list, label_pos: tuple,
                                   plausible_range: tuple) -> dict:
    """
    Find the numeric token closest to label_pos, prioritising small
    horizontal (X) offset — dimension text is typically X-aligned with
    its dimension line/label — while still requiring the value to fall
    within a physically plausible range for what's being measured.
    Returns None if nothing plausible is found nearby.
    """
    label_x, label_y = label_pos
    candidates = []

    for w in words:
        if not re.match(r"^\d{3,6}$", w["text"]):
            continue
        value = int(w["text"])
        if not (plausible_range[0] <= value <= plausible_range[1]):
            continue

        dx = abs(w["left"] - label_x)
        dy = abs(w["top"] - label_y)
        if dx > MAX_SEARCH_DISTANCE_X or dy > MAX_SEARCH_DISTANCE_Y:
            continue

        candidates.append((dx, dy, value, w))

    if not candidates:
        return None

    # Prioritise horizontal alignment (smallest dx), using dy as a tiebreaker.
    candidates.sort(key=lambda c: (c[0], c[1]))
    dx, dy, value, word = candidates[0]
    return {"value": value, "position": (word["left"], word["top"]), "dx": dx, "dy": dy}


def extract_labelled_dimensions(pdf_path: Path, dpi: int = 300, page_indices: list = None) -> SpatialExtractionResult:
    """
    Extracts every dimension type in LABEL_TARGETS from the specified
    pages (default: ALL pages, for maximum completeness).

    HONEST PERFORMANCE NOTE, based on real measurement, not assumption:
    an earlier version of this function tried to speed things up with a
    "cheap" pre-screening pass (plain text OCR first, to skip pages with
    no relevant keywords before running the expensive bounding-box OCR).
    Measured directly against the real drawing this module is built
    against, that optimisation did NOT help: reliable keyword detection
    itself required the same 300 DPI resolution as the full extraction
    (lower DPI — tested at 100, 150, 200 — either found nothing at all,
    or missed pages that 300 DPI correctly flagged), at a real cost of
    ~35s/page. Combined with the ~75s/page cost of the full extraction
    pass on whichever pages the screening flagged, the two-pass approach
    (140s screening + 225s selective extraction = 365s) was SLOWER than
    simply running full extraction on every page directly (4 pages ×
    75s = 300s). The prescreening idea was reverted after being tested
    and found not to deliver its assumed benefit — worth knowing if
    extending this module further, rather than re-attempting the same
    optimisation.

    Expect roughly 75 SECONDS PER PAGE at 300 DPI. For a large drawing
    set, pass page_indices to restrict processing to specific pages you
    already know contain relevant dimensions, if you want to trade
    completeness for speed deliberately.
    """
    result = SpatialExtractionResult(source_file=str(pdf_path))

    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise RuntimeError("Spatial extraction requires pdf2image and the Poppler system dependency.")

    all_pages = convert_from_path(str(pdf_path), dpi=dpi)
    indices_to_process = page_indices if page_indices is not None else range(len(all_pages))

    for i in indices_to_process:
        if i >= len(all_pages):
            continue
        page_image = all_pages[i]
        words = get_word_boxes_from_image(page_image)
        if not words:
            continue

        for label_name, tokens in LABEL_TARGETS.items():
            for label_pos in find_label_positions(words, tokens):
                plausible_range = PLAUSIBLE_RANGES_MM[label_name]
                match = find_nearest_plausible_number(words, label_pos, plausible_range)
                if match:
                    result.dimensions.append(LabelledDimension(
                        label=label_name,
                        value_mm=match["value"],
                        label_position=label_pos,
                        value_position=match["position"],
                        horizontal_offset_px=match["dx"],
                        vertical_offset_px=match["dy"],
                    ))

    return result


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Spatial Dimension Extractor")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pages", type=str, default=None,
                         help="Comma-separated 0-indexed page numbers to process, e.g. '1,3'. "
                              "Default: all pages (slow — see extract_labelled_dimensions docstring "
                              "for the real, measured per-page cost).")
    args = parser.parse_args()
    page_indices = [int(p) for p in args.pages.split(",")] if args.pages else None
    result = extract_labelled_dimensions(Path(args.input), dpi=args.dpi, page_indices=page_indices)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
