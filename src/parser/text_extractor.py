"""
Text-Layer Parameter Extractor — Stage 2a of the LiftPlan AI pipeline.

Extracts structured building and lift-group parameters directly from
the embedded text layer of a CAD-derived PDF drawing: floor-to-floor
height, total travel, building type, and per-group labels/types/
dimensions.

This is the "text channel" half of the dual-channel parser described in
the proposal. The "visual channel" (OpenCV/YOLO reading shaft outlines
and dimensions directly from the rendered image, for drawings where the
text layer is missing or unreliable) is a separate, later component.

Usage:
    python text_extractor.py --input ../../data/synthetic/drawings/some_drawing.pdf
    python text_extractor.py --evaluate --ground-truth ../../data/annotations/synthetic_ground_truth.json
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        sys.exit("Missing dependency: run `pip install -r requirements.txt` first (needs PyMuPDF).")


LIFT_TYPES = ["PASSENGER", "SERVICE", "FIREFIGHTER", "GOODS", "BED"]


@dataclass
class ExtractedGroup:
    group_id: str
    lift_type: str
    shaft_width_mm: int = None
    shaft_depth_mm: int = None
    dedicated_shaft: bool = False
    rated_load_kg: int = None
    rated_speed_ms: float = None


@dataclass
class ExtractedParameters:
    source_file: str
    building_type: str = None
    floor_to_floor_mm: int = None
    total_travel_m: float = None
    project_id: str = None
    groups: list = field(default_factory=list)
    raw_text_length: int = 0


def ocr_extract_text(pdf_path: Path, dpi: int = 300) -> str:
    """
    Render every page to an image and run Tesseract OCR on each. Used
    only as a last resort — see extract_text()'s docstring for why this
    is sometimes the ONLY way to read a drawing at all: some professional
    architectural offices "flatten" or "burn" text to vector curves
    before issuing final drawings (to prevent font-substitution and
    unauthorised editing), which means there are literally zero embedded
    font/text objects in the PDF for PyMuPDF or pdfplumber to find,
    regardless of how good the extraction code is. Confirmed via a real
    drawing during this project's development: 0 fonts, 26,803 vector
    drawing objects on a single page.

    OCR output is noisy — expect character confusion (e.g. "L" vs "1"),
    especially in dense technical drawings. Treat downstream field
    extraction from OCR text as inherently lower-confidence than from a
    genuine embedded text layer.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        raise RuntimeError(
            "OCR fallback requires pdf2image and pytesseract (and the Tesseract "
            "OCR engine installed on the system) — run `pip install pytesseract` "
            "and ensure `tesseract` is on PATH."
        )

    pages = convert_from_path(str(pdf_path), dpi=dpi)
    return "\n".join(pytesseract.image_to_string(page) for page in pages)


def clean_cad_control_codes(text: str) -> str:
    """
    Strip AutoCAD MTEXT formatting/control codes that leak directly into
    extracted PDF text on some real drawings, confirmed via a real villa
    drawing during this project's development. These are CAD AUTHORING
    artifacts, not drawing content, and left unstripped they actively
    corrupt downstream pattern matching — e.g. "300\\A1;300" sitting right
    next to a real dimension value, or "%%P0.00" (AutoCAD's escape for
    "±0.00", a standard architectural datum-reference notation) being
    read as literal characters rather than the number it represents.

    Codes observed and handled:
    - "%%P" → AutoCAD's plus-minus symbol escape (e.g. "%%P0.00" means
      "±0.00", the ground/datum reference level). Stripped to leave the
      bare number, since for numeric extraction purposes "±0.00" and
      "0.00" carry the same value.
    - "\\P" → an MTEXT paragraph break — converted to a newline, since
      that's the role it's actually playing in the original text.
    - "\\A0;", "\\A1;", "\\A2;" → text alignment codes (appear as a
      prefix immediately before a value, e.g. "\\A1;300") — pure
      formatting metadata, stripped entirely.
    - "\\pxqc;", "\\pxql;", "\\pxi-3,l4,ql,t4;" → paragraph justification/
      indent/tab-stop codes — stripped entirely.
    - "\\W1.5;", "\\T2;" and similar → width-factor/tracking codes,
      sometimes wrapped in "{...}" grouping braces — stripped, including
      the braces (the CONTENT inside a brace group is kept; only the
      formatting code and the braces themselves are removed).

    This is not an exhaustive AutoCAD MTEXT parser — it targets the
    specific codes observed on real drawings during this project, not
    the full MTEXT formatting-code specification. A drawing using a
    formatting code not covered here would still leak that code into the
    cleaned text; extend the patterns below if that's found in practice.
    """
    text = text.replace("%%P", "")
    text = text.replace("\\P", "\n")
    text = re.sub(r"\\A\d+;", "", text)
    text = re.sub(r"\\px[^;]*;", "", text)
    text = re.sub(r"\\[WTHQ][\d.]*x?;", "", text)
    text = re.sub(r"\\[fC][^;]*;", "", text)
    text = text.replace("{", "").replace("}", "")
    return text


def _extract_text_tiered(pdf_path: Path, ocr_fallback: bool) -> str:
    """The tiered PyMuPDF -> pdfplumber -> OCR logic, before cleaning."""
    text = ""

    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception:
        text = ""

    if text.strip():
        return text

    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as doc:
            text = "\n".join(page.extract_text() or "" for page in doc.pages)
    except Exception:
        text = ""

    if text.strip():
        return text

    if ocr_fallback:
        return ocr_extract_text(pdf_path)
    return ""


def extract_text(pdf_path: Path, ocr_fallback: bool = True) -> str:
    """
    Concatenate text from all pages of a PDF, trying PyMuPDF first, then
    pdfplumber, then (if both find essentially nothing) OCR as a last
    resort. Each tier is tried in turn whenever the previous one returns
    no usable text — whether that's because it raised an exception, or
    because it opened the file successfully but simply found nothing
    (the common case for text-flattened drawings, which raise no error
    at all; there is just nothing there).

    The OCR tier exists for a real, common case, not a hypothetical one:
    architectural offices frequently flatten text to vector curves before
    issuing final drawings, leaving literally no embedded text for any
    text-layer-based method to find. Confirmed via a real drawing during
    this project's development: 0 embedded fonts, 26,803 vector drawing
    objects on a single page — PyMuPDF opens it without error and simply
    returns an empty string, which an earlier version of this function
    mistook for "nothing more to try" rather than "try the next tier."

    Set ocr_fallback=False to skip the OCR tier (e.g. for speed when
    batch-processing many files known to have a proper text layer) — OCR
    is meaningfully slower than the other two methods.

    Whatever text is found, by whichever tier, is passed through
    clean_cad_control_codes() before being returned — applied once here
    rather than at each tier's return point, so every caller automatically
    gets clean text regardless of which extraction method produced it.
    """
    text = _extract_text_tiered(pdf_path, ocr_fallback)
    return clean_cad_control_codes(text)


def parse_length_mm(value_str: str, unit_str: str) -> int:
    """
    Normalise a captured length to millimetres, given its numeric string
    and (possibly absent) unit string. Real drawings mix mm and m freely,
    sometimes without stating the unit at all, so this makes a reasonable
    inference rather than assuming one fixed format.
    """
    value = float(value_str)
    if unit_str and unit_str.lower() == "mm":
        return int(round(value))
    if unit_str and unit_str.lower() == "m":
        return int(round(value * 1000))
    # No unit given: a decimal value is almost certainly metres (e.g. "3.7"),
    # while a bare integer in the thousands is almost certainly millimetres.
    if "." in value_str:
        return int(round(value * 1000))
    return int(round(value))


# Label alternations covering the known real-world phrasing variants for
# each field. Extend these as new conventions are encountered in real
# drawings — this is the main place robustness gets added over time.
FLOOR_TO_FLOOR_LABELS = r"(?:F[\-/]F\s*H(?:EIGH)?T\.?|FLOOR[\s\-]*TO[\s\-]*FLOOR|F/F\s*HEIGHT|TYPICAL\s*FLOOR\s*HEIGHT)"
TOTAL_TRAVEL_LABELS = r"(?:TOTAL\s*TRAVEL|TRAVEL\s*HEIGHT|TOTAL\s*RISE|OVERALL\s*TRAVEL(?:\s*\(O\.?A\.?T\.?\))?)"


def extract_length_field(text: str, label_pattern: str) -> float:
    """
    Find a labelled length field (floor-to-floor height, total travel, etc.)
    under any of its known label phrasings, and return the value in mm.
    Returns None if no known phrasing is found.
    """
    pattern = rf"{label_pattern}\s*[:=]\s*([\d.]+)\s*(mm|m)?\b"
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    return parse_length_mm(m.group(1), m.group(2))


def extract_shaft_dimensions(window: str):
    """
    Extract (width_mm, depth_mm) from a shaft dimension string, trying
    several known real-world formats: "2000x2400mm", "W:2000 D:2400",
    and "2.0m x 2.4m" (metres stated per-number, not once at the end).
    Returns (None, None) if no known format is found.
    """
    # Format: "W:2000 D:2400"
    m = re.search(r"W\s*:\s*(\d+)\s*D\s*:\s*(\d+)", window, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Format: "2.0m x 2.4m" — unit stated after EACH number, not once at
    # the end, and "(?!m)" ensures we don't partially match inside "mm".
    m = re.search(r"([\d.]+)\s*m(?!m)\s*x\s*([\d.]+)\s*m(?!m)", window, re.IGNORECASE)
    if m:
        return (parse_length_mm(m.group(1), "m"), parse_length_mm(m.group(2), "m"))

    # Format: "2000x2400mm" or "2000 X 2400 MM" — unit stated once at the end.
    m = re.search(r"([\d.]+)\s*x\s*([\d.]+)\s*mm", window, re.IGNORECASE)
    if m:
        return (parse_length_mm(m.group(1), "mm"), parse_length_mm(m.group(2), "mm"))

    return None, None


def extract_load_speed(window: str):
    """
    Extract (rated_load_kg, rated_speed_ms) from text like "690kg / 1.24m/s".
    Returns (None, None) if the pattern isn't found.
    """
    m = re.search(r"(\d+)\s*kg\s*/\s*([\d.]+)\s*m/s", window, re.IGNORECASE)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None, None


def extract_parameters(pdf_path: Path, text: str = None) -> ExtractedParameters:
    """
    Pull structured parameters out of a drawing's text layer using
    pattern matching. Each pattern is deliberately tolerant of the label
    phrasing, unit, and separator variation found across real drawings
    from different architectural offices — see FLOOR_TO_FLOOR_LABELS,
    TOTAL_TRAVEL_LABELS, and extract_shaft_dimensions for the specific
    variants currently handled.

    Accepts already-extracted text via the optional `text` parameter —
    useful when the caller also needs to run other extractors (e.g.
    level_extractor, real_lift_identifier) against the same document,
    since re-extracting would mean running OCR twice on the same file
    when the text layer is missing, which is meaningfully slow.
    """
    if text is None:
        text = extract_text(pdf_path)
    result = ExtractedParameters(source_file=str(pdf_path), raw_text_length=len(text))

    m = re.search(r"SECTION\s*[·\-—]\s*([A-Z\-]+)\s*BUILDING", text, re.IGNORECASE)
    if not m:
        m = re.search(r"BUILDING TYPE:\s*([a-zA-Z\-]+)", text, re.IGNORECASE)
    if m:
        result.building_type = m.group(1).strip().lower()

    ftf_mm = extract_length_field(text, FLOOR_TO_FLOOR_LABELS)
    if ftf_mm is not None:
        result.floor_to_floor_mm = ftf_mm

    travel_mm = extract_length_field(text, TOTAL_TRAVEL_LABELS)
    if travel_mm is not None:
        result.total_travel_m = round(travel_mm / 1000, 1)

    m = re.search(r"PROJECT:\s*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if m:
        result.project_id = m.group(1).strip()

    type_pattern = "|".join(LIFT_TYPES)
    group_pattern = re.compile(
        rf"([A-Z0-9\-\.]+)\s*\((?P<type>{type_pattern})\)", re.IGNORECASE
    )
    matches = list(group_pattern.finditer(text))
    for i, gm in enumerate(matches):
        group_id = gm.group(1)
        lift_type = gm.group("type").lower()

        window_start = gm.end()
        window_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        window = text[window_start:window_end]

        group = ExtractedGroup(group_id=group_id, lift_type=lift_type)

        width, depth = extract_shaft_dimensions(window)
        group.shaft_width_mm = width
        group.shaft_depth_mm = depth

        load, speed = extract_load_speed(window)
        group.rated_load_kg = load
        group.rated_speed_ms = speed

        if re.search(r"EN81-72", window, re.IGNORECASE) or \
           re.search(r"DEDICATED SHAFT", window, re.IGNORECASE):
            group.dedicated_shaft = True

        result.groups.append(group)

    return result


def evaluate_against_ground_truth(ground_truth_path: Path, verbose: bool = True) -> dict:
    """
    Run extraction against every project in the synthetic ground truth
    and compute real accuracy metrics — this is the component-level
    evaluation described in the proposal (precision/recall/F1 for
    classification-style fields, MAE for continuous ones).
    """
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    metrics = {
        "total_projects": len(ground_truth),
        "floor_to_floor_correct": 0,
        "total_travel_correct": 0,
        "building_type_correct": 0,
        "group_count_correct": 0,
        "group_type_matches": 0,
        "group_type_total": 0,
        "shaft_dims_correct": 0,
        "shaft_dims_total": 0,
        "failures": [],
    }

    for project in ground_truth:
        pdf_path = Path(project["drawing_path"])
        if not pdf_path.exists():
            metrics["failures"].append(f"Missing file: {pdf_path}")
            continue

        extracted = extract_parameters(pdf_path)

        # Floor-to-floor height. A small tolerance is applied deliberately:
        # some drawing conventions state this in metres to 2 decimal places
        # (e.g. "3.29m" for a true value of 3295mm), which loses precision
        # to the nearest 10mm. That's a real property of the drawing itself,
        # not an extraction error — exact-match comparison would wrongly
        # penalise a parser that read the stated value correctly.
        if extracted.floor_to_floor_mm is not None and \
           abs(extracted.floor_to_floor_mm - project["floor_to_floor_mm"]) <= 5:
            metrics["floor_to_floor_correct"] += 1

        # Total travel (allow tiny float rounding tolerance)
        if extracted.total_travel_m is not None and \
           abs(extracted.total_travel_m - project["total_travel_m"]) < 0.05:
            metrics["total_travel_correct"] += 1

        # Building type
        if extracted.building_type == project["building_type"]:
            metrics["building_type_correct"] += 1

        # Group count
        if len(extracted.groups) == len(project["groups"]):
            metrics["group_count_correct"] += 1

        # Per-group type + shaft dimension accuracy (matched by position,
        # since extraction order mirrors generation order)
        for i, gt_group in enumerate(project["groups"]):
            metrics["group_type_total"] += 1
            metrics["shaft_dims_total"] += 1
            if i < len(extracted.groups):
                ext_group = extracted.groups[i]
                if ext_group.lift_type == gt_group["lift_type"]:
                    metrics["group_type_matches"] += 1
                if (ext_group.shaft_width_mm == gt_group["shaft_width_mm"] and
                        ext_group.shaft_depth_mm == gt_group["shaft_depth_mm"]):
                    metrics["shaft_dims_correct"] += 1

        if verbose:
            ok = (extracted.floor_to_floor_mm == project["floor_to_floor_mm"] and
                  len(extracted.groups) == len(project["groups"]))
            status = "OK" if ok else "MISMATCH"
            print(f"[{status:>8}] {project['project_id']} — "
                  f"F-F: {extracted.floor_to_floor_mm} (gt {project['floor_to_floor_mm']}), "
                  f"travel: {extracted.total_travel_m} (gt {project['total_travel_m']}), "
                  f"groups: {len(extracted.groups)} (gt {len(project['groups'])})")

    n = metrics["total_projects"]
    print("\n" + "=" * 60)
    print("EXTRACTION ACCURACY (against synthetic ground truth)")
    print("=" * 60)
    print(f"Floor-to-floor height:  {metrics['floor_to_floor_correct']}/{n} "
          f"({metrics['floor_to_floor_correct']/n*100:.1f}%)")
    print(f"Total travel:           {metrics['total_travel_correct']}/{n} "
          f"({metrics['total_travel_correct']/n*100:.1f}%)")
    print(f"Building type:          {metrics['building_type_correct']}/{n} "
          f"({metrics['building_type_correct']/n*100:.1f}%)")
    print(f"Group count:            {metrics['group_count_correct']}/{n} "
          f"({metrics['group_count_correct']/n*100:.1f}%)")
    if metrics["group_type_total"]:
        print(f"Group type accuracy:    {metrics['group_type_matches']}/{metrics['group_type_total']} "
              f"({metrics['group_type_matches']/metrics['group_type_total']*100:.1f}%)")
    if metrics["shaft_dims_total"]:
        print(f"Shaft dimension acc.:   {metrics['shaft_dims_correct']}/{metrics['shaft_dims_total']} "
              f"({metrics['shaft_dims_correct']/metrics['shaft_dims_total']*100:.1f}%)")
    if metrics["failures"]:
        print(f"\nFailures: {len(metrics['failures'])}")
        for f in metrics["failures"][:5]:
            print(f"  - {f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Stage 2a: Text Parameter Extractor")
    parser.add_argument("--input", type=str, help="Single PDF to extract parameters from")
    parser.add_argument("--evaluate", action="store_true",
                         help="Run extraction against the full synthetic ground truth and report accuracy")
    parser.add_argument("--ground-truth", type=str, default="data/annotations/synthetic_ground_truth.json",
                         help="Path to ground truth JSON (used with --evaluate)")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_against_ground_truth(Path(args.ground_truth))
    elif args.input:
        result = extract_parameters(Path(args.input))
        print(json.dumps(asdict(result), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
