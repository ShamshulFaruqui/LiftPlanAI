"""
Extracts per-lift-group specification details (capacity, speed, stops,
travel, power, etc.) from a "brief specification" table embedded directly
in a real architectural/VT drawing.

Why this module exists, and why it's OCR-based rather than text-layer
based like the rest of the real-drawing extractors: confirmed directly
against a real VT (vertical transportation) consultant drawing during
this project — the drawing's normal notes, title block, and headings
(including the literal words "BRIEF SPECIFICATION") ARE present in the
PDF's embedded text layer, but the specification table's own data VALUES
are not, in either PyMuPDF or pdfplumber output. This is the same
"text flattened to vector curves" phenomenon already documented in
text_extractor.py's OCR fallback, but here it affects only ONE table on
an otherwise text-readable page, so the existing near-empty-page OCR
trigger never fires. This module instead: (1) uses the normal text layer
just to detect WHICH page contains a specification table at all (cheap,
and that heading text IS present), then (2) OCRs only that one page to
recover the table contents that text extraction genuinely cannot reach.

Table format handled: a "BRIEF SPECIFICATION" block with one row per
field (LIFT DESIGNATION, USAGE, CAPACITY, SPEED, STOPS / NO OF OPENINGS,
TRAVEL, OPERATION, POWER PER LIFT, etc.) and one column per lift-group
range (e.g. "PL-01 ~ PL-04", "PL-05 ~ PL-07", "MU01"). Cells that are
merged across multiple lift-group columns in the source drawing collapse
to fewer values than groups when read via OCR line-by-line; this is
handled by matching the number of parsed values against the number of
declared groups and broadcasting a single shared value across all
groups when only one value is found for a multi-group row — the common
case (all passenger cars sharing one capacity/speed) is exactly this
shape. This heuristic is NOT a general table-structure parser and has
only been verified against the one real drawing format described above.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re


# Canonical field name -> the label text(s) that identify it in the table,
# matched case-insensitively against the start of an OCR'd line.
FIELD_LABELS = {
    "lift_designation": ["LIFT DESIGNATION"],
    "usage": ["USAGE"],
    "elevator_type": ["TYPE OF ELEVATOR"],
    "capacity": ["CAPACITY"],
    "speed": ["SPEED"],
    "stops": ["STOPS / NO OF OPENINGS", "STOPS/NO OF OPENINGS", "STOPS / NO. OF OPENINGS"],
    "serving_levels": ["SERVING LEVELS"],
    "non_stop_floor": ["NON STOPS FLOOR", "NON STOP FLOOR"],
    "emergency_floor": ["EMERGENCY FLOOR"],
    "travel": ["TRAVEL"],
    "operation": ["OPERATION"],
    "counterweight_safety_gear": ["COUNTER WEIGHT SAFETY GEAR", "COUNTERWEIGHT SAFETY GEAR"],
    "power_per_lift": ["POWER PER LIFT"],
    "heat_dissipation": ["HEAT DISSIPATION PER LIFT", "HEAT DISSIPATION"],
}

# Fields whose value block should NOT be split by "~" the way a lift-ID
# range is (SERVING LEVELS legitimately contains "P1~P4" etc. as part of
# its own value, not as a group-range separator).
_LEVEL_RANGE_FIELD = "serving_levels"

HEADING_KEYWORDS = ["BRIEF SPECIFICATION"]


@dataclass
class LiftGroupSpec:
    lift_ids: str  # e.g. "PL-01 ~ PL-07" (already merged if groups share every field)
    fields: dict = field(default_factory=dict)  # canonical field name -> value string


def find_specification_page(pdf_path: Path) -> int:
    """
    Scan the PDF's normal (fast) text layer for a specification-table
    heading. Returns the 0-indexed page number, or -1 if none is found.
    Deliberately does NOT use OCR here — this check must stay cheap since
    it runs against every page of files that may not contain a spec table
    at all (e.g. a plain floor plan).
    """
    import fitz

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return -1

    for page_num in range(doc.page_count):
        text = doc[page_num].get_text().upper()
        if any(kw in text for kw in HEADING_KEYWORDS):
            doc.close()
            return page_num
    doc.close()
    return -1


def _ocr_page(pdf_path: Path, page_num: int, dpi: int = 200) -> str:
    """OCR the whole page — used only as a fallback when a targeted crop
    isn't available."""
    from pdf2image import convert_from_path
    import pytesseract

    pages = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_num + 1, last_page=page_num + 1)
    return pytesseract.image_to_string(pages[0])


def _ocr_table_crop(pdf_path: Path, page_num: int, heading_keywords, dpi: int = 350) -> str:
    """
    Locate the specification table's heading via the (fast, reliable)
    text layer, then render and OCR only a generous crop around it at
    high effective resolution. Full-page OCR of an A1-size drawing was
    tried first and confirmed too low-resolution for this specific table's
    text to be read reliably — the table occupies a small fraction of a
    huge sheet, so its text renders tiny even at a normally-generous page
    DPI. Cropping first and OCRing at high DPI solves this directly,
    rather than OCRing the whole page at an DPI high enough to compensate
    (which would be far slower and produce a huge, mostly-irrelevant image).
    """
    import fitz

    doc = fitz.open(pdf_path)
    page = doc[page_num]
    heading_rect = None
    for kw in heading_keywords:
        rects = page.search_for(kw)
        if rects:
            heading_rect = rects[0]
            break
    if heading_rect is None:
        doc.close()
        return ""

    # Generous crop: table extends right and down from its heading. Margins
    # tuned against, and verified visually against, the one real drawing
    # this module has been tested on (a wide multi-column spec table).
    clip = fitz.Rect(
        heading_rect.x0 - 20,
        heading_rect.y0 - 15,
        heading_rect.x0 + 820,
        heading_rect.y0 + 460,
    )
    pix = page.get_pixmap(clip=clip, dpi=dpi)
    doc.close()

    import pytesseract
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _split_group_labels(value_line: str) -> list:
    """Split a LIFT DESIGNATION line like 'PL-01 ~ PL-04 PL-05 ~ PL-07 MU01'
    into its individual group labels. Groups are either an 'X ~ Y' range or
    a single bare ID; both are separated by 2+ spaces in the OCR output."""
    chunks = re.split(r"\s{2,}", value_line.strip())
    return [c.strip() for c in chunks if c.strip()]


def _collect_label_blocks(lines: list, canonical_by_label: dict) -> list:
    """
    Walk the OCR'd lines and group them into (canonical_field, block_lines)
    blocks: every block starts at a line beginning with a known field label
    and continues until the next recognised label line. Handles both
    single-line rows (label and values on the same line) and the
    multi-line rows this real table also produces (label alone on its own
    line, values following on one line per lift-group) uniformly, since
    both just become one block of one-or-more lines either way.
    """
    blocks = []
    current_label = None
    current_lines = []

    def _flush():
        if current_label is not None:
            blocks.append((current_label, current_lines))

    for ln in lines:
        upper = ln.upper()
        matched = None
        for lab in canonical_by_label:
            if upper.startswith(lab):
                matched = lab
                break
        if matched is not None:
            _flush()
            current_label = canonical_by_label[matched]
            remainder = ln[len(matched):].strip(" :")
            current_lines = [remainder] if remainder else []
        elif current_label is not None:
            current_lines.append(ln.strip())
    _flush()
    return blocks


_VALUE_PATTERNS = {
    "capacity": r"\d+\s*Kgs?\s*/\s*\d+\s*PERSONS?",
    "speed": r"\d+(?:\.\d+)?\s*MPS",
    "stops": r"\d+\s*/\s*\d+",
    "power_per_lift": r"\d+\s*KW",
    "heat_dissipation": r"\d+(?:\.\d+)?\s*KW",
    "non_stop_floor": r"\bNA\b",
    "emergency_floor": r"\bNA\b",
}


def _values_from_block(block_lines: list, n_groups: int, is_level_field: bool, canon: str = None) -> list:
    """Turn a label's collected lines into per-group value strings."""
    block_lines = [ln for ln in block_lines if ln.strip()]
    if not block_lines:
        return []

    # Case 1: everything landed on one line. Whitespace-based column
    # splitting only works when OCR preserved a wide gap between cells,
    # which is not reliable for short, similarly-shaped values sitting
    # close together (e.g. "1350 Kgs / 18 PERSONS 1600 Kgs / 21 PERSONS"
    # with only a single space at the real cell boundary). Where a known
    # value shape exists for this field, prefer matching that shape
    # directly over guessing from whitespace.
    if len(block_lines) == 1:
        line = block_lines[0].strip()
        if canon in _VALUE_PATTERNS:
            matches = re.findall(_VALUE_PATTERNS[canon], line, flags=re.IGNORECASE)
            if matches:
                return [m.strip() for m in matches]
        values = re.split(r"\s{2,}", line)
        return [v.strip() for v in values if v.strip()]

    # Case 2 (SERVING LEVELS-shaped): each group's value spans a fixed
    # number of lines (e.g. a "FRONT - ..." line then a "REAR - ..." line).
    # If the line count divides evenly by the group count, treat each
    # equal-sized chunk as one group's (possibly multi-line) value.
    if is_level_field and len(block_lines) % n_groups == 0:
        chunk = len(block_lines) // n_groups
        return [
            " / ".join(block_lines[i * chunk:(i + 1) * chunk])
            for i in range(n_groups)
        ]

    # Case 3: one value per line (the common case for this table's
    # LIFT DESIGNATION row, and any other row OCR'd the same way).
    return [ln.strip() for ln in block_lines]


def extract_lift_group_specs(pdf_path: Path) -> list:
    """
    Returns a list of LiftGroupSpec, one per distinct set of lift IDs that
    share identical field values (already merged — see module docstring).
    Returns an empty list if no specification table was found, or if OCR
    could not recover a LIFT DESIGNATION row to anchor the column count.
    """
    page_num = find_specification_page(pdf_path)
    if page_num == -1:
        return []

    try:
        ocr_text = _ocr_table_crop(pdf_path, page_num, HEADING_KEYWORDS)
        if not ocr_text.strip():
            ocr_text = _ocr_page(pdf_path, page_num)
    except Exception:
        return []

    lines = [ln for ln in ocr_text.split("\n") if ln.strip()]

    canonical_by_label = {}
    for canon, labels in FIELD_LABELS.items():
        for lab in labels:
            canonical_by_label[lab] = canon

    blocks = _collect_label_blocks(lines, canonical_by_label)

    # LIFT DESIGNATION anchors how many lift-groups (columns) the table has.
    group_labels = None
    for canon, block_lines in blocks:
        if canon == "lift_designation":
            group_labels = _values_from_block(block_lines, n_groups=0, is_level_field=False)
            break
    if not group_labels:
        return []

    n_groups = len(group_labels)
    per_group_fields = [dict() for _ in group_labels]

    for canon, block_lines in blocks:
        if canon == "lift_designation":
            continue
        values = _values_from_block(block_lines, n_groups, is_level_field=(canon == _LEVEL_RANGE_FIELD), canon=canon)
        if not values:
            continue

        if len(values) == n_groups:
            for i, v in enumerate(values):
                per_group_fields[i][canon] = v
        elif len(values) == 1:
            # One shared value spanning every group (e.g. TRAVEL, OPERATION).
            for i in range(n_groups):
                per_group_fields[i][canon] = values[0]
        elif 1 < len(values) < n_groups:
            # A value was merged across more than one but not all groups
            # (the common real case: all passenger cars share one value,
            # a separate multi-utility car has its own). Broadcast the
            # first value across the head groups and the last to the tail —
            # correct for the verified 2-value/N-group real case; not
            # guaranteed for 3+ distinct partial groupings.
            for i in range(n_groups - 1):
                per_group_fields[i][canon] = values[0]
            per_group_fields[-1][canon] = values[-1]
        # else: more values than groups — ambiguous OCR split, field left unset for this row.

    groups = [
        LiftGroupSpec(lift_ids=label, fields=per_group_fields[i])
        for i, label in enumerate(group_labels)
    ]

    # Merge adjacent groups whose fields are identical, satisfying "if
    # they're all the same, show PL1~PL7 once" rather than repeating rows.
    merged = []
    for g in groups:
        if merged and merged[-1].fields == g.fields:
            prev = merged[-1]
            prev_start = prev.lift_ids.split("~")[0].strip()
            this_end = g.lift_ids.split("~")[-1].strip()
            merged[-1] = LiftGroupSpec(lift_ids=f"{prev_start} ~ {this_end}", fields=prev.fields)
        else:
            merged.append(g)

    return merged
