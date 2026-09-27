"""
Layout-Aware Extractor (v2)
===========================

Extracts per-lift and building-level parameters from realistic lift-detail
sheets (plan + car sections + core section + title block), for both
vector PDFs (text layer) and scanned PDFs (no text layer -> OCR).

Pipeline
--------
1. Text acquisition -> a single list of positioned TextLines:
     * vector PDF: PyMuPDF text dict, keeping each line's orientation
       (dimension strings on vertical dimension lines are rotated 90 deg);
     * scanned PDF: Tesseract (LSTM engine) word boxes, two passes - the
       page as-is for horizontal text, and rotated 90 deg for vertical
       dimension text - mapped back to page coordinates.
2. Lift detection: lift-label patterns built ONLY from the project keyword
   lexicon (docs/keyword_lexicon.md, written before the test set existed)
   plus the drafting conventions seen in the DEV split. Labels split over
   two lines are re-joined by merging vertically adjacent, centred lines.
3. Per-lift dimensions by spatial association around each label anchor:
   car width (horizontal number directly above the label), car depth
   (vertical number to the left), door width (horizontal number below),
   shaft width (next horizontal number above the car width), shaft depth
   (vertical number to the right).
4. Levels from signed elevation values (e.g. +3.900M, F.F.L. +3.90,
   FFL +3900); pit and overhead from the vertical dimension chain below
   the lowest / above the highest level.
5. Optional lift schedule table -> rated load / speed per lift.

Everything here is deterministic and rule-based by design; the learned
components of the system are the OCR engine (Tesseract LSTM) and the
optional YOLO detector (src/detector/), evaluated separately.
"""

import json
import re
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pymupdf as fitz

# ---------------------------------------------------------------------------
# Vocabulary (frozen: keyword lexicon + dev-split conventions)
# ---------------------------------------------------------------------------
TYPE_TERMS = {
    "passenger": ["PASSENGER"],
    "service": ["SERVICE", "SERV", "SVC", "BED", "STRETCHER"],
    "firefighter": ["FIREFIGHTING", "FIREFIGHTER", "FIREMAN", "FIRE"],
    "goods": ["GOODS", "FREIGHT"],
}
CODE_PREFIX = {  # short codes - require a digit suffix (lexicon collision rule)
    "PL": "passenger", "EL": "passenger",
    "SL": "service", "SE": "service", "SV": "service", "SVC": "service",
    "FFL": "firefighter", "FF": "firefighter",
    "GL": "goods",
}
_TYPE_ALT = "|".join(sorted({t for ts in TYPE_TERMS.values() for t in ts}, key=len, reverse=True))
_NOUN = r"(?:LIFT|ELEVATOR|ELEV)"
RE_TYPED = re.compile(rf"\b({_TYPE_ALT})\s+{_NOUN}\b\s*(?:NO\.?\s*)?([A-Z]?\s?\d{{1,2}})?\b")
RE_NUMBERED = re.compile(rf"\b{_NOUN}\s*NO\.?\s*(\d{{1,2}})\s*\(\s*({_TYPE_ALT})\s*\)")
_CODE_ALT = "|".join(sorted(CODE_PREFIX, key=len, reverse=True))
RE_CODE = re.compile(rf"(?<![A-Z0-9.])({_CODE_ALT})\s?-?\s?(\d{{1,2}})\b(?![.\d])")

RE_SIGNED = re.compile(r"(?<![\d.])([+\-\u00b1])\s?(\d{1,3}\.\d{1,3}|\d{1,6})(?![\d.])")
RE_INT = re.compile(r"^\D{0,2}(\d{3,5})\D{0,2}$")


@dataclass
class TextLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    vertical: bool = False

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2


@dataclass
class LiftPrediction:
    label: str
    lift_type: str
    x: float
    y: float
    car_w: int = None
    car_d: int = None
    door_w: int = None
    shaft_w: int = None
    shaft_d: int = None
    rated_load_kg: int = None
    rated_speed_ms: float = None


@dataclass
class LayoutResult:
    source: str
    text_source: str = "vector"
    lifts: list = field(default_factory=list)
    levels_m: list = field(default_factory=list)
    num_stops: int = None
    total_travel_m: float = None
    typical_floor_to_floor_mm: int = None
    pit_depth_mm: int = None
    overhead_mm: int = None
    building_type: str = None
    ffl_required: bool = None
    ffl_present: bool = None
    is_lift_drawing: bool = False
    notes: list = field(default_factory=list)
    # Lift-like labels seen on the sheet but outside the vocabulary. Reported
    # for engineer review only - never counted as lifts, never typed, and not
    # used by the evaluation harness (so they cannot change any score).
    unrecognised_labels: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Text acquisition
# ---------------------------------------------------------------------------
def vector_lines(page) -> list:
    out = []
    d = page.get_text("dict")
    for b in d["blocks"]:
        for ln in b.get("lines", []):
            txt = "".join(s["text"] for s in ln["spans"]).strip()
            if not txt:
                continue
            dx, dy = ln["dir"]
            x0, y0, x1, y1 = ln["bbox"]
            out.append(TextLine(txt, x0, y0, x1, y1, vertical=abs(dy) > abs(dx)))
    return out


def _ocr_lines(img, scale, rotated, page_h_px):
    import pytesseract
    data = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
    groups = {}
    for i, w in enumerate(data["text"]):
        if not w.strip() or float(data["conf"][i]) < 30:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append(i)
    out = []
    for idx in groups.values():
        # split a tesseract line where there is a large horizontal gap
        idx.sort(key=lambda i: data["left"][i])
        cur = [idx[0]]
        chunks = []
        for i in idx[1:]:
            p = cur[-1]
            gap = data["left"][i] - (data["left"][p] + data["width"][p])
            if gap > 2.5 * max(data["height"][p], data["height"][i]):
                chunks.append(cur); cur = [i]
            else:
                cur.append(i)
        chunks.append(cur)
        for ch in chunks:
            txt = " ".join(data["text"][i] for i in ch)
            l = min(data["left"][i] for i in ch); t = min(data["top"][i] for i in ch)
            r = max(data["left"][i] + data["width"][i] for i in ch)
            bt = max(data["top"][i] + data["height"][i] for i in ch)
            if rotated:
                # image was rotated 90deg clockwise: new(u,v) <- old(x=v, y=H-1-u)
                x0, x1 = t, bt
                y0, y1 = page_h_px - r, page_h_px - l
                out.append(TextLine(txt, x0 / scale, y0 / scale, x1 / scale, y1 / scale, vertical=True))
            else:
                out.append(TextLine(txt, l / scale, t / scale, r / scale, bt / scale, vertical=False))
    return out


def ocr_lines(page, dpi=200) -> list:
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    scale = dpi / 72
    lines = _ocr_lines(img, scale, False, pix.height)
    lines += _ocr_lines(img.rotate(-90, expand=True), scale, True, pix.height)
    return lines


def get_lines(pdf_path: Path, cache_dir: Path = None):
    """Return (lines, source). Caches OCR output, which is the slow part."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    lines = vector_lines(page)
    if sum(len(l.text) for l in lines) >= 40:
        return lines, "vector"
    if cache_dir:
        cp = Path(cache_dir) / (Path(pdf_path).stem + ".ocr.json")
        if cp.exists():
            return [TextLine(**d) for d in json.loads(cp.read_text())], "ocr"
    lines = ocr_lines(page)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps([asdict(l) for l in lines]))
    return lines, "ocr"


# ---------------------------------------------------------------------------
# 2. Lift label detection
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = s.upper().replace("\u2014", "-").replace("\u2013", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _type_of(term: str) -> str:
    term = term.upper()
    for t, words in TYPE_TERMS.items():
        if term in words:
            return t
    return CODE_PREFIX.get(term)


def merge_label_blocks(lines: list) -> list:
    """Join horizontal lines stacked vertically and centred on each other
    (two-line labels such as 'LIFT No.1' / '(PASSENGER)')."""
    horiz = sorted([l for l in lines if not l.vertical], key=lambda l: l.y0)
    used = set()
    blocks = []
    for i, a in enumerate(horiz):
        if i in used:
            continue
        group = [a]
        used.add(i)
        last = a
        for j in range(i + 1, len(horiz)):
            if j in used:
                continue
            b = horiz[j]
            h = max(last.y1 - last.y0, 4)
            if b.y0 - last.y1 > 1.2 * h:
                if b.y0 - last.y1 > 4 * h:
                    break
                continue
            if abs(b.cx - last.cx) < 0.35 * max(last.x1 - last.x0, b.x1 - b.x0) + 6:
                group.append(b); used.add(j); last = b
        blocks.append(TextLine(" ".join(g.text for g in group),
                               min(g.x0 for g in group), min(g.y0 for g in group),
                               max(g.x1 for g in group), max(g.y1 for g in group)))
    return blocks


def find_labels(lines: list) -> list:
    found = []
    for blk in merge_label_blocks(lines):
        t = _norm(blk.text)
        hits = []
        for m in RE_NUMBERED.finditer(t):
            hits.append((m.start(), f"LIFT NO.{int(m.group(1))} ({m.group(2)})", _type_of(m.group(2))))
        for m in RE_TYPED.finditer(t):
            if any(abs(m.start() - h[0]) < 12 for h in hits):
                continue
            num = (m.group(2) or "").replace(" ", "")
            noun = re.search(_NOUN, m.group(0)).group(0)
            hits.append((m.start(), f"{m.group(1)} {noun} {num}".strip(), _type_of(m.group(1))))
        for m in RE_CODE.finditer(t):
            if any(h[0] <= m.start() <= h[0] + 30 for h in hits):
                continue
            hits.append((m.start(), f"{m.group(1)}-{int(m.group(2)):02d}", _type_of(m.group(1))))
        for _, label, ltype in hits:
            found.append(LiftPrediction(label=label, lift_type=ltype, x=blk.cx, y=blk.cy))
    return found


def canonical_label(label: str) -> str:
    """Normalisation shared by extractor and evaluator."""
    s = _norm(label)
    s = re.sub(r"NO\.\s*", "NO.", s)
    s = re.sub(r"\(\s*", "(", s); s = re.sub(r"\s*\)", ")", s)
    m = re.fullmatch(r"([A-Z]+)\s?-?\s?0*(\d+)", s)
    if m and m.group(1) in CODE_PREFIX:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return s


# ---------------------------------------------------------------------------
# 3. Spatial association of dimensions
# ---------------------------------------------------------------------------
def _num(line: TextLine, lo=300, hi=6000):
    m = RE_INT.match(line.text.replace(",", "").replace(" ", ""))
    if not m:
        return None
    v = int(m.group(1))
    return v if lo <= v <= hi else None


def associate_dimensions(lift: LiftPrediction, lines: list, all_labels: list):
    nums_h = [(l, _num(l)) for l in lines if not l.vertical and _num(l)]
    nums_v = [(l, _num(l)) for l in lines if l.vertical and _num(l)]
    # spacing to neighbouring labels bounds the search window
    others = [o for o in all_labels if o is not lift and abs(o.y - lift.y) < 60]
    half = min([abs(o.x - lift.x) / 2 for o in others] + [120])

    above = sorted([(lift.y - l.cy, l, v) for l, v in nums_h
                    if l.cy < lift.y - 2 and abs(l.cx - lift.x) < half and lift.y - l.cy < 160],
                   key=lambda t: t[0])
    if above:
        lift.car_w = above[0][2]
        car_w_line = above[0][1]
        higher = sorted([(car_w_line.cy - l.cy, l, v) for l, v in nums_h
                         if l.cy < car_w_line.y0 - 4 and abs(l.cx - lift.x) < half and car_w_line.cy - l.cy < 220],
                        key=lambda t: t[0])
        if higher:
            lift.shaft_w = higher[0][2]
    below = sorted([(l.cy - lift.y, l, v) for l, v in nums_h
                    if l.cy > lift.y + 2 and abs(l.cx - lift.x) < half and l.cy - lift.y < 200],
                   key=lambda t: t[0])
    if below:
        lift.door_w = below[0][2]
    left = sorted([(lift.x - l.cx, l, v) for l, v in nums_v
                   if l.cx < lift.x and lift.x - l.cx < half and abs(l.cy - lift.y) < 60],
                  key=lambda t: t[0])
    if left:
        lift.car_d = left[0][2]
        lift._car_d_line = left[0][1]


def assign_shaft_depths(lifts: list, lines: list):
    """Shaft depth = nearest vertical dimension to the RIGHT of the label that
    is not already another car's depth, before the next lift's label.
    Each vertical dimension can serve only one role (dev-split finding: a
    fixed half-way window fails when a car is offset in its shaft)."""
    claimed = {id(getattr(l, "_car_d_line", None)) for l in lifts}
    nums_v = [(l, _num(l)) for l in lines if l.vertical and _num(l) and id(l) not in claimed]
    for lift in lifts:
        nxt = [o.x for o in lifts if o.x > lift.x + 5 and abs(o.y - lift.y) < 80]
        limit = min(nxt) if nxt else lift.x + 200
        cand = sorted([(l.cx - lift.x, v) for l, v in nums_v
                       if lift.x < l.cx < limit and abs(l.cy - lift.y) < 90], key=lambda t: t[0])
        if cand:
            lift.shaft_d = cand[0][1]


# ---------------------------------------------------------------------------
# 4. Levels, pit, overhead
# ---------------------------------------------------------------------------
def parse_levels(lines: list):
    readings = []
    for l in lines:
        if l.vertical:
            continue
        t = _norm(l.text).replace(",", ".")
        for m in RE_SIGNED.finditer(t):
            sign, val = m.group(1), m.group(2)
            if "." in val:
                v = float(val)
            elif val.strip("0") == "":
                v = 0.0
            elif len(val) >= 4 or int(val) >= 100:
                v = int(val) / 1000
            else:
                continue  # bare small signed integer - ambiguous, skip
            v = -v if sign == "-" else v
            if -30 <= v <= 400:
                readings.append((round(v, 3), l))
    # keep values that sit in a vertical column (the section's level marks)
    if len(readings) < 2:
        return readings
    xs = [l.x0 for _, l in readings]
    med = statistics.median(xs)
    col = [(v, l) for v, l in readings if abs(l.x0 - med) < 80]
    uniq = {}
    for v, l in col:
        uniq.setdefault(round(v, 2), (v, l))
    return sorted(uniq.values(), key=lambda t: t[0])


def pit_and_overhead(levels, lines):
    if len(levels) < 2:
        return None, None
    bottom_y = max(l.cy for _, l in levels)
    top_y = min(l.cy for _, l in levels)
    level_x = min(l.x0 for _, l in levels)
    vnums = [(l, _num(l, 800, 7000)) for l in lines if l.vertical and _num(l, 800, 7000)]
    vnums = [(l, v) for l, v in vnums if level_x - 260 < l.cx < level_x]
    pit = [(l.cy - bottom_y, v) for l, v in vnums if bottom_y + 5 < l.cy < bottom_y + 250]
    ovh = [(top_y - l.cy, v) for l, v in vnums if top_y - 300 < l.cy < top_y - 5]
    pit_v = min(pit)[1] if pit else None
    ovh_v = min(ovh)[1] if ovh else None
    return pit_v, ovh_v


# ---------------------------------------------------------------------------
# 5. Lift schedule table
# ---------------------------------------------------------------------------
def parse_schedule(lines, lifts):
    by_label = {canonical_label(l.label): l for l in lifts}
    kg = [l for l in lines if not l.vertical and re.search(r"\b\d{3,4}\s?KG\b", _norm(l.text))]
    for k in kg:
        row = [l for l in lines if not l.vertical and abs(l.cy - k.cy) < 4]
        row_text = " ".join(_norm(l.text) for l in sorted(row, key=lambda l: l.x0))
        labs = find_labels([TextLine(row_text, k.x0, k.y0, k.x1, k.y1)])
        if not labs:
            continue
        key = canonical_label(labs[0].label)
        target = by_label.get(key)
        if target is None:
            continue
        m = re.search(r"\b(\d{3,4})\s?KG\b", row_text)
        target.rated_load_kg = int(m.group(1))
        m = re.search(r"\b(\d(?:\.\d{1,2})?)\s?M/S\b", row_text)
        if m:
            target.rated_speed_ms = float(m.group(1))


# ---------------------------------------------------------------------------
# 6. Unrecognised lift-like labels (reported, not counted)
# ---------------------------------------------------------------------------
RE_CODE_LIKE = re.compile(r"^([A-Z]{2,5})\s?-\s?(\d{1,2})$")
RE_KG = re.compile(r"\b\d{3,4}\s?KG\b")


def find_unrecognised_labels(lines: list, recognised: list) -> list:
    """Code-style labels (e.g. 'PAX-1') that the vocabulary does not cover,
    kept only where the sheet's structure says they name a lift:
      * the label sits where a car label sits - a horizontal dimension
        directly above it and a vertical dimension directly to its left; or
      * the label begins a lift-schedule row (a row containing 'nnnn KG').
    Both tests are structural, so code-like tokens elsewhere on the sheet
    (distribution boards, grid references) are not reported."""
    known = {canonical_label(l.label) for l in recognised}
    horiz = [l for l in lines if not l.vertical]
    found = {}

    for ln in horiz:
        t = _norm(ln.text)
        if not RE_CODE_LIKE.match(t) or find_labels([ln]):
            continue
        probe = LiftPrediction(label=t, lift_type=None, x=ln.cx, y=ln.cy)
        associate_dimensions(probe, lines, recognised + [probe])
        if probe.car_w and probe.car_d:
            found.setdefault(canonical_label(t), t)

    for k in horiz:
        if not RE_KG.search(_norm(k.text)):
            continue
        row = sorted([l for l in horiz if abs(l.cy - k.cy) < 4], key=lambda l: l.x0)
        first = _norm(row[0].text)
        if RE_KG.search(first) or find_labels([row[0]]):
            continue
        if RE_CODE_LIKE.match(first) or re.search(r"\b(LIFT|ELEVATOR|ELEV|CAR)\b", first):
            found.setdefault(canonical_label(first), first)

    return sorted(v for k, v in found.items() if k not in known)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
FFL_THRESHOLD_M = 23.0


def extract_layout(pdf_path: Path, cache_dir: Path = None) -> LayoutResult:
    from src.parser.real_lift_identifier import infer_building_type
    lines, source = get_lines(pdf_path, cache_dir)
    res = LayoutResult(source=str(pdf_path), text_source=source)
    # Schedule-table rows repeat labels; keep the plan instance (the one
    # with the most dimension numbers around it) by de-duplicating on label.
    labels = find_labels(lines)
    for lab in labels:
        associate_dimensions(lab, lines, labels)
    assign_shaft_depths(labels, lines)
    best = {}
    for lab in labels:
        key = canonical_label(lab.label)
        score = sum(v is not None for v in (lab.car_w, lab.car_d, lab.door_w, lab.shaft_w, lab.shaft_d))
        if key not in best or score > best[key][0]:
            best[key] = (score, lab)
    res.lifts = sorted([b[1] for b in best.values()], key=lambda l: l.x)
    parse_schedule(lines, res.lifts)

    levels = parse_levels(lines)
    if len(levels) >= 2:
        vals = [v for v, _ in levels]
        res.levels_m = vals
        res.num_stops = len(vals)
        res.total_travel_m = round(vals[-1] - vals[0], 3)
        diffs = [round((b - a) * 1000 / 10) * 10 for a, b in zip(vals, vals[1:])]
        # most frequent storey height; ties broken towards the upper floors,
        # which are the typical storeys (ground/podium/basement differ)
        counts = {d: diffs.count(d) for d in diffs}
        top = max(counts.values())
        # "typical" needs at least two storeys of the same height; otherwise
        # abstain rather than call an arbitrary storey typical (dev finding)
        if top >= 2:
            res.typical_floor_to_floor_mm = int(next(d for d in reversed(diffs) if counts[d] == top))
        res.pit_depth_mm, res.overhead_mm = pit_and_overhead(levels, lines)
    full_text = " ".join(l.text for l in lines)
    norm_text = _norm(full_text)
    # mixed-use is checked first: 'RETAIL AND RESIDENTIAL' must not
    # resolve to residential (dev-split finding)
    if re.search(r"MIXED[\s-]?USE|RETAIL AND RESIDENTIAL", norm_text):
        res.building_type = "mixed-use"
    else:
        res.building_type = infer_building_type(norm_text)
    res.is_lift_drawing = len(res.lifts) > 0
    res.unrecognised_labels = find_unrecognised_labels(lines, res.lifts)
    if res.total_travel_m is not None:
        res.ffl_required = res.total_travel_m > FFL_THRESHOLD_M
    res.ffl_present = any(l.lift_type == "firefighter" for l in res.lifts)
    return res


if __name__ == "__main__":
    import sys
    r = extract_layout(Path(sys.argv[1]), Path("/tmp/ocr_cache"))
    print(json.dumps(asdict(r), indent=1, default=str))
