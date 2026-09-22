"""
Synthetic Dataset Generator — bootstraps a mock dataset of architectural
drawing PDFs and specification documents so the rest of the pipeline
(parser, classifier, traffic engine, contradiction detection) can be
built and tested before real project drawings are available.

This is NOT a replacement for real data — it's a stepping stone. Every
synthetic drawing comes with machine-generated ground truth, which lets
you validate that each pipeline stage extracts what it *should* before
ever testing against messier, ambiguous real drawings. Document this
choice explicitly in your dissertation as a deliberate methodological
decision (train/validate on synthetic first, then evaluate transfer to
real drawings once obtained).

Usage:
    python generate_dataset.py --count 30 --distractors 20 --output-dir data/synthetic
"""

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Realistic parameter ranges, drawn from the project's domain knowledge
# ---------------------------------------------------------------------------
BUILDING_TYPES = ["residential", "commercial", "hotel", "hospital", "mixed-use", "warehouse", "villa"]

FLOOR_TO_FLOOR_MM = {
    "residential": (2900, 3300),
    "commercial": (3600, 4200),
    "hotel": (3000, 3400),
    "hospital": (3600, 4500),
    "mixed-use": (3200, 3800),
    # Tall clearance for racking/forklifts/storage — genuinely different
    # from any office/residential floor height, not a copy-pasted range.
    "warehouse": (6000, 9000),
    # Standard residential floor height — a villa is still a house, not
    # a different construction type, so this matches "residential".
    "villa": (2900, 3300),
}

FLOOR_COUNT_RANGE = {
    # Caps are calculated so that (floor_count + max_basement_count) *
    # max_floor_to_floor_mm never exceeds the KONE MonoSpace 500 travel
    # limit of 75m, even in the worst-case combination of both maximums.
    # Note this depends on BASEMENT_COUNT_RANGE below, not a flat 3 for
    # every type — villa/warehouse use a smaller basement cap (see there).
    "residential": (3, 18),   # (18+3)*3300mm = 69.3m
    "commercial": (3, 14),    # (14+3)*4200mm = 71.4m
    "hotel": (4, 18),         # (18+3)*3400mm = 71.4m
    "hospital": (3, 12),      # (12+3)*4500mm = 67.5m
    "mixed-use": (4, 16),     # (16+3)*3800mm = 72.2m
    # Warehouses are realistically 1-3 storeys (large floor plates, not
    # tall stacks of floors) — capped low for realism, not just to fit
    # the travel envelope, though it comfortably does: (3+1)*9000=36m.
    "warehouse": (1, 3),
    # Villas are realistically G+1 to G+3 — capped low for realism, not
    # just to fit the travel envelope: (4+1)*3300mm=16.5m, well within cap.
    "villa": (2, 4),
}

# Basement count range per building type. Originally a flat random(0,3)
# for every type — villas and warehouses realistically don't have
# multi-level basements as often as towers do (a villa with 3 basement
# levels would be unusual; most have none or one), so these two types
# get a restricted range rather than inheriting the tower-scale default.
BASEMENT_COUNT_RANGE = {
    "residential": (0, 3), "commercial": (0, 3), "hotel": (0, 3),
    "hospital": (0, 3), "mixed-use": (0, 3),
    "warehouse": (0, 1),
    "villa": (0, 1),
}

# ---------------------------------------------------------------------------
# Real manufacturer duty-range data, from the publicly published KONE
# MonoSpace 500 Planning Guide (traction MRL elevator, the product class
# these synthetic buildings are modelled on). Using these bounds — rather
# than arbitrary "plausible-looking" numbers — grounds the synthetic
# dataset's shaft/speed/load parameters in a real, citable source. Note
# this caps what a SINGLE lift group can realistically serve; the
# FLOOR_COUNT_RANGE above is deliberately kept within this envelope so
# total travel per group never exceeds what one MonoSpace 500 unit can
# actually do (a real >75m building would need a different product line
# or multiple lift zones, which is a simplification noted as a limitation).
# Source: KONE MonoSpace 500 Planning Guide.
#   https://pdf.medicalexpo.com/pdf/kone/kone-monospace-500-planning-guide/78068-88049.html
KONE_MAX_TRAVEL_M = 75
KONE_MAX_STOPS = 24
KONE_LOAD_RANGE_KG = (320, 1150)
KONE_SPEED_RANGE_MS = (1.00, 1.75)
KONE_MAX_PIT_MM = 1550

# Type-specific load/speed sub-ranges within the overall KONE envelope,
# reflecting typical real-world assignment by lift purpose.
LOAD_RANGE_BY_TYPE = {
    "passenger": (630, 1150),     # 8-15 person cars, the common mid/high-rise range
    "service": (630, 1150),       # similar load range, but see speed below
    "firefighter": (630, 1000),   # EN81-72 minimum is typically 630kg (8-person)
    "goods": (1000, 1150),        # towards the top of the MonoSpace 500 duty range
}
SPEED_RANGE_BY_TYPE = {
    "passenger": (1.00, 1.75),
    "service": (1.00, 1.60),      # service lifts are rarely specified at the top speed
    "firefighter": (1.00, 1.75),
    "goods": (1.00, 1.00),        # goods lifts are typically specified at base speed
}


def pick_load_speed(lift_type: str, travel_m: float) -> tuple:
    """
    Pick a rated load and speed within the KONE-derived envelope for this
    lift type. Speed is weighted toward the higher end of the type's range
    as travel increases, reflecting real specification practice (taller
    buildings are specified faster lifts to keep round-trip time reasonable).
    """
    load = random.choice(range(LOAD_RANGE_BY_TYPE[lift_type][0],
                                LOAD_RANGE_BY_TYPE[lift_type][1] + 1, 10))
    speed_lo, speed_hi = SPEED_RANGE_BY_TYPE[lift_type]
    if speed_hi == speed_lo:
        speed = speed_lo
    else:
        # Bias toward speed_hi as travel approaches the KONE max travel
        bias = min(travel_m / KONE_MAX_TRAVEL_M, 1.0)
        speed = round(speed_lo + bias * (speed_hi - speed_lo) * random.uniform(0.5, 1.0), 2)
    return load, speed

# Realistic-ish messy filename fragments, mirroring real project conventions
DWG_PREFIXES = ["360", "421", "158", "902", "277"]
DISCIPLINE_CODES = ["AR", "ARC"]
TITLE_FRAGMENTS = ["LIFT DETAILS", "LIFT DETALS", "LIFT SECTION", "ELEVATOR LAYOUT", "LIFT GA"]

# Typo injection rate for OCR/fuzzy-matching stress testing
TYPO_RATE = 0.15

# Non-elevator drawing types, used to generate "distractor" files that should
# be correctly IGNORED by the triage stage — this is what a real 16GB project
# folder mostly consists of (structural, MEP, landscape), and a triage system
# that flags everything as relevant isn't actually doing useful filtering.
DISTRACTOR_TYPES = {
    "structural": {
        "discipline_code": "ST",
        "titles": ["FOUNDATION PLAN", "REBAR SCHEDULE", "COLUMN LAYOUT",
                   "STRUCTURAL FRAMING PLAN", "BEAM SCHEDULE"],
        "body_lines": ["REINFORCEMENT DETAIL", "FOOTING SCHEDULE", "SLAB THICKNESS: 200mm",
                       "CONCRETE GRADE C40", "COLUMN GRID REFERENCE"],
    },
    "mep_electrical": {
        "discipline_code": "EL",
        "titles": ["ELECTRICAL LAYOUT", "LIGHTING PLAN", "POWER DISTRIBUTION",
                   "SINGLE LINE DIAGRAM", "CABLE ROUTING PLAN"],
        "body_lines": ["DISTRIBUTION BOARD SCHEDULE", "CIRCUIT LAYOUT", "LOAD CALCULATION",
                       "LIGHTING FIXTURE SCHEDULE", "CABLE TRAY ROUTE"],
    },
    "mep_plumbing": {
        "discipline_code": "PL",  # deliberately similar to the "PL" abbreviation risk
        "titles": ["PLUMBING LAYOUT", "DRAINAGE PLAN", "WATER SUPPLY SCHEMATIC",
                   "SANITARY RISER DIAGRAM", "FIRE FIGHTING PIPING PLAN"],
        "body_lines": ["PIPE SIZING SCHEDULE", "DRAINAGE INVERT LEVELS", "PUMP ROOM LAYOUT",
                       "WATER TANK CAPACITY", "SPRINKLER HEAD LAYOUT"],
    },
    "landscape": {
        "discipline_code": "LA",
        "titles": ["LANDSCAPE MASTER PLAN", "PLANTING SCHEDULE", "IRRIGATION LAYOUT",
                   "HARDSCAPE DETAIL", "SITE BOUNDARY PLAN"],
        "body_lines": ["TREE SPECIES SCHEDULE", "IRRIGATION ZONE LAYOUT", "PAVING PATTERN DETAIL",
                       "SOFT LANDSCAPE AREA", "SITE LEVELS"],
    },
}


@dataclass
class LiftGroup:
    group_id: str
    lift_type: str  # passenger | service | firefighter | goods
    num_cars: int
    floors_served_low: int   # negative = basement level
    floors_served_high: int
    shaft_width_mm: int
    shaft_depth_mm: int
    dedicated_shaft: bool
    rated_load_kg: int = None
    rated_speed_ms: float = None


@dataclass
class ProjectSpec:
    project_id: str
    building_type: str
    floor_count: int
    basement_count: int
    floor_to_floor_mm: int
    total_travel_m: float
    groups: list = field(default_factory=list)
    ffl_required: bool = False        # per Dubai Building Code, travel > 23m
    ffl_present: bool = False         # whether a firefighter group actually exists
    drawing_filename: str = ""
    spec_contradiction: dict = None   # set if the spec doc deliberately disagrees


def maybe_typo(text: str) -> str:
    """Randomly corrupt a string to simulate real-world drawing typos."""
    if random.random() > TYPO_RATE or len(text) < 4:
        return text
    chars = list(text)
    i = random.randint(1, len(chars) - 2)
    if random.random() < 0.5:
        del chars[i]
    else:
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def build_project(index: int, force_short: bool = False) -> ProjectSpec:
    building_type = random.choice(BUILDING_TYPES)

    if force_short:
        # Deliberately generate a building that stays under the 23m FFL
        # threshold, so the "FFL not required" branch gets real test coverage
        # rather than relying on chance draws from a wide realistic range.
        #
        # The short-range bounds are clamped to each building type's own
        # realistic FLOOR_COUNT_RANGE, not a flat (2,5) for every type —
        # an earlier version used a flat range regardless of building
        # type, which let a "short" WAREHOUSE (realistic range 1-3
        # floors) come out with 5 floors, silently violating the type's
        # own documented realism bounds. Found via testing a generated
        # batch, not assumed correct from the code alone.
        type_min, type_max = FLOOR_COUNT_RANGE[building_type]
        short_low = max(2, type_min)
        short_high = min(5, type_max)
        if short_high < short_low:
            short_high = short_low  # type's own range is narrower than the short-case window
        floor_count = random.randint(short_low, short_high)
        basement_count = 0
    else:
        floor_count = random.randint(*FLOOR_COUNT_RANGE[building_type])
        basement_count = random.randint(*BASEMENT_COUNT_RANGE[building_type])

    floor_to_floor = random.randint(*FLOOR_TO_FLOOR_MM[building_type])
    total_travel_m = round((floor_count + basement_count) * floor_to_floor / 1000, 1)

    ffl_required = total_travel_m > 23.0

    groups = []

    # Villas are realistically served by at most one small lift — never a
    # multi-car passenger group, and never a separate service group (the
    # building is too small to justify one). This is a genuinely different
    # configuration, not the standard multi-car logic below with smaller
    # numbers substituted in.
    if building_type == "villa":
        num_cars = 1
        passenger_load, passenger_speed = pick_load_speed("passenger", total_travel_m)
        groups.append(LiftGroup(
            group_id="L01",
            lift_type="passenger",
            num_cars=1,
            floors_served_low=-basement_count,
            floors_served_high=floor_count,
            shaft_width_mm=random.choice([1100, 1200, 1350]),  # small "home lift" scale shaft
            shaft_depth_mm=random.choice([1400, 1500]),
            dedicated_shaft=False,
            rated_load_kg=min(passenger_load, 450),  # home-lift scale, well under KONE's tower-scale envelope
            rated_speed_ms=min(passenger_speed, 1.0),  # villas don't need high-speed lifts
        ))

    else:
        num_cars = random.randint(2, 6)
        passenger_load, passenger_speed = pick_load_speed("passenger", total_travel_m)
        groups.append(LiftGroup(
            group_id="-".join(f"EL{n:02d}" for n in range(1, min(num_cars, 2) + 1)) +
                      ("..." if num_cars > 2 else ""),
            lift_type="passenger",
            num_cars=num_cars,
            floors_served_low=-basement_count,
            floors_served_high=floor_count,
            shaft_width_mm=random.choice([2000, 2100, 2200]),
            shaft_depth_mm=random.choice([2300, 2400, 2500]),
            dedicated_shaft=False,
            rated_load_kg=passenger_load,
            rated_speed_ms=passenger_speed,
        ))

        if building_type == "warehouse":
            # Goods movement is the dominant real requirement in a
            # warehouse — a goods lift here, not the default service lift
            # every other building type gets, and a much higher chance of
            # being present (warehouses without any freight lift at all
            # are the less common case, unlike a typical office).
            if random.random() < 0.8:
                goods_load, goods_speed = pick_load_speed("goods", total_travel_m)
                groups.append(LiftGroup(
                    group_id="GL01",
                    lift_type="goods",
                    num_cars=1,
                    floors_served_low=-basement_count,
                    floors_served_high=floor_count,
                    shaft_width_mm=random.choice([2800, 3000, 3200]),  # goods lifts need a larger shaft
                    shaft_depth_mm=random.choice([3200, 3500]),
                    dedicated_shaft=False,
                    rated_load_kg=goods_load,
                    rated_speed_ms=goods_speed,
                ))
        elif random.random() < 0.6:
            service_load, service_speed = pick_load_speed("service", total_travel_m)
            groups.append(LiftGroup(
                group_id=f"SL{random.randint(1,2):02d}",
                lift_type="service",
                num_cars=random.choice([1, 2]),
                floors_served_low=-basement_count,
                floors_served_high=floor_count,
                shaft_width_mm=random.choice([2400, 2600]),
                shaft_depth_mm=random.choice([2600, 2800]),
                dedicated_shaft=False,
                rated_load_kg=service_load,
                rated_speed_ms=service_speed,
            ))

    ffl_present = False
    if ffl_required and random.random() < 0.75:
        ffl_load, ffl_speed = pick_load_speed("firefighter", total_travel_m)
        groups.append(LiftGroup(
            group_id="FFL01",
            lift_type="firefighter",
            num_cars=1,
            floors_served_low=-basement_count,
            floors_served_high=floor_count,
            shaft_width_mm=2100,
            shaft_depth_mm=2400,
            dedicated_shaft=True,
            rated_load_kg=ffl_load,
            rated_speed_ms=ffl_speed,
        ))
        ffl_present = True

    project_id = f"SYN-{index:03d}"

    prefix = random.choice(DWG_PREFIXES)
    discipline = random.choice(DISCIPLINE_CODES)
    title = maybe_typo(random.choice(TITLE_FRAGMENTS))
    filename = f"{prefix}-{1000+index}-DWG-{discipline}-40{index%9}-{title}-00.pdf"

    project = ProjectSpec(
        project_id=project_id,
        building_type=building_type,
        floor_count=floor_count,
        basement_count=basement_count,
        floor_to_floor_mm=floor_to_floor,
        total_travel_m=total_travel_m,
        groups=[asdict(g) for g in groups],
        ffl_required=ffl_required,
        ffl_present=ffl_present,
        drawing_filename=filename,
    )

    if random.random() < 0.3:
        field_choice = random.choice(["total_travel_m", "floor_count", "shaft_width_mm"])
        if field_choice == "total_travel_m":
            spec_value = round(total_travel_m + random.uniform(1.5, 4.0), 1)
        elif field_choice == "floor_count":
            spec_value = floor_count + random.choice([-2, -1, 1, 2])
        else:
            spec_value = groups[0].shaft_width_mm + random.choice([-200, 200])
        project.spec_contradiction = {"field": field_choice, "spec_value": spec_value}

    return project


def build_distractor(index: int) -> dict:
    """Build a non-elevator drawing that the triage stage should correctly skip."""
    category = random.choice(list(DISTRACTOR_TYPES.keys()))
    spec = DISTRACTOR_TYPES[category]
    prefix = random.choice(DWG_PREFIXES)
    title = maybe_typo(random.choice(spec["titles"]))
    filename = f"{prefix}-{2000+index}-DWG-{spec['discipline_code']}-{300+index}-{title}-00.pdf"
    return {
        "category": category,
        "filename": filename,
        "title": title,
        "body_lines": random.sample(spec["body_lines"], k=min(3, len(spec["body_lines"]))),
    }


# ---------------------------------------------------------------------------
# Format variation — real drawings from different architectural offices
# phrase the same information differently. Testing against only ONE
# consistent format (as the first version of this generator did) validates
# the extraction *logic* but not its real-world robustness. Each project
# is assigned one "office style" per field, applied consistently within
# that drawing (mimicking a single office's convention), so the extractor
# must handle genuine format diversity across the dataset.
# ---------------------------------------------------------------------------
FLOOR_TO_FLOOR_FORMATS = [
    lambda mm: f"F-F HT: {mm}mm",
    lambda mm: f"FLOOR TO FLOOR: {mm/1000:.2f}m",
    lambda mm: f"F/F HEIGHT = {mm}",
    lambda mm: f"TYPICAL FLOOR HEIGHT: {mm} MM",
]

TOTAL_TRAVEL_FORMATS = [
    lambda m: f"TOTAL TRAVEL: {m}m",
    lambda m: f"TRAVEL HEIGHT: {int(round(m*1000))}mm",
    lambda m: f"TOTAL RISE: {m} M",
    lambda m: f"OVERALL TRAVEL (O.A.T.): {m}m",
]

SHAFT_DIM_FORMATS = [
    lambda w, d: f"{w}x{d}mm",
    lambda w, d: f"{w} X {d} MM",
    lambda w, d: f"W:{w} D:{d}",
    lambda w, d: f"{w/1000:.1f}m x {d/1000:.1f}m",
]


def render_drawing_pdf(project: ProjectSpec, output_path: Path, format_variation: bool = False):
    """
    Render a simplified architectural SECTION view — floor lines, a shaft
    per lift group, dimension callouts, and a title block.
    """
    doc = fitz.open()
    page = doc.new_page(width=842, height=1191)  # A3 portrait, points

    margin_x = 80
    top_y = 100
    bottom_y = 1000
    total_levels = project.basement_count + project.floor_count
    level_height_px = (bottom_y - top_y) / max(total_levels, 1)

    page.insert_text((margin_x, 60), f"SECTION — {project.building_type.upper()} BUILDING",
                      fontsize=14, fontname="helv")

    for i in range(total_levels + 1):
        y = top_y + i * level_height_px
        page.draw_line((margin_x, y), (margin_x + 500, y), width=0.5, color=(0.5, 0.5, 0.5))
        level_num = project.floor_count - i
        if level_num >= 0:
            label = "RF" if i == 0 else ("GF" if level_num == 0 else f"L{level_num:02d}")
        else:
            label = f"B{-level_num}"
        page.insert_text((margin_x - 40, y + 3), label, fontsize=7)

    page.insert_text((margin_x + 510, top_y + level_height_px / 2),
                      FLOOR_TO_FLOOR_FORMATS[random.randrange(len(FLOOR_TO_FLOOR_FORMATS)) if format_variation else 0](project.floor_to_floor_mm),
                      fontsize=8)
    page.insert_text((margin_x + 510, top_y + level_height_px * 1.5),
                      TOTAL_TRAVEL_FORMATS[random.randrange(len(TOTAL_TRAVEL_FORMATS)) if format_variation else 0](project.total_travel_m),
                      fontsize=8)

    shaft_x = margin_x + 60
    dim_format_idx = random.randrange(len(SHAFT_DIM_FORMATS)) if format_variation else 0
    for g in project.groups:
        shaft_w_pts = g["shaft_width_mm"] / 40
        rect = fitz.Rect(shaft_x, top_y, shaft_x + shaft_w_pts, bottom_y)
        page.draw_rect(rect, color=(0, 0, 0), width=1)
        label = f"{g['group_id']} ({g['lift_type'].upper()})"
        page.insert_text((shaft_x, top_y - 10), label, fontsize=7)
        page.insert_text((shaft_x, bottom_y + 15),
                          SHAFT_DIM_FORMATS[dim_format_idx](g['shaft_width_mm'], g['shaft_depth_mm']),
                          fontsize=6)
        if g.get("rated_load_kg") and g.get("rated_speed_ms"):
            page.insert_text((shaft_x, bottom_y + 37),
                              f"{g['rated_load_kg']}kg / {g['rated_speed_ms']}m/s", fontsize=6)
        if g["dedicated_shaft"]:
            page.insert_text((shaft_x, bottom_y + 26), "EN81-72 DEDICATED SHAFT", fontsize=6)
        shaft_x += shaft_w_pts + 20

    page.draw_rect(fitz.Rect(margin_x, 1050, 762, 1140), color=(0, 0, 0), width=0.8)
    page.insert_text((margin_x + 10, 1075), project.drawing_filename.replace(".pdf", ""), fontsize=8)
    page.insert_text((margin_x + 10, 1095), f"PROJECT: {project.project_id}", fontsize=8)
    page.insert_text((margin_x + 10, 1115), f"BUILDING TYPE: {project.building_type}", fontsize=8)

    doc.save(output_path)
    doc.close()


def render_distractor_pdf(distractor: dict, output_path: Path):
    """Render a plausible non-elevator drawing sheet."""
    doc = fitz.open()
    page = doc.new_page(width=842, height=1191)
    page.insert_text((80, 60), distractor["title"], fontsize=14, fontname="helv")

    y = 120
    for line in distractor["body_lines"]:
        page.insert_text((80, y), line, fontsize=10)
        y += 30

    for gx in range(80, 700, 80):
        page.draw_line((gx, 200), (gx, 900), width=0.3, color=(0.7, 0.7, 0.7))
    for gy in range(200, 900, 80):
        page.draw_line((80, gy), (700, gy), width=0.3, color=(0.7, 0.7, 0.7))

    page.draw_rect(fitz.Rect(80, 1050, 762, 1140), color=(0, 0, 0), width=0.8)
    page.insert_text((90, 1075), distractor["filename"].replace(".pdf", ""), fontsize=8)
    page.insert_text((90, 1095), f"DISCIPLINE: {distractor['category']}", fontsize=8)

    doc.save(output_path)
    doc.close()


def render_spec_pdf(project: ProjectSpec, output_path: Path):
    """Render a simple client specification document, sometimes contradicting the drawing."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    y = 60
    page.insert_text((60, y), f"ELEVATOR SPECIFICATION — {project.project_id}", fontsize=14)
    y += 40

    travel = project.total_travel_m
    floors = project.floor_count
    shaft_w = project.groups[0]["shaft_width_mm"]

    if project.spec_contradiction:
        f = project.spec_contradiction["field"]
        value = project.spec_contradiction["spec_value"]
        if f == "total_travel_m":
            travel = value
        elif f == "floor_count":
            floors = value
        elif f == "shaft_width_mm":
            shaft_w = value

    lines = [
        f"Building type: {project.building_type}",
        f"Number of floors: {floors}",
        f"Total travel: {travel}m",
        f"Primary shaft width: {shaft_w}mm",
        "",
        "Client requirements:",
        "- Destination Control System (DCS) required for passenger group",
        "- Earthquake operation (EN81-77) required",
    ]
    if project.ffl_required:
        lines.append("- Firefighter lift required per Dubai Building Code (travel > 23m)")

    for line in lines:
        page.insert_text((60, y), line, fontsize=10)
        y += 20

    doc.save(output_path)
    doc.close()


def generate_dataset(count: int, output_dir: Path, distractor_count: int = 0, format_variation: bool = False):
    drawings_dir = output_dir / "drawings"
    specs_dir = output_dir / "specs"
    drawings_dir.mkdir(parents=True, exist_ok=True)
    specs_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = []
    force_short_indices = set(random.sample(range(1, count + 1), max(1, count // 4)))

    for i in range(1, count + 1):
        project = build_project(i, force_short=(i in force_short_indices))

        drawing_path = drawings_dir / project.drawing_filename
        render_drawing_pdf(project, drawing_path, format_variation=format_variation)

        spec_path = specs_dir / f"{project.project_id}_SPEC.pdf"
        render_spec_pdf(project, spec_path)

        record = asdict(project)
        record["drawing_path"] = str(drawing_path)
        record["spec_path"] = str(spec_path)
        ground_truth.append(record)

        contradiction_note = ""
        if project.spec_contradiction:
            contradiction_note = f"  [contradiction: {project.spec_contradiction['field']}]"
        missing_ffl_note = "  [MISSING REQUIRED FFL]" if project.ffl_required and not project.ffl_present else ""
        print(f"[{i}/{count}] {project.project_id} — {project.building_type}, "
              f"{project.floor_count} floors, {len(project.groups)} groups"
              f"{contradiction_note}{missing_ffl_note}")

    gt_path = output_dir.parent / "annotations" / "synthetic_ground_truth.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {count} elevator-relevant projects.")

    distractor_records = []
    if distractor_count > 0:
        print(f"\nGenerating {distractor_count} distractor (non-elevator) drawings...")
        for i in range(1, distractor_count + 1):
            distractor = build_distractor(i)
            distractor_path = drawings_dir / distractor["filename"]
            render_distractor_pdf(distractor, distractor_path)
            distractor["drawing_path"] = str(distractor_path)
            distractor_records.append(distractor)
            print(f"[distractor {i}/{distractor_count}] {distractor['category']} — {distractor['filename']}")

        distractor_gt_path = output_dir.parent / "annotations" / "synthetic_distractors.json"
        with open(distractor_gt_path, "w") as f:
            json.dump(distractor_records, f, indent=2)
        print(f"Distractor list: {distractor_gt_path}")

    print(f"\nDrawings: {drawings_dir}  ({count} relevant + {distractor_count} distractor = "
          f"{count + distractor_count} total)")
    print(f"Specs:    {specs_dir}")
    print(f"Ground truth: {gt_path}")


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Synthetic Dataset Generator")
    parser.add_argument("--count", type=int, default=20, help="Number of synthetic projects to generate")
    parser.add_argument("--distractors", type=int, default=0,
                         help="Number of non-elevator distractor drawings to generate")
    parser.add_argument("--format-variation", action="store_true",
                         help="Vary how floor height/travel/dimensions are phrased per project, "
                              "simulating different architectural offices' conventions")
    parser.add_argument("--output-dir", type=str, default="data/synthetic",
                         help="Base output directory (drawings/ and specs/ subfolders created inside)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    generate_dataset(args.count, Path(args.output_dir), distractor_count=args.distractors,
                      format_variation=args.format_variation)


if __name__ == "__main__":
    main()
