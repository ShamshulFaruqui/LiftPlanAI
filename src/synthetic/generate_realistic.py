"""
Realistic Synthetic Drawing Generator (v2)
==========================================

Replaces the v1 generator's simplified single-section sheets with drawings
modelled on the structure of real consultant lift-detail sheets:

  * landscape A1 sheet with border, title block, general notes and a
    revision table (fictitious consultant / owner / project names only);
  * LIFT CORE PLAN: shafts with solid structural walls, car inside each
    shaft, counterweight, door opening, car-label text, car and shaft
    dimension strings, shaft-width dimension chain along the front;
  * CAR SECTIONS (interior elevations) - realistic numeric clutter
    (cab height 2500, door height 2100 ...) that an extractor must NOT
    confuse with plan dimensions;
  * LIFT CORE SECTION: floor slabs with FFL level markers, floor-height
    dimension chain, pit depth and overhead;
  * optional per-lift specification table (load / speed / stops / travel);
  * optional "scanned" variant: page rasterised with noise and a slight
    skew, so it has NO text layer and must go through OCR.

Every drawing has full ground truth (per-lift label, type, car / shaft /
door dimensions, levels, travel, pit, overhead) and YOLO-format bounding
boxes for shafts, cars and level markers, so the same dataset supports both
rule-based extraction evaluation and training a learned detector.

Naming conventions are split into SEEN styles (used in dev and test) and
UNSEEN styles (test only), so the test set measures generalisation to
drafting conventions never used during development, not just memorisation.

Car / door sizes follow the standard load -> car-size pairings of
ISO 8100-30 (formerly ISO 4190-1) for passenger lifts.

Usage:
    python -m src.synthetic.generate_realistic --split dev  --count 500 --distractors 100 --seed 1
    python -m src.synthetic.generate_realistic --split test --count 500 --distractors 100 --seed 2
"""

import argparse
import io
import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageFilter
import numpy as np

PAGE_W, PAGE_H = 2384, 1684          # A1 landscape, points
PT_PER_MM_PAPER = 72 / 25.4

# ---------------------------------------------------------------------------
# Domain tables
# ---------------------------------------------------------------------------
# ISO 8100-30 style load -> (car width, car depth, door width), mm
CAR_BY_LOAD = {
    450: [(1000, 1250, 800)],
    630: [(1100, 1400, 800)],
    800: [(1350, 1400, 800)],
    1000: [(1600, 1400, 900), (1100, 2100, 900)],
    1275: [(2000, 1400, 1100)],
    1600: [(1400, 2400, 1300), (2100, 1600, 1100)],
    2000: [(1500, 2700, 1300)],
    2500: [(1800, 2700, 1500)],
}
LOADS_BY_TYPE = {
    "passenger": [630, 800, 1000, 1275, 1600],
    "service": [1000, 1275, 1600, 2000],
    "firefighter": [1000, 1275, 1600],
    "goods": [2000, 2500],
}
FLOOR_TO_FLOOR_MM = {
    "residential": (3000, 3400), "commercial": (3600, 4200), "hotel": (3100, 3500),
    "hospital": (3900, 4500), "mixed-use": (3300, 3900), "warehouse": (6000, 8000),
    "villa": (3300, 4600),
}
ABOVE_GROUND = {
    "residential": (4, 22), "commercial": (3, 18), "hotel": (4, 20), "hospital": (2, 10),
    "mixed-use": (5, 22), "warehouse": (1, 2), "villa": (1, 2),
}
BASEMENTS = {"residential": (0, 3), "commercial": (0, 3), "hotel": (0, 2), "hospital": (0, 2),
             "mixed-use": (0, 3), "warehouse": (0, 0), "villa": (0, 1)}
FFL_THRESHOLD_M = 23.0

PROJECT_NAMES = {
    "residential": ["PROPOSED RESIDENTIAL BUILDING", "RESIDENTIAL APARTMENTS", "RESIDENTIAL TOWER"],
    "commercial": ["PROPOSED OFFICE BUILDING", "COMMERCIAL OFFICE TOWER", "BUSINESS CENTRE"],
    "hotel": ["PROPOSED HOTEL", "HOTEL APARTMENTS", "BEACH RESORT HOTEL"],
    "hospital": ["PROPOSED HOSPITAL", "MEDICAL CENTRE EXTENSION", "SPECIALIST HOSPITAL"],
    "mixed-use": ["MIXED USE DEVELOPMENT", "RETAIL AND RESIDENTIAL BUILDING", "MIXED-USE TOWER"],
    "warehouse": ["PROPOSED WAREHOUSE", "LOGISTICS WAREHOUSE", "DISTRIBUTION CENTRE"],
    "villa": ["PROPOSED RESIDENTIAL VILLA", "PRIVATE VILLA", "G+1 VILLA"],
}
# Deliberately fictitious - no real consultant, contractor or owner names.
CONSULTANTS = ["NORTHGATE ENGINEERING CONSULTANTS", "BLUE DUNE ARCHITECTS", "MERIDIAN DESIGN STUDIO",
               "SANDSTONE CONSULTING ENGINEERS", "AXIS PLANNING ASSOCIATES", "FALCON BAY ARCHITECTS"]
AREAS = ["AL QUSAIS", "JUMEIRAH", "AL BARSHA", "MIRDIF", "NAD AL SHEBA", "AL FURJAN",
         "MUSSAFAH", "KHALIFA CITY", "AL NAHDA", "DUBAI SOUTH"]
EMIRATES = ["DUBAI", "ABU DHABI", "SHARJAH", "AJMAN"]
GENERAL_NOTES = [
    "DIMENSIONS ARE NOT TO BE SCALED FROM THIS DRAWING.",
    "ALL DIMENSIONS ARE IN MILLIMETRES UNLESS OTHERWISE STATED.",
    "THE CONTRACTOR SHALL VERIFY ALL DIMENSIONS ON SITE BEFORE COMMENCING WORK.",
    "THIS DRAWING IS TO BE READ IN CONJUNCTION WITH STRUCTURAL AND MEP DRAWINGS.",
    "ANY DISCREPANCY SHALL BE REPORTED TO THE CONSULTANT BEFORE EXECUTION.",
    "LIFT SHAFT DIMENSIONS TO BE CONFIRMED BY THE LIFT SPECIALIST.",
    "ALL WORKS TO COMPLY WITH LOCAL AUTHORITY REQUIREMENTS.",
]

# ---------------------------------------------------------------------------
# Drafting-convention styles. SEEN styles appear in dev and test; UNSEEN
# styles appear ONLY in the test split (generalisation check).
# ---------------------------------------------------------------------------
TYPE_WORDS = {  # style -> type -> formatter(index)
    "words": {"passenger": lambda i: f"PASSENGER LIFT {i}", "service": lambda i: f"SERVICE LIFT {i}",
              "firefighter": lambda i: f"FIRE LIFT {i}", "goods": lambda i: f"GOODS LIFT {i}"},
    "codes": {"passenger": lambda i: f"PL-{i:02d}", "service": lambda i: f"SL-{i:02d}",
              "firefighter": lambda i: f"FFL-{i:02d}", "goods": lambda i: f"GL-{i:02d}"},
    "numbered": {"passenger": lambda i: f"LIFT No.{i} (PASSENGER)", "service": lambda i: f"LIFT No.{i} (SERVICE)",
                 "firefighter": lambda i: f"LIFT No.{i} (FIREFIGHTING)", "goods": lambda i: f"LIFT No.{i} (GOODS)"},
    # --- unseen (test-only) ---
    "elevator": {"passenger": lambda i: f"PASSENGER ELEVATOR E{i}", "service": lambda i: f"SERVICE ELEVATOR E{i}",
                 "firefighter": lambda i: f"FIREMAN ELEVATOR E{i}", "goods": lambda i: f"FREIGHT ELEVATOR E{i}"},
    "abbrev": {"passenger": lambda i: f"PAX-{i}", "service": lambda i: f"SVC-{i}",
               "firefighter": lambda i: f"FF-{i}", "goods": lambda i: f"FRT-{i}"},
}
SEEN_LABEL_STYLES = ["words", "codes", "numbered"]
UNSEEN_LABEL_STYLES = ["elevator", "abbrev"]


def lvl_fmt(style, name, ffl_m):
    s = f"{ffl_m:+.3f}".replace("+0.000", "\u00b10.000")
    if style == "named":      # +3.900M FIRST FLOOR FFL  (villa example)
        return [f"{ffl_m:+.3f}M", f"{name} FFL"]
    if style == "ffl_dots":   # F.F.L. +3.90
        return [f"F.F.L. {ffl_m:+.2f}", name]
    if style == "mm":         # LEVEL 01  FFL +3900
        return [f"FFL {int(round(ffl_m*1000)):+d}", name]
    if style == "el":         # EL. +3.900 (FIRST FLOOR)  - unseen
        return [f"EL. {ffl_m:+.3f}", f"({name})"]
    if style == "tos":        # TOS +3.90 M  - unseen (top of slab)
        return [f"T.O.S. {ffl_m:+.2f} M", name]
    raise ValueError(style)


SEEN_LEVEL_STYLES = ["named", "ffl_dots", "mm"]
UNSEEN_LEVEL_STYLES = ["el", "tos"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Lift:
    label: str
    lift_type: str
    index: int
    rated_load_kg: int
    rated_speed_ms: float
    car_w: int
    car_d: int
    door_w: int
    shaft_w: int
    shaft_d: int
    shaft_d_shown: bool = True


@dataclass
class Level:
    name: str
    ffl_m: float


@dataclass
class DrawingGT:
    drawing_id: str
    split: str
    filename: str
    is_lift_drawing: bool
    building_type: str = None
    building_type_stated: bool = None
    project_name: str = None
    levels: list = field(default_factory=list)
    typical_floor_to_floor_mm: int = None
    total_travel_m: float = None
    num_stops: int = None
    pit_depth_mm: int = None
    overhead_mm: int = None
    ffl_required: bool = None
    ffl_present: bool = None
    lifts: list = field(default_factory=list)
    style: dict = field(default_factory=dict)
    boxes: list = field(default_factory=list)   # [class, x0, y0, x1, y1] in PDF points
    spec_table_present: bool = False
    scanned: bool = False


# ---------------------------------------------------------------------------
# Building / lift synthesis
# ---------------------------------------------------------------------------
LEVEL_NAMES_ABOVE = ["GROUND FLOOR", "FIRST FLOOR", "SECOND FLOOR", "THIRD FLOOR"]


def level_name(i, n_above, style_short):
    if i == 0:
        return "GROUND FLOOR" if not style_short else "GF"
    if i == n_above:
        return "ROOF FLOOR" if not style_short else "RF"
    if not style_short and i < len(LEVEL_NAMES_ABOVE):
        return LEVEL_NAMES_ABOVE[i]
    return f"LEVEL {i:02d}" if not style_short else f"L{i:02d}"


def build_levels(btype, rng):
    n_above = rng.randint(*ABOVE_GROUND[btype])
    n_base = rng.randint(*BASEMENTS[btype])
    typ = rng.randrange(*FLOOR_TO_FLOOR_MM[btype], 50)
    gf_mm = rng.choice([0, 150, 300])
    short = rng.random() < 0.4
    levels = []
    # basements
    y = gf_mm
    for b in range(1, n_base + 1):
        y -= rng.choice([3600, 3900, 4050, 4200])
        levels.append(Level(f"BASEMENT {b}" if not short else f"B{b}", y / 1000))
    levels.reverse()
    levels.append(Level(level_name(0, n_above, short), gf_mm / 1000))
    y = gf_mm
    for i in range(1, n_above + 1):
        h = typ
        if i == 1 and btype not in ("villa", "warehouse") and rng.random() < 0.6:
            h = typ + rng.choice([600, 900, 1200, 1500])  # taller ground floor / podium
        y += h
        levels.append(Level(level_name(i, n_above, short), y / 1000))
    return levels, typ


def pick_speed(travel_m, rng):
    if travel_m < 12:
        return rng.choice([0.4, 0.63, 1.0])
    if travel_m < 30:
        return rng.choice([1.0, 1.6])
    if travel_m < 60:
        return rng.choice([1.6, 1.75, 2.0])
    return rng.choice([2.5, 3.0, 3.5])


def make_lift(ltype, idx, travel, rng, villa=False):
    load = 450 if villa else rng.choice(LOADS_BY_TYPE[ltype])
    car_w, car_d, door_w = rng.choice(CAR_BY_LOAD[load])
    shaft_w = car_w + rng.choice([500, 550, 600, 700, 800])
    shaft_d = car_d + rng.choice([400, 450, 500, 550])
    return dict(lift_type=ltype, index=idx, rated_load_kg=load,
                rated_speed_ms=min(pick_speed(travel, rng), 1.0) if villa else pick_speed(travel, rng),
                car_w=car_w, car_d=car_d, door_w=door_w, shaft_w=shaft_w, shaft_d=shaft_d)


def build_lifts(btype, travel, rng):
    if btype == "villa":
        return [make_lift("passenger", 1, travel, rng, villa=True)]
    lifts = []
    n_pass = rng.randint(1, 2) if btype == "warehouse" else rng.randint(1, 5)
    for i in range(n_pass):
        lifts.append(make_lift("passenger", i + 1, travel, rng))
    if btype == "warehouse":
        for _ in range(rng.randint(1, 2)):
            lifts.append(make_lift("goods", 0, travel, rng))
    elif btype == "hospital" or rng.random() < 0.45:
        for _ in range(rng.randint(1, 2)):
            lifts.append(make_lift("service", 0, travel, rng))
    if travel > FFL_THRESHOLD_M and rng.random() < 0.8:
        lifts.append(make_lift("firefighter", 0, travel, rng))
    # Lifts of the same type form one group and share one duty (load, speed,
    # car, door, shaft) - real groups are specified to a common duty.
    duty_keys = ("rated_load_kg", "rated_speed_ms", "car_w", "car_d", "door_w", "shaft_w", "shaft_d")
    first = {}
    for d in lifts:
        ref = first.setdefault(d["lift_type"], d)
        for k in duty_keys:
            d[k] = ref[k]
    rng.shuffle(lifts) if rng.random() < 0.3 else None
    return lifts


def assign_labels(lifts, style, rng):
    # numbering: either global running index or per-type index
    per_type = rng.random() < 0.5 and style != "numbered"
    counters = {}
    out = []
    for k, d in enumerate(lifts, start=1):
        counters[d["lift_type"]] = counters.get(d["lift_type"], 0) + 1
        i = counters[d["lift_type"]] if per_type else k
        label = TYPE_WORDS[style][d["lift_type"]](i)
        out.append(Lift(label=label, **d))
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
BLACK, GREY, LIGHT, RED = (0, 0, 0), (0.45, 0.45, 0.45), (0.8, 0.8, 0.8), (0.75, 0.05, 0.05)


def text(page, x, y, s, size=9, color=BLACK, rotate=0, font="helv"):
    page.insert_text((x, y), s, fontsize=size, color=color, rotate=rotate, fontname=font)


def hdim(page, x0, x1, y, value, size=8, tick=4):
    """Horizontal dimension line with ticks and centred value text."""
    page.draw_line((x0, y), (x1, y), color=BLACK, width=0.4)
    for x in (x0, x1):
        page.draw_line((x - tick, y + tick), (x + tick, y - tick), color=BLACK, width=0.8)
        page.draw_line((x, y - 2 * tick), (x, y + 2 * tick), color=BLACK, width=0.3)
    s = str(value)
    w = fitz.get_text_length(s, fontsize=size)
    text(page, (x0 + x1) / 2 - w / 2, y - 3, s, size)


def vdim(page, x, y0, y1, value, size=8, tick=4):
    """Vertical dimension line; text rotated 90 deg reading bottom-up."""
    page.draw_line((x, y0), (x, y1), color=BLACK, width=0.4)
    for y in (y0, y1):
        page.draw_line((x - tick, y + tick), (x + tick, y - tick), color=BLACK, width=0.8)
        page.draw_line((x - 2 * tick, y), (x + 2 * tick, y), color=BLACK, width=0.3)
    s = str(value)
    w = fitz.get_text_length(s, fontsize=size)
    text(page, x - 3, (y0 + y1) / 2 + w / 2, s, size, rotate=90)


def draw_border_and_titleblock(page, gt, rng, consultant, drawing_title, scale_txt):
    page.draw_rect(fitz.Rect(20, 20, PAGE_W - 20, PAGE_H - 20), color=BLACK, width=1.5)
    tb = fitz.Rect(PAGE_W - 380, 20, PAGE_W - 20, PAGE_H - 20)
    page.draw_rect(tb, color=BLACK, width=1.2)
    x = tb.x0 + 10
    # General notes
    text(page, x + 120, 45, "GENERAL NOTES", 10)
    y = 65
    for n, note in enumerate(rng.sample(GENERAL_NOTES, rng.randint(3, 5)), start=1):
        text(page, x, y, f"{n}. {note}"[:62], 6.5)
        y += 13
    # revision table
    ry = 700
    for r in range(6):
        page.draw_line((tb.x0, ry + r * 22), (tb.x1, ry + r * 22), color=BLACK, width=0.5)
    page.draw_line((tb.x0 + 45, ry), (tb.x0 + 45, ry + 110), color=BLACK, width=0.5)
    page.draw_line((tb.x1 - 80, ry), (tb.x1 - 80, ry + 110), color=BLACK, width=0.5)
    date = f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.2026"
    text(page, x, ry + 104, "No.", 7); text(page, x + 110, ry + 104, "REVISION / ISSUE", 7)
    text(page, tb.x1 - 70, ry + 104, "DATE", 7)
    text(page, x, ry + 82, "R0", 7); text(page, x + 50, ry + 82, rng.choice(
        ["ISSUED FOR APPROVAL", "ISSUED FOR TENDER", "ISSUED FOR CONSTRUCTION"]), 7)
    text(page, tb.x1 - 70, ry + 82, date, 7)
    # consultant
    cy = 850
    page.draw_line((tb.x0, cy), (tb.x1, cy), color=BLACK, width=0.8)
    text(page, x, cy + 18, "CONSULTANT:", 7)
    text(page, x, cy + 40, consultant, 8.5)
    text(page, x, cy + 56, f"P.O. Box {rng.randint(1000, 99999)}, {rng.choice(EMIRATES)}, UAE", 7)
    # owner / project
    py = 950
    page.draw_line((tb.x0, py), (tb.x1, py), color=BLACK, width=0.8)
    text(page, x, py + 18, "PROJECT:", 7)
    text(page, x, py + 40, gt.project_name, 9)
    text(page, x, py + 56, f"PLOT No. {rng.randint(100000, 9999999)}, {rng.choice(AREAS)}", 8)
    if gt.levels:
        n_base = sum(1 for l in gt.levels if l.ffl_m < -0.01)
        desc = (f"{n_base}B+" if n_base else "") + f"G+{len(gt.levels) - n_base - 1}"
        text(page, x, py + 72, f"PROJECT DESCRIPTION: {desc}", 7)
    # drawing info
    dy = 1100
    page.draw_line((tb.x0, dy), (tb.x1, dy), color=BLACK, width=0.8)
    for label, val, yy in (("DRAWN BY", rng.choice(["AJ", "MK", "SR", "HN"]), dy + 20),
                           ("CHECKED BY", rng.choice(["MY", "RT", "FA"]), dy + 60),
                           ("DATE", date, dy + 100), ("SCALE", scale_txt, dy + 140),
                           ("PAPER SIZE", "A1", dy + 180)):
        text(page, x, yy, label, 6.5); text(page, x + 120, yy + 4, val, 9)
    ty = 1330
    page.draw_rect(fitz.Rect(tb.x0 + 5, ty, tb.x1 - 5, ty + 150), color=BLACK, width=2)
    text(page, x, ty + 20, "DRAWING TITLE", 7)
    yy = ty + 50
    for line in drawing_title:
        text(page, x + 10, yy, line, 14); yy += 22
    text(page, x, 1510, "DWG. No.", 7)
    text(page, x + 10, 1545, gt.filename.split("_")[0], 16)
    text(page, tb.x1 - 90, 1510, "REV. No.", 7)
    text(page, tb.x1 - 70, 1545, "0", 14)


def draw_plan(page, gt, lifts, origin, scale, rng, label_color):
    """Lift core plan. Returns boxes."""
    boxes = []
    mm = PT_PER_MM_PAPER / scale
    wall = 200 * mm
    ox, oy = origin
    x = ox + wall
    max_d = max(l.shaft_d for l in lifts)
    front_y = oy + wall + max_d * mm        # front (door) wall inner face
    fill_walls = rng.random() < 0.6
    wall_fill = GREY if fill_walls else None
    total_w = sum(l.shaft_w for l in lifts) * mm + wall * (len(lifts) + 1)
    # outer core outline and walls
    page.draw_rect(fitz.Rect(ox, oy, ox + total_w, front_y + wall), color=BLACK, fill=wall_fill, width=1)
    lobby_depth = rng.choice([1800, 2400, 3000]) * mm
    page.draw_rect(fitz.Rect(ox, front_y + wall, ox + total_w, front_y + wall + lobby_depth),
                   color=BLACK, width=0.6)
    text(page, ox + total_w / 2 - 30, front_y + wall + lobby_depth / 2 + 4, "LIFT LOBBY", 8, GREY)
    dim_chain = []
    for l in lifts:
        sw, sd = l.shaft_w * mm, l.shaft_d * mm
        shaft = fitz.Rect(x, front_y - sd, x + sw, front_y)
        page.draw_rect(shaft, color=BLACK, fill=(1, 1, 1), width=0.8)
        boxes.append(["shaft", *shaft])
        # car: centred-ish, counterweight at side or rear
        cw_side = rng.random() < 0.6 and (l.shaft_w - l.car_w) >= 550
        cw, cd = l.car_w * mm, l.car_d * mm
        if cw_side:
            cx0 = x + 60 * mm
            cwt = fitz.Rect(x + sw - 250 * mm, front_y - sd + 300 * mm, x + sw - 80 * mm, front_y - sd + 1000 * mm)
        else:
            cx0 = x + (sw - cw) / 2
            cwt = fitz.Rect(x + (sw - 800 * mm) / 2, front_y - sd + 60 * mm,
                            x + (sw + 800 * mm) / 2, front_y - sd + 220 * mm)
        cy1 = front_y - 120 * mm
        car = fitz.Rect(cx0, cy1 - cd, cx0 + cw, cy1)
        page.draw_rect(car, color=BLACK, width=1)
        page.draw_rect(car + (3, 3, -3, -3), color=GREY, width=0.4)
        boxes.append(["car", *car])
        page.draw_rect(cwt, color=BLACK, fill=LIGHT, width=0.5)
        # door opening in front wall
        dw = l.door_w * mm
        dx0 = cx0 + (cw - dw) / 2
        page.draw_rect(fitz.Rect(dx0, front_y, dx0 + dw, front_y + wall), color=(1, 1, 1), fill=(1, 1, 1), width=0)
        page.draw_line((dx0, front_y + wall / 2), (dx0 + dw, front_y + wall / 2), color=BLACK, width=1.2)
        page.draw_line((dx0, front_y), (dx0, front_y + wall), color=BLACK, width=0.5)
        page.draw_line((dx0 + dw, front_y), (dx0 + dw, front_y + wall), color=BLACK, width=0.5)
        # car cross (typical CAD car symbol) - sometimes
        if rng.random() < 0.5:
            page.draw_line(car.tl, car.br, color=LIGHT, width=0.4)
            page.draw_line(car.tr, car.bl, color=LIGHT, width=0.4)
        # label - may be two lines
        size = rng.choice([8, 9, 10])
        words = l.label.split(" ")
        lines = [l.label] if fitz.get_text_length(l.label, fontsize=size) < cw - 8 else \
            [" ".join(words[:len(words) // 2 + 1]), " ".join(words[len(words) // 2 + 1:])]
        ly = car.y0 + cd / 2 - 6 * (len(lines) - 1)
        for ln in lines:
            w = fitz.get_text_length(ln, fontsize=size)
            text(page, car.x0 + cw / 2 - w / 2, ly, ln, size, label_color)
            ly += size + 3
        # car dims
        hdim(page, car.x0, car.x1, car.y0 + 14, l.car_w, 7)
        vdim(page, car.x0 + 14, car.y0, car.y1, l.car_d, 7)
        # shaft depth on shaft right side (inside)
        l.shaft_d_shown = rng.random() < 0.7
        if l.shaft_d_shown:
            vdim(page, shaft.x1 - 10, shaft.y0, shaft.y1, l.shaft_d, 7)
        dim_chain.append((x, x + sw, l.shaft_w))
        # door width
        hdim(page, dx0, dx0 + dw, front_y + wall + 26, l.door_w, 7)
        x += sw + wall
    # shaft width chain along the top
    chain_y = oy - 22
    for x0, x1, v in dim_chain:
        hdim(page, x0, x1, chain_y, v, 8)
    hdim(page, ox, ox + total_w, chain_y - 26, int(round((total_w) / mm / 10) * 10), 8)
    return boxes


def draw_car_sections(page, lifts, origin, scale, rng):
    """Interior car elevations - numeric clutter that is NOT plan data."""
    mm = PT_PER_MM_PAPER / scale
    ox, oy = origin
    seen = []
    for l in lifts:
        key = (l.car_w, l.car_d, l.door_w)
        if key in seen:
            continue
        seen.append(key)
        cab_h = rng.choice([2400, 2500, 2600])
        door_h = rng.choice([2100, 2200])
        for view_w, title in ((l.car_d, "SIDE"), (l.car_w, "FRONT")):
            w, h = view_w * mm, cab_h * mm
            r = fitz.Rect(ox, oy, ox + w, oy + h)
            if r.x1 > 1500:
                return
            page.draw_rect(r, color=BLACK, width=1)
            page.draw_line((ox, oy + h * 0.45), (ox + w, oy + h * 0.45), color=GREY, width=1.5)  # handrail
            hdim(page, ox, ox + w, oy - 8, view_w, 7)
            vdim(page, ox + w - 8, oy, oy + h, cab_h, 7)
            if title == "FRONT":
                dw = l.door_w * mm
                page.draw_rect(fitz.Rect(ox + (w - dw) / 2, oy + h - door_h * mm, ox + (w + dw) / 2, oy + h),
                               color=BLACK, width=0.6)
                vdim(page, ox + (w - dw) / 2 + 10, oy + h - door_h * mm, oy + h, door_h, 7)
            ox += w + 40
        ox += 30
    text(page, origin[0], oy + 2500 * mm + 30, "CAR SECTIONS", 10)
    text(page, origin[0], oy + 2500 * mm + 42, f"Scale 1/{scale}", 7)
    if rng.random() < 0.6:
        text(page, origin[0], oy + 2500 * mm + 58, "50mm DIA STAINLESS STEEL HANDRAIL - CAR FINISHES AS PER SPECIFICATION", 7)


def draw_section(page, gt, lifts, origin, max_h_pt, rng, lvl_style):
    boxes = []
    levels = gt.levels
    pit, overhead = gt.pit_depth_mm, gt.overhead_mm
    total_mm = (levels[-1].ffl_m - levels[0].ffl_m) * 1000 + pit + overhead
    scale = next(s for s in (50, 75, 100, 150, 200, 250, 300, 400) if total_mm / s * PT_PER_MM_PAPER <= max_h_pt)
    mm = PT_PER_MM_PAPER / scale
    ox, oy = origin
    shaft_w = max(l.shaft_w for l in lifts) * mm
    top_y = oy + overhead * mm
    base = levels[0].ffl_m

    def y_of(ffl):
        return top_y + (levels[-1].ffl_m - ffl) * 1000 * mm

    pit_y = y_of(base) + pit * mm
    page.draw_rect(fitz.Rect(ox - 8, oy, ox, pit_y + 8), color=BLACK, fill=GREY, width=0.5)
    page.draw_rect(fitz.Rect(ox + shaft_w, oy, ox + shaft_w + 8, pit_y + 8), color=BLACK, fill=GREY, width=0.5)
    page.draw_rect(fitz.Rect(ox - 8, oy - 8, ox + shaft_w + 8, oy), color=BLACK, fill=GREY, width=0.5)
    page.draw_rect(fitz.Rect(ox - 8, pit_y, ox + shaft_w + 8, pit_y + 8), color=BLACK, fill=GREY, width=0.5)
    text(page, ox + shaft_w / 2 - 14, pit_y - 6, "LIFT PIT", 6, RED)
    dim_x = ox - 40
    prev_y = None
    for lv in levels:
        y = y_of(lv.ffl_m)
        page.draw_rect(fitz.Rect(ox - 60, y, ox - 8, y + 6), color=BLACK, fill=LIGHT, width=0.4)
        page.draw_line((ox, y), (ox + shaft_w, y), color=LIGHT, width=0.3, dashes="[4 3] 0")
        # level marker triangle + text to the right of shaft
        mx = ox + shaft_w + 30
        page.draw_polyline([(mx, y), (mx - 5, y - 8), (mx + 5, y - 8), (mx, y)], color=BLACK, width=0.6)
        page.draw_line((mx - 10, y), (mx + 110, y), color=BLACK, width=0.4)
        l1, l2 = lvl_fmt(lvl_style, lv.name, lv.ffl_m)
        text(page, mx + 10, y - 12, l1, 7)
        text(page, mx + 10, y + 9, l2, 6.5)
        boxes.append(["level_marker", mx - 12, y - 24, mx + 120, y + 14])
        if prev_y is not None:
            vdim(page, dim_x, prev_y, y, int(round((prev_lv - lv.ffl_m) * 1000)), 6)
        prev_y, prev_lv = y, lv.ffl_m
    vdim(page, dim_x, y_of(base), pit_y, pit, 6)
    vdim(page, dim_x, oy, y_of(levels[-1].ffl_m), overhead, 6)
    if rng.random() < 0.7:  # overall travel dimension (not always present - like real sheets)
        vdim(page, dim_x - 35, y_of(levels[-1].ffl_m), y_of(base), int(round((levels[-1].ffl_m - base) * 1000)), 6)
    text(page, ox - 20, pit_y + 40, "LIFT CORE SECTION", 10)
    text(page, ox - 20, pit_y + 52, f"Scale 1/{scale}", 7)
    return boxes, scale


def draw_spec_table(page, gt, lifts, origin, rng):
    ox, oy = origin
    cols = ["LIFT", "LOAD", "SPEED", "STOPS", "TRAVEL"]
    cw = [150, 80, 70, 60, 80]
    rows = [[l.label, f"{l.rated_load_kg} KG", f"{l.rated_speed_ms} M/S", str(gt.num_stops),
             f"{gt.total_travel_m:.2f} M"] for l in lifts]
    text(page, ox, oy - 10, "LIFT SCHEDULE", 9)
    y = oy
    for r in [cols] + rows:
        x = ox
        for c, w in zip(r, cw):
            page.draw_rect(fitz.Rect(x, y, x + w, y + 18), color=BLACK, width=0.5)
            text(page, x + 4, y + 13, c[:24], 7)
            x += w
        y += 18


def rasterise(pdf_bytes, rng, dpi=150):
    """'Scanned' variant: no text layer, grey noise, slight skew."""
    src = fitz.open("pdf", pdf_bytes)
    pix = src[0].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    angle = rng.uniform(-0.6, 0.6)
    img = img.rotate(angle, fillcolor=255, resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.int16)
    arr = arr + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 10, arr.shape).astype(np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.4))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=70)
    out = fitz.open()
    page = out.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=buf.getvalue())
    return out.tobytes(garbage=3, deflate=True), angle


# ---------------------------------------------------------------------------
# Drawing assembly
# ---------------------------------------------------------------------------
def build_drawing(idx, split, rng, unseen_rate):
    btype = rng.choice(list(FLOOR_TO_FLOOR_MM))
    levels, typ = build_levels(btype, rng)
    travel = round(levels[-1].ffl_m - levels[0].ffl_m, 3)
    lifts_raw = build_lifts(btype, travel, rng)
    use_unseen = split == "test" and rng.random() < unseen_rate
    label_style = rng.choice(UNSEEN_LABEL_STYLES if use_unseen else SEEN_LABEL_STYLES)
    lvl_style = rng.choice(UNSEEN_LEVEL_STYLES if use_unseen else SEEN_LEVEL_STYLES)
    lifts = assign_labels(lifts_raw, label_style, rng)
    speed = max(l.rated_speed_ms for l in lifts)
    pit = 1200 if speed <= 1.0 else 1500 if speed <= 1.75 else 1800 if speed <= 2.5 else 2400
    pit += rng.choice([0, 50, 100, 150])
    overhead = 3600 if speed <= 1.0 else 4000 if speed <= 1.75 else 4400 if speed <= 2.5 else 5000
    overhead += rng.choice([0, 100, 200, 300])
    did = f"{split.upper()}-{idx:04d}"
    code = rng.choice(["A", "AR", "ARC"])
    fname = f"{code}-{rng.randint(500, 599)}_{rng.choice(['LIFT_DETAILS', 'ELEVATOR_DETAILS', 'LIFT_CORE_PLAN_AND_SECTION', 'LIFT_PLAN_SECTION'])}_{did}.pdf"
    gt = DrawingGT(
        drawing_id=did, split=split, filename=fname, is_lift_drawing=True, building_type=btype,
        project_name=rng.choice(PROJECT_NAMES[btype]) if rng.random() < 0.85 else "PROPOSED DEVELOPMENT",
        levels=levels, typical_floor_to_floor_mm=typ,
        building_type_stated=None, total_travel_m=travel, num_stops=len(levels),
        pit_depth_mm=pit, overhead_mm=overhead, ffl_required=travel > FFL_THRESHOLD_M,
        ffl_present=any(l.lift_type == "firefighter" for l in lifts), lifts=lifts,
        style={"label_style": label_style, "level_style": lvl_style, "unseen": use_unseen},
    )
    gt.building_type_stated = gt.project_name != "PROPOSED DEVELOPMENT"
    return gt


def render(gt, rng, scan_rate):
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    lifts = gt.lifts
    plan_scale = 50 if sum(l.shaft_w for l in lifts) < 14000 else 75
    if sum(l.shaft_w for l in lifts) > 20000:
        plan_scale = 100
    label_color = RED if rng.random() < 0.3 else BLACK
    draw_border_and_titleblock(
        page, gt, rng, rng.choice(CONSULTANTS),
        rng.choice([["ELEVATOR DETAILS"], ["LIFT", "PLAN AND SECTION"], ["LIFT CORE", "PLAN & SECTION"]]),
        f"AS SHOWN" if rng.random() < 0.5 else f"1:{plan_scale}")
    boxes = draw_plan(page, gt, lifts, (140, 200), plan_scale, rng, label_color)
    text(page, 140, 200 + 7000 * PT_PER_MM_PAPER / plan_scale + 120, "LIFT CORE PLAN", 11)
    text(page, 140, 200 + 7000 * PT_PER_MM_PAPER / plan_scale + 133, f"Scale 1/{plan_scale}", 7)
    draw_car_sections(page, lifts, (140, 1080), 25, rng)
    sec_boxes, sec_scale = draw_section(page, gt, lifts, (1640, 90), 1400, rng, gt.style["level_style"])
    boxes += sec_boxes
    if rng.random() < 0.35:
        draw_spec_table(page, gt, lifts, (720, 800), rng)  # clear of plan (above) and car sections (below)
        gt.spec_table_present = True
    elif rng.random() < 0.5:
        text(page, 1050, 120, "FOR LIFT SPECIFICATIONS REFER TO SPECIFICATION SECTION No. 14200", 7)
    gt.style.update({"plan_scale": plan_scale, "section_scale": sec_scale})
    gt.boxes = [[c, round(a, 1), round(b, 1), round(cc, 1), round(d, 1)] for c, a, b, cc, d in boxes]
    data = doc.tobytes(garbage=3, deflate=True)
    if rng.random() < scan_rate:
        data, angle = rasterise(data, rng)
        gt.scanned = True
        gt.style["skew_deg"] = round(angle, 2)
    return data


DISTRACTORS = {
    "structural": ("S", ["FOUNDATION PLAN", "COLUMN LAYOUT", "SLAB REINFORCEMENT"], ["COLUMN C1 600x600", "SLAB 200 THK", "CONCRETE GRADE C40", "REFER TO STRUCTURAL NOTES"]),
    "electrical": ("E", ["LIGHTING LAYOUT", "POWER LAYOUT", "SINGLE LINE DIAGRAM"], ["DB-01", "ELECTRICAL RISER", "LIFT POWER ISOLATOR BY MEP", "CABLE TRAY 300x50"]),
    "plumbing": ("P", ["DRAINAGE LAYOUT", "WATER SUPPLY LAYOUT"], ["PLUMBING SHAFT", "FLOOR DRAIN", "SOIL STACK 100 DIA", "LIFT PIT SUMP BY MEP"]),
    "architectural": ("A", ["TYPICAL FLOOR PLAN", "TOILET DETAILS", "STAIRCASE DETAILS"], ["LIFT LOBBY", "STAIR 1", "CORRIDOR", "F.F.L. +3.900", "APARTMENT TYPE A"]),
}


def build_and_render_distractor(idx, split, rng):
    cat = rng.choice(list(DISTRACTORS))
    code, titles, words = DISTRACTORS[cat]
    title = rng.choice(titles)
    did = f"{split.upper()}-D{idx:04d}"
    fname = f"{code}-{rng.randint(100, 499)}_{title.replace(' ', '_')}_{did}.pdf"
    gt = DrawingGT(drawing_id=did, split=split, filename=fname, is_lift_drawing=False,
                   project_name=rng.choice(sum(PROJECT_NAMES.values(), [])), style={"category": cat})
    doc = fitz.open(); page = doc.new_page(width=PAGE_W, height=PAGE_H)
    draw_border_and_titleblock(page, gt, rng, rng.choice(CONSULTANTS), title.split(" ", 1), "1:100")
    for gx in range(200, 1900, 170):
        page.draw_line((gx, 150), (gx, 1500), color=GREY, width=0.3, dashes="[12 4 2 4] 0")
    for gy in range(200, 1500, 170):
        page.draw_line((150, gy), (1900, gy), color=GREY, width=0.3, dashes="[12 4 2 4] 0")
    for _ in range(rng.randint(8, 20)):
        x, y = rng.uniform(200, 1700), rng.uniform(200, 1400)
        page.draw_rect(fitz.Rect(x, y, x + rng.uniform(60, 300), y + rng.uniform(60, 200)), color=BLACK, width=0.8)
    for _ in range(rng.randint(5, 12)):
        text(page, rng.uniform(200, 1700), rng.uniform(200, 1450), rng.choice(words), 8)
    return gt, doc.tobytes(garbage=3, deflate=True)


def gt_to_json(gt):
    d = asdict(gt)
    return d


def write_yolo(gt, path, classes=("shaft", "car", "level_marker")):
    lines = []
    for c, x0, y0, x1, y1 in gt.boxes:
        cx, cy = (x0 + x1) / 2 / PAGE_W, (y0 + y1) / 2 / PAGE_H
        w, h = (x1 - x0) / PAGE_W, (y1 - y0) / PAGE_H
        lines.append(f"{classes.index(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines))


def generate(split, count, distractors, out_root, seed, scan_rate=0.2, unseen_rate=0.3,
             start=1, stop=None):
    """Each sheet has its own deterministic seed ("<seed>-<index>"), so
    generation is resumable and order-independent: sheets already written
    (per-sheet ground truth present) are skipped, and any sub-range can be
    generated separately. ground_truth.json is assembled once all exist."""
    out = Path(out_root) / split
    for sub in ("drawings", "labels", "gt"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    stop = stop or count + distractors
    for k in range(start, stop + 1):
        is_distractor = k > count
        i = k - count if is_distractor else k
        gt_file = out / "gt" / (f"D{i:04d}.json" if is_distractor else f"{i:04d}.json")
        if gt_file.exists():
            continue
        rng = random.Random(f"{seed}-{'D' if is_distractor else ''}{i}")
        if is_distractor:
            gt, data = build_and_render_distractor(i, split, rng)
        else:
            gt = build_drawing(i, split, rng, unseen_rate)
            data = render(gt, rng, scan_rate)
            write_yolo(gt, out / "labels" / (Path(gt.filename).stem + ".txt"))
        (out / "drawings" / gt.filename).write_bytes(data)
        gt_file.write_text(json.dumps(gt_to_json(gt)))
    files = sorted((out / "gt").glob("*.json"))
    if len(files) == count + distractors:
        records = [json.loads(f.read_text()) for f in sorted(files, key=lambda f: (f.stem.startswith("D"), f.stem))]
        (out / "ground_truth.json").write_text(json.dumps(records, indent=1))
        return records
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--distractors", type=int, default=100)
    ap.add_argument("--out", default="data/synthetic_v2")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scan-rate", type=float, default=0.2)
    ap.add_argument("--unseen-rate", type=float, default=0.3)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--stop", type=int, default=None)
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else (1 if a.split == "dev" else 2)
    recs = generate(a.split, a.count, a.distractors, a.out, seed, a.scan_rate, a.unseen_rate, a.start, a.stop)
    if recs is None:
        print("partial: run again to continue")
    else:
        n_lift = sum(r["is_lift_drawing"] for r in recs)
        print(f"{a.split}: {n_lift} lift drawings + {len(recs) - n_lift} distractors -> {a.out}/{a.split}")


if __name__ == "__main__":
    main()
