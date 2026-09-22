# LiftPlan AI

An AI-driven system for automated elevator specification from architectural
drawings — MSc Data Science & AI Individual Project (CST4275).

```
liftplan-ai/
├── data/
│   ├── raw/            <- put your source PDFs here (not committed to git)
│   ├── synthetic/       <- generated mock drawings + specs (see below)
│   ├── processed/      <- triage reports, extracted parameters, etc.
│   └── annotations/    <- ground truth (real or synthetic) for training
├── src/
│   ├── triage/          Stage 1 — file filtering (WORKING)
│   ├── synthetic/        Mock dataset generator (WORKING)
│   ├── parser/           Stage 2a — text-layer parameter extraction (WORKING)
│   ├── classifier/        Stage 3 — lift group CV classifier (not started)
│   ├── traffic_engine/    Stage 4a — CIBSE Guide D traffic study (not started)
│   ├── spec_engine/       Stage 4b — XGBoost specification recommender (not started)
│   └── dashboard/         Stage 5 — Streamlit dashboard (not started)
├── tests/               unit tests (pytest) — 29 passing
├── docs/                 reference docs (keyword lexicon, etc.)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

## Progress so far

### ✅ Stage 1 — File Triage (`src/triage/file_triage.py`)
Scans a project folder and flags which PDFs are elevator-relevant using
fuzzy keyword matching on filenames and embedded text.

```bash
python src/triage/file_triage.py --input data/synthetic/drawings --output data/processed/report.csv
```

### ✅ Synthetic Dataset Generator (`src/synthetic/generate_dataset.py`)
Since public developer drawings (EMAAR etc.) only show simplified sales
floor plans and real tender-portal drawings require supplier registration,
this generates a mock dataset with full machine-generated ground truth.

```bash
# Basic set
python src/synthetic/generate_dataset.py --count 30 --output-dir data/synthetic

# With distractors (structural/MEP/landscape drawings that should be IGNORED)
python src/synthetic/generate_dataset.py --count 30 --distractors 20 --output-dir data/synthetic

# With format variation (different "office conventions" for the same fields —
# use this to stress-test the parser, not just the triage stage)
python src/synthetic/generate_dataset.py --count 30 --distractors 20 --format-variation --output-dir data/synthetic
```

### ✅ Stage 2a — Text Parameter Extractor (`src/parser/text_extractor.py`)
Extracts floor-to-floor height, total travel, building type, and per-group
lift type/dimensions directly from the PDF's embedded text layer.

```bash
# Single file
python src/parser/text_extractor.py --input data/synthetic/drawings/some_drawing.pdf

# Full accuracy evaluation against ground truth
python src/parser/text_extractor.py --evaluate --ground-truth data/annotations/synthetic_ground_truth.json
```

**Current accuracy (against 30 synthetic drawings with format variation
enabled):** 100% across floor-to-floor height, total travel, building
type, group count, group type classification, and shaft dimensions.

### ✅ Real manufacturer data grounding
Shaft/load/speed ranges are now derived from the publicly published
**KONE MonoSpace 500 Planning Guide** rather than arbitrary estimates:
load 320–1150kg, speed 1.00–1.75 m/s, max travel 75m, max 24 stops.
Floor-count ranges per building type were recalculated so no single
lift group's travel exceeds this real-world limit (a genuine
simplification worth noting in your dissertation — real buildings
taller than ~75m either use a different product line or split into
multiple lift zones, which this synthetic dataset doesn't yet model).
Source: https://pdf.medicalexpo.com/pdf/kone/kone-monospace-500-planning-guide/78068-88049.html

**Note on the 23m firefighter-lift threshold:** this figure comes from
professional knowledge shared during this project's development, not
an independently verified citation — a public search did not surface
a specific Dubai Building Code section confirming the exact number.
The compliance-checking logic itself is unaffected; if you want to
cite this rule formally in your dissertation, verify the exact
section reference against the actual code document.

### ✅ Stage 2b — Visual Parameter Extractor (`src/parser/visual_extractor.py`)
Reads the same information as the text extractor, but from the RENDERED
IMAGE of the page using OpenCV geometry detection — a fallback channel
for drawings where the text layer is missing, unreliable, or scanned.
Detects shaft outlines (count + width), and floor line count.

```bash
# Single file
python src/parser/visual_extractor.py --input data/synthetic/drawings/some_drawing.pdf

# Full accuracy evaluation against ground truth
python src/parser/visual_extractor.py --evaluate --ground-truth data/annotations/synthetic_ground_truth.json
```

**Current accuracy:** 100% shaft count, 100% floor count, 100% shaft
width (within 50mm tolerance) across all 30 synthetic drawings.

**Known limitations, worth stating explicitly in your dissertation:**
- The pixel-to-mm conversion is calibrated to this synthetic generator's
  specific (arbitrary) rendering scale. Real drawings would need proper
  scale-bar detection or a known reference dimension to calibrate this.
- Shaft *depth* is not visually extractable from a section view (only
  width is visible) — this mirrors how real section drawings work too,
  and would need a plan view or the text layer instead.
- This synthetic generator does not preserve real proportional floor-to-
  floor height in the vertical spacing of lines (always evenly distributes
  lines across a fixed vertical span) — so floor-to-floor height in mm
  cannot be recovered visually here, only the floor *count* can.

### ✅ Merged Extractor — Text + Visual Cross-Validation (`src/parser/merged_extractor.py`)
Combines both channels and flags disagreements for engineer review,
exactly as described in the proposal's Methods section. Deliberately
does NOT silently resolve disagreements — it reports both values and a
confidence level ("high" or "review_needed"), preserving human oversight.

```bash
# Single file
python src/parser/merged_extractor.py --input data/synthetic/drawings/some_drawing.pdf

# Full cross-validation summary against ground truth
python src/parser/merged_extractor.py --evaluate --ground-truth data/annotations/synthetic_ground_truth.json
```

**Current result:** 100% group-count agreement, 100% shaft-width
agreement (within a 75mm tolerance), 0/30 falsely flagged for review.
That tolerance matters — even clean synthetic drawings show a natural
~7mm gap between the text channel's exact stated value and the visual
channel's pixel-measured value, purely from measurement rounding. Without
an appropriate tolerance, the merge logic would falsely flag *every*
drawing for review, which would be useless in practice. Tested with both
constructed agreement and constructed genuine-disagreement cases (see
`tests/test_merged_extractor.py`) to confirm the logic actually
distinguishes real discrepancies from expected measurement noise, not
just that it stays quiet on clean data.

### ✅ Stage 3 — Rule-Based Lift Group Classifier (`src/classifier/rule_based_classifier.py`)
Classifies a lift group's type (passenger, service, firefighter, goods)
purely from its group ID prefix (e.g. "FFL01" → firefighter) — a
genuinely different, independent signal from the text extractor's
explicit "(TYPE)" label reading, since real drawings won't always spell
the type out. This is the rule-based half of the "hybrid lift group
classifier" the proposal describes; a supervised model can be layered on
top later for cases these rules can't resolve.

```bash
python src/classifier/rule_based_classifier.py --group-id "FFL01"
python src/classifier/rule_based_classifier.py --evaluate --ground-truth data/annotations/synthetic_ground_truth.json
```

**Current accuracy:** 100% on all 53 groups across the synthetic dataset.

**Deliberately fails safely rather than guessing on ambiguous IDs** —
tested against several realistic naming conventions not produced by this
project's synthetic generator but plausible in real drawings: a bare "P"
prefix (could mean Passenger, Parking, or Plant — deliberately NOT
mapped, mirroring the earlier "PL"/"EL" discipline-code collision lesson
from the triage stage), a bare "L" prefix (doesn't specify type), bare
numeric IDs, spelled-out labels ("LIFT-01"), and an ambiguous "FS01"
("Fire Service"?). All correctly return `None` rather than a wrong guess
— an unrecognised prefix should be flagged for engineer review, not
silently misclassified. See `tests/test_rule_based_classifier.py`.

### ✅ Stage 4a — Traffic Study Engine (`src/traffic_engine/traffic_study.py`)
Implements the conventional up-peak Round Trip Time (RTT) method from
CIBSE Guide D: probable stops, highest reversal floor, flight time
kinematics, RTT, interval, and 5-minute handling capacity.

```bash
python src/traffic_engine/traffic_study.py --floors 15 --population 300 \
    --lifts 6 --speed 1.75 --load 1150 --floor-height 3.2
```

**Validation status — read this before relying on it.** The core formula
(`RTT = 2H·tv + (S+1)·ts + 2P·tp`) is corroborated by an independent
peer-reviewed source citing CIBSE Guide D directly (see the module
docstring for the citation). However, an exact published **numeric**
worked example to check this implementation's output figures against
could not be retrieved in this session — the two sources found (a
Scribd document, a HKU teaching-notes PDF) were both blocked by bot
protection. Rather than claim a validation this project doesn't actually
have, this module was instead checked through:
- Manual hand-calculation of the kinematics sub-formula against known
  physics (see `test_flight_time_trapezoidal_case_matches_manual_calculation`).
- 25 consistency tests confirming every output moves in the correct
  direction as inputs change (more floors → longer RTT; more lifts →
  shorter interval; more population → lower handling capacity; etc.),
  which would catch sign errors, swapped terms, or unit mistakes even
  without one single "known-correct" reference number.
- Running it against real synthetic projects and confirming an
  intentionally well-elevatored test case (many lifts for a small
  building) lands in the commonly cited "good service" interval band
  (15–35s), while an intentionally under-elevatored case (2 lifts for
  13 floors, 400 people) correctly shows a poor interval (82.5s) —
  i.e. the tool correctly identifies bad designs as bad, not just
  producing plausible-looking numbers regardless of input.

**Before using this for a real specification:** cross-check a sample
calculation against the actual CIBSE Guide D document or a validated
commercial tool — this has not been done, and the module says so.

**Documented simplifications:**
- Kinematics use a simple constant-acceleration model, not the
  jerk-limited S-curve profile real lifts actually follow. CIBSE Guide D
  itself notes the "conventional" method makes this same simplification
  and offers corrected formulae for jerk-limited motion — treated here
  as a known limitation and natural next step, not attempted.
- Population is not derived from extracted drawing parameters (floor
  area / unit counts aren't currently extracted) — `estimate_population()`
  is a clearly-labelled rough placeholder (persons-per-floor by building
  type) that should be replaced with real occupancy data before
  production use.
- Running this against the synthetic dataset's randomly-generated
  `num_cars` values often shows poor traffic performance — this is
  expected, not a bug. Those values were randomised to test the
  *parsing* pipeline, not to represent well-designed buildings, so the
  traffic engine correctly reports them as under-elevatored.

### ✅ Stage 4b — Specification Recommender (`src/spec_engine/recommender.py`)
Searches for the cheapest lift configuration (fewest lifts, then lowest
speed, then lowest load) that meets published performance targets for a
building, using Stage 4a's traffic study engine as the evaluation
function for every candidate configuration tried.

```bash
python src/spec_engine/recommender.py --floors 15 --population 300 --floor-height 3.2 --building-type commercial
```

**Why this is rule-based search, not the XGBoost model the proposal
describes.** The proposal's Methods section specifies an XGBoost
recommender "trained on historical project data." No real historical
specification data exists in this project — only the synthetic dataset,
whose `rated_load_kg`/`rated_speed_ms` values were assigned by
`generate_dataset.py`'s `pick_load_speed()` using a loose, somewhat
arbitrary weighting, **not** by any genuine traffic-adequacy criterion.
Training a supervised model on those labels would teach it to reproduce
that arbitrary assignment logic, not real engineering judgement — a
misleading result dressed up as machine learning. This search-based
approach is the honest choice available now, and its outputs are
genuinely meaningful labels a future supervised model could train on,
once enough configurations exist this way or real historical data is
obtained. State this explicitly as a scope decision in your dissertation,
not as a gap you didn't notice.

**Target criteria used, with sourcing:**

| Building type | Min handling capacity | Max interval | Source |
|---|---|---|---|
| Residential | 6% | 60s | ISO 8100-32:2020 Table 2 (maps directly to CIBSE Guide D Table 3.4–3.5) |
| Commercial | 12% | 30s | CIBSE Guide D / British Council for Offices (BCO) 2009 guidelines |
| Hotel | 8% (estimated) | 60s | Interval from ISO 8100-32:2020; handling capacity not directly found this session — estimated |
| Hospital | 10% (estimated) | 45s | Not directly found this session — estimated, flagged for verification |
| Mixed-use | 10% (estimated) | 40s | Blended estimate between residential and commercial |

Only residential and commercial are grounded in directly-found published
figures. The others are honestly labelled as estimates in the code
itself (`TARGET_CRITERIA[...]["source"]`) — verify against the actual
CIBSE Guide D document before relying on them.

**Tested behaviour, not just "it runs":**
- A small residential building correctly gets the cheapest configuration
  (2 lifts, minimum speed/load), with interval as the binding constraint.
- A large office building correctly needs more lifts at higher speed,
  with *both* interval and handling capacity landing close to their
  targets — exactly the "minimal adequate design" behaviour a real
  search should produce.
- An extreme building (22 floors, 2000 population) correctly reports
  infeasible within the search space, with a message noting the building
  likely needs multiple lift zones — a real engineering conclusion, not
  a crash or a silently-wrong answer.
- The recommended configuration is verified to actually satisfy its own
  selection criteria (not just that *some* configuration was returned).

### ✅ Contradiction Detection (`src/contradiction/contradiction_check.py`, `src/parser/spec_reader.py`)
Cross-references client specification documents against drawing-extracted
parameters, flagging genuine discrepancies for engineer review — the
drawing is always treated as authoritative, matching established
industry practice and the proposal's stated design.

```bash
python src/contradiction/contradiction_check.py --drawing data/synthetic/drawings/X.pdf --spec data/synthetic/specs/Y_SPEC.pdf
python src/contradiction/contradiction_check.py --evaluate --ground-truth data/annotations/synthetic_ground_truth.json
```

**Scope — read before relying on this.** Only two fields are compared:
`total_travel_m` and the primary group's `shaft_width_mm`. **Floor count
comparison is deliberately not attempted**, despite being the *most
common* contradiction type in the synthetic test data (3 of 5 cases).
Why: a genuine ambiguity was found, not papered over. The drawing
generator's own visual floor-labelling always relabels the top numbered
floor as "RF" (roof) regardless of what number it would otherwise carry,
so counting floor labels in the drawing text undercounts the true floor
count by one relative to what the spec document means by "number of
floors." Resolving this needs either a consistent floor-counting
convention added to the text extractor or a dedicated field — neither
exists yet. Stating this clearly is more useful to you than a detector
that silently guesses wrong on the majority of its real test cases.

**Result on the 30-project synthetic set:** both in-scope contradictions
(SYN-017's total travel, SYN-022's shaft width) correctly flagged, all
25 genuinely clean projects correctly left unflagged (0 false
positives), and the 3 out-of-scope floor-count contradictions correctly
produce no flag (since that field isn't compared) rather than a
misleading result.

### ✅ Stage 5 — Dashboard (`src/dashboard/app.py`, `src/dashboard/pipeline.py`)
A Streamlit app that runs every working stage — extraction, cross-
validation, classification, traffic study, recommendation, and
contradiction detection — on an uploaded drawing (and optional spec
document), and presents the results per lift group with a compliance
checklist. `pipeline.py` holds all the orchestration logic (unit tested
independently of the UI); `app.py` only renders what it returns.

```bash
streamlit run src/dashboard/app.py
```

Then upload a drawing from `data/synthetic/drawings/` (any file with
"LIFT" in the name). For a demo showing both a compliance flag and a
contradiction flag together, use the drawing containing "421-1017" paired
with `data/synthetic/specs/SYN-017_SPEC.pdf`.

**Verified to actually start**, not just assumed correct from reading the
code: run headless and confirmed an HTTP 200 response with no exceptions
in the server log before considering this done.

**A real gap this surfaced and fixed**: the text extractor had never
learned to read the rated load/speed text ("690kg / 1.24m/s") that
`generate_dataset.py` writes into every drawing — added when the KONE
data was integrated earlier in the project, but never wired into
extraction. Fixed by adding `extract_load_speed()` before building the
dashboard, since the traffic study needs these values per group.

**Another honest gap, found and handled rather than hidden**: group IDs
are sometimes truncated in the drawing label when there are more than
two cars (e.g. "EL01-EL02..." meaning "at least these two, possibly
more" — this project's own synthetic generator does this). The pipeline
does not silently guess the true count. `estimate_car_count()` returns a
lower-bound estimate and flags it as uncertain, and the dashboard
surfaces that flag directly rather than passing a wrong number silently
into the traffic study.

## Testing against REAL drawings (once you have them)

Every `--evaluate` flag elsewhere in this project compares against the
synthetic dataset's known ground truth — which real drawings obviously
don't have. `src/dashboard/real_drawing_test.py` is built for exactly
this moment: drop anonymised real PDFs into `data/real_test/` and run:

```bash
python src/dashboard/real_drawing_test.py --input data/real_test --output data/processed/real_test_report.csv
```

It runs triage first (exactly like a real project folder), then the
full pipeline on anything flagged relevant, and writes a CSV report —
**no ground truth required, no pass/fail judgement made.** The output is
something you manually compare against the real drawing yourself.

**This is not a formality — it's where your dissertation's real
evaluation comes from.** For each drawing, record: what extracted
correctly, what extracted wrong or not at all, and *why* (a label
convention the lexicon doesn't cover, a scanned page with no text layer,
a font rendering issue, whatever it turns out to be). A drawing that
partially fails and tells you exactly why is a more useful dissertation
result than one that silently works — it's real evidence about where the
system's real-world boundaries actually are, which is precisely what an
evaluation chapter needs and what testing only against your own
synthetic data cannot give you.

**Anonymisation before use:** redact client name, project name/code, and
site address from the title block only — the technical content
(dimensions, floor labels, lift group labels, shaft sizes) is what the
pipeline needs and isn't sensitive.

## Findings from first real-drawing test (fixed, not just noted)

Testing against real drawings (not this project's synthetic data) for
the first time immediately surfaced two genuine issues — exactly the
point of doing this testing. Both are now fixed:

**1. A real design flaw: one channel's failure was discarding the
other's successful results.** `run_pipeline()` originally wrapped both
text and visual extraction in a single try/except. When the visual
channel failed (see #2 below), the exception handler discarded
*everything*, including text-channel results that had already succeeded
independently. Fixed by attempting both channels separately — a failure
in one now degrades gracefully to text-only or visual-only results
(clearly flagged as such) rather than losing everything. See
`test_visual_channel_failure_does_not_discard_text_results` in
`tests/test_pipeline.py`.

**2. Some real CAD-exported PDFs break PyMuPDF entirely.** Drawings
exported with layers (OCGs) from AutoCAD/Revit can trigger `MuPDF error:
format error: No default Layer config` — a known bug category in how
MuPDF handles Optional Content Groups, with no documented fix found in
current PyMuPDF. Added a `pdfplumber` fallback (a different underlying
PDF library with no shared code path for this specific failure) in
`extract_text()`. This fixes the *text* channel for such files; the
*visual* channel still has no result for them, since only PyMuPDF is
used there — a real, stated limitation, not something papered over.

**3. Poppler is not installed by `pip install pdf2image` on Windows.**
If you see `"Unable to get page count. Is poppler installed and in
PATH?"`, the visual channel cannot run at all until this is fixed:
1. Download Poppler for Windows: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract it somewhere permanent (e.g. `C:\poppler`)
3. Add the `Library\bin` folder inside it to your system PATH
4. Restart your terminal and re-run

With fix #1 in place, a missing Poppler installation now degrades to
text-only results (with a clear flag) rather than losing everything —
so re-running your batch test now should recover most or all of your 13
files, even before Poppler is installed, if their text layer is intact.

## Real-drawing structural finding: a completely different information format

Testing against a real drawing (not this project's synthetic data)
revealed that real architectural drawings don't state floor count or
total travel directly at all — the synthetic generator's assumed format
(`"TOTAL TRAVEL: 24.1m"`) doesn't exist in practice. Instead, real
drawings list every floor **level** with its own name and elevation,
following common BIM/CAD convention:

```
P3_PODIUM3            12200.000
P2_PODIUM2             8700.000
P1_PODIUM1             5200.000
GR_GROUND FLOOR            0.000
B1_BASEMENT1           -4450.000
B2_BASEMENT2           -8050.000
B3_WATER TANK LEVEL    -9200.000
B3_UNDERGROUND LEVEL  -11450.000
```

Floor count is the number of distinct levels; total travel is the
difference between the highest and lowest elevation. **This is a second,
independent extraction path** (`src/parser/level_extractor.py`) built
specifically for this — it does not replace the original text extractor
(which remains valid for testing against this project's synthetic
dataset), it addresses a real structural format the synthetic generator
never produced.

**Unit assumption, stated explicitly:** raw elevation values are treated
as millimetres. The real drawing this was built against contains a note
reading "ALL DIMENSIONS ARE IN MILLIMETERS AND LEVELS IN METERS" — taken
literally this would mean these values are already in metres, but the
observed numbers only make physical sense as millimetres (a ~3500-5200
difference between adjacent podium floors is a realistic floor height in
mm; the same numbers as metres would imply multi-kilometre floors). This
is standard BIM/Revit elevation-display convention. Flagged here as an
assumption, not asserted as certain — a different real drawing could
prove this wrong.

**Two real bugs found and fixed while building this against the actual
file:**
1. The elevation regex capped the integer part at 3 digits with no
   start-of-number anchor, silently truncating `"12200.000"` down to
   `"200.000"` by matching only the last 3 digits before the decimal.
2. The level-label regex was far too permissive, matching OCR noise
   fragments like `"TO_BE INTERFACE"` and `"FFL_H"` as if they were
   genuine level codes. Restricted to the specific prefixes actually
   observed (P, B, GR, RF, F, each optionally followed by digits).

**A real OCR error caught by a new anomaly-detection feature, not hidden:**
running against the actual file, one level (`P3_PODIUM3`) was OCR-misread
as `72200.000` instead of the true `~12200.000` (a classic 7↔1 digit
confusion) — creating an implausible 46.8m gap to the next level.
`detect_gap_anomalies()` flags any gap more than 3× the typical
(median) gap between levels, surfacing exactly this case for manual
verification rather than silently trusting every OCR'd number. This is
the same design principle used throughout this project (flag ambiguity,
don't guess) — see `tests/test_level_extractor.py` for the regression
test built directly from this real case.

**Honest limitation:** OCR at 300 DPI does not reliably capture every
individual repeated floor label on a dense drawing — testing against
the real file found a plausible ~68.5m gap between two "adjacent"
(by extraction) levels, strongly suggesting several intermediate typical
floors exist on the real drawing but were not picked up by OCR at all.
Floor count from this method should be treated as **a lower-bound
estimate**, not a guaranteed exact count, for buildings with many
repetitive floor labels — flagged by the same anomaly detection above,
but not fully solved by it.

```bash
python src/parser/level_extractor.py --input data/real_test/some_drawing.pdf
```

## Real-drawing-format now fully wired into the pipeline and dashboard

`level_extractor.py` and `real_lift_identifier.py` (described above) were
initially standalone modules. They are now integrated directly into
`src/dashboard/pipeline.py` as an automatic fallback: when the
synthetic-format extractor finds nothing (no building type, no total
travel, no groups), the pipeline automatically retries using the
real-drawing-format extractors on the same already-extracted text (no
duplicate OCR pass). `app.py`'s dashboard displays the results in a
dedicated "Real-drawing extraction details" section — car labels,
service/firefighter duty flags, the full level/elevation table, and any
gap-anomaly warnings — rather than leaving them buried in raw JSON.

**Two more real integration bugs found and fixed by testing this
end-to-end against the actual real drawing, not just unit-testing each
module in isolation:**

1. **Population estimation silently returned nothing.** It used
   `total_levels_visual` (the visual/CV channel's floor count) as the
   floor-count input — but the visual channel has never been validated
   against real drawings and produced nothing useful for this file. Fixed
   to use the validated `level_extractor` floor count when in
   real-drawing-format mode instead.
2. **Compliance check reported the opposite of reality.** It only checked
   `result.groups` (populated by the synthetic-format extractor, which is
   always empty in real-drawing-format mode) to decide whether a
   Firefighter Lift was present — so a drawing where firefighter duty
   *was* clearly detected (title: "PASSENGER AND FIRE SERVICE LIFT")
   incorrectly showed "a Firefighter Lift is typically required but none
   was detected." Fixed to also check `real_firefighter_duty_mentioned`
   in real-drawing-format mode.

Both were caught specifically because the pipeline was run against the
real file end-to-end after wiring, not assumed correct because each
underlying module passed its own unit tests — a good example of why
integration testing matters even when the pieces are individually solid.
Regression tests for both are in `tests/test_pipeline.py`.

## Visual/CV channel tested against the real drawing: confirmed non-functional, with numbers

Rather than leave this as an assumption, the visual/CV channel
(`visual_extractor.py`) was actually run against the real drawing. The
result quantifies exactly how badly it fails: it reported a "shaft
width" of **18,125mm (18 metres)** and detected **0 floor lines**. Both
are obvious nonsense — real elevator shafts are 1.5–3m wide, and a
12-level building obviously has more than 0 floor lines. This confirms,
with real numbers rather than a guess, that the visual channel's pixel-
intensity calibration (built entirely against this project's own clean
synthetic renderings) does not transfer to real drawings, which have far
denser and more varied linework. **The visual channel should not be used
on real drawings without a genuine object-detection retrain (e.g. YOLO),
which was always the proposal's longer-term plan** — this finding
confirms why that step is necessary, not optional.

## Spatial (OCR bounding-box) dimension extraction — a third approach that works

`level_extractor.py` and `real_lift_identifier.py` work because their
labels and values sit next to each other in plain OCR text. Shaft width
does not — the drawing conveys it through a dimension line and arrow,
which plain text OCR discards entirely (confirmed: `"CLEAR SHAFT WIDTH"`
and its actual numeric value are nowhere near each other in the OCR text
stream). `src/parser/spatial_dimension_extractor.py` solves this using
Tesseract's **word-level bounding-box output** (`image_to_data`, not
`image_to_string`), associating each label with the nearest numeric
value by pixel proximity — prioritising horizontal (X-axis) alignment,
since dimension text sits directly above/below its dimension line,
sharing the line's X position.

```bash
python src/parser/spatial_dimension_extractor.py --input data/real_test/some_drawing.pdf
```

**Confirmed against the real drawing**: three separate "SHAFT WIDTH"
label occurrences (presumably three different lift shafts on the same
floor plan) all independently resolved to the **same value, 2550mm**,
with small horizontal offsets (67px, 3px, 2px) — convergent evidence
this is genuinely correct, not a coincidence, and 2550mm is a highly
plausible real elevator shaft width.

**A genuine performance finding, not a bug, documented rather than
hidden:** `image_to_data` (full layout analysis) took **~75 seconds per
page** at the 300 DPI resolution needed for small dimension text to be
readable at all. This module's page-selection design evolved twice
during development based on real measurement — see "Spatial dimension
extraction extended to car_depth" below for the full story, including an
attempted speed optimisation that was tested and reverted after being
found not to help.

## Broadened level-label coverage — an informed extension, not proof of generalization

`level_extractor.py`'s `LEVEL_LABEL_PATTERN` was originally restricted to
only the prefixes observed on the one real drawing available (P, B, GR,
RF, F). Extended to cover other common real-world AEC level-naming
conventions based on general domain knowledge — GF, LG, UG, MZ, PH, TER,
FL, LVL, and S (basement) — each individually checked for false-positive
risk before adding. Notably, **"L" (Level) was deliberately NOT added**,
despite being common in practice: it's already used for lift CAR labels
elsewhere in this project, and adding it here would create a direct,
unresolvable ambiguity between "Level 5" and "Car L05" that cannot be
disambiguated from text alone.

**Be clear about what this does and doesn't prove**: with only one real
drawing available, this broadening cannot be *verified* against further
real files — it's an informed extension based on known conventions, not
proof the system now "works for all drawings." Re-running against the
original real drawing confirmed zero regressions (same 12 levels, same
95.7m travel, same anomaly flag) and 12 new tests cover every added
prefix plus the critical collision-avoidance cases (bare "S" correctly
still rejected to avoid matching "SERVICE"/"SUMP"/"STAIR"; bare "L"
correctly excluded entirely).

## Spatial dimension extraction extended to car_depth — and a real dead-end, reverted

`spatial_dimension_extractor.py` now also extracts **car depth** (found
genuinely present in the real drawing as "CAR DEPTH", confirmed via two
independent label occurrences both resolving to the same value, 1650mm
— convergent evidence alongside shaft width's three-way convergence on
2550mm).

**A real finding worth knowing if extending this further**: an attempt
was made to speed up multi-page processing with a "cheap" pre-screening
pass (plain text OCR first, to skip pages with no relevant keywords
before running the expensive bounding-box OCR). Measured directly
against the real drawing, this did NOT help — reliable keyword detection
itself required the same 300 DPI resolution as full extraction (100/150
DPI found nothing at all; 200 DPI found some pages but missed others
that 300 DPI correctly flagged). The two-pass approach (140s screening +
225s selective extraction ≈ 365s) was measured to be **slower** than
simply running full extraction on every page directly (4 pages × 75s =
300s). This was reverted after being tested and found not to deliver
its assumed benefit, rather than kept because it seemed like it should
help — a good example of measuring rather than assuming when optimising.

**Also confirmed, not assumed**: "SHAFT WIDTH" and "CAR DEPTH" appear on
*different* pages of the same document (page 2 and page 4 respectively)
— disproving an earlier assumption that a drawing set's key dimensions
cluster on its early pages. `extract_labelled_dimensions` now defaults
to processing **all** pages for completeness, with an optional
`page_indices` parameter to restrict scope deliberately when you already
know which pages matter (trading completeness for speed on purpose,
not by accident).

```bash
python src/parser/spatial_dimension_extractor.py --input data/real_test/some_drawing.pdf --pages 1,3
```

## Extended to 7 building typologies for ML training diversity

Originally 5 typologies (residential, commercial, hotel, hospital,
mixed-use). **Warehouse and villa added** — not as copies of existing
ranges with different numbers, but with genuinely different lift
configurations reflecting real-world differences:

- **Villa**: 2-4 floors, a **single small "home lift" car** (≤450kg,
  ≤1.0m/s) — never a multi-car group, never a service group, never a
  firefighter lift (too small a building to need one; the floor-to-floor
  × floor-count math also never reaches the 23m FFL threshold given
  these ranges).
- **Warehouse**: 1-3 floors, very tall floor-to-floor height (6-9m, for
  racking/forklift clearance), and — genuinely different from every
  other type — a **goods lift by default (80% of projects) instead of
  the standard service lift**, reflecting that freight movement, not
  staff amenity, is the real secondary requirement in a warehouse.

Both use restricted basement ranges (0-1, not the tower-scale 0-3 every
other type uses) and their own population-density and traffic-target
assumptions (`ROUGH_PERSONS_PER_FLOOR`, `TARGET_CRITERIA` — all honestly
labelled as estimates, consistent with every other assumption in this
project).

**Two real bugs found and fixed while generating a test batch, not
assumed correct from the code alone:**

1. **`force_short` ignored per-type floor ranges.** It used a flat
   `(2,5)` range for every building type regardless of that type's own
   realistic bounds — so a "short" warehouse (meant to stay in the
   realistic 1-3 floor range) could come out with 5 floors, silently
   violating its own documented constraint. Fixed to clamp against each
   type's `FLOOR_COUNT_RANGE`.
2. **A keyword collision misclassified villas as residential.**
   `real_lift_identifier.py`'s building-type inference added "PRIVATE
   RESIDENCE" as a villa keyword — which contains "RESIDENCE" as a
   substring, colliding with residential's own keyword and being
   matched first due to dict ordering. Confirmed with a direct test
   (`infer_building_type("SMITH FAMILY PRIVATE RESIDENCE, DUBAI")`
   returned `"residential"`, not `"villa"`) before fixing it by removing
   the ambiguous overlap entirely, rather than relying on a fragile
   ordering fix.

**Honest scope note**: these two typologies are genuinely different in
configuration *logic*, but their dimensional ranges (shaft sizes, load
ranges) are still estimates informed by general domain knowledge, not
verified against a real warehouse or villa drawing — none was available
to test against this session, unlike the office building this project's
other real-drawing findings are grounded in.

## Test fragility found and fixed: hardcoded filenames don't survive regeneration

Regenerating the dataset with more projects (150 vs the earlier 30) and
new building types silently broke several tests — not with a failure,
but by **finding nothing and quietly skipping**, providing zero real
coverage without any signal that had happened. Several tests hardcoded
a specific filename (e.g. `"158-1003"`, assumed to always be "the
project with 3 cars and a truncated label") from one particular
generation run. Regenerating shifts the entire random sequence, even
with the same seed, so a fixed filename no longer reliably has the same
properties.

Fixed by searching the **current** ground truth dynamically for
whatever project currently satisfies the needed condition
(`_find_project_matching` in `tests/test_pipeline.py`), rather than
assuming a fixed seed reproduces identical content forever. One test
(the drawing/spec contradiction case) now gracefully skips when this
particular 150-project batch happens to contain zero projects matching
both conditions simultaneously — confirmed this is genuine statistical
variance (10 travel-contradiction projects × 17 FFL-missing projects
among 150 independent random draws lands on zero overlap by chance
roughly as often as not), not a bug, and skipping honestly is the
correct behaviour here, not a false assertion.

## Third and fourth real drawings: a warehouse and a villa — genuinely new structural patterns

Two more real drawings were tested (a warehouse/logistics building and a
private villa, both real Dubai projects), on top of the office building
from earlier. Both revealed patterns the office building alone never
would have surfaced, confirming the office building's conventions were
never going to generalize on their own.

**AutoCAD MTEXT control codes leaking into extracted text** — confirmed
via the villa drawing: raw formatting codes like `\A1;300`, `%%P0.00`
(AutoCAD's legacy plus/minus symbol — genuinely unrelated to MTEXT's
`\P` paragraph break despite the similar letter), and
`\pxqc;{\W1.5;FUTURE\PKIDS POOL}` appeared directly in the text layer
instead of being resolved to their final visual form. Left unstripped,
these actively corrupt extraction — a dimension sitting right next to
its own alignment code, or a paragraph break splitting "FUTURE KIDS
POOL" into two words wrongly joined as one. A `clean_cad_control_codes()`
function already existed in the codebase (found before duplicating it)
but had zero test coverage — given 9 new tests, each built from an
exact real code found in this drawing, not a hypothetical example.

**Value comes FIRST, not last** — the villa states levels as `"+5.80
FIRST FLOOR F.F.L"`, the exact opposite order from the office building's
`"P3_PODIUM3"` then `"12200.000"` convention. A second, complementary
extraction path (`extract_value_first_levels`) was built for this. Two
real bugs were found and fixed while testing it against the actual
drawing text, not assumed correct on the first attempt:
1. The pattern required uppercase-only text, but this drawing's native
   PDF text layer preserves genuine mixed case ("Main Villa", "Gate
   Level") even though other labels in the same drawing are fully
   uppercase — found zero matches until fixed.
2. The pattern required an explicit `+`/`-` sign, but "Gate Level" (the
   0.00 datum reference) has no sign at all once `%%P` is stripped —
   missed entirely until fixed.

**Units aren't consistent even in mm vs metres** — the villa's `"5.80"`
genuinely means 5.80 **metres**, while the office building's
`"12200.000"` means millimetres. A magnitude+precision heuristic (2
decimal places and under 1000 → metres; otherwise already mm) resolves
this, explicitly flagged as an inference from two confirmed real
examples, not a universal rule.

**Not every level is a floor the lift serves** — without filtering,
this villa (a real G+1 building) would show 5 "floors" and 11.6m
travel, because "Gate Level" (site entrance elevation reference) and
"Parapet Level" (roof perimeter edge, not occupied or lift-accessible
space) aren't floors at all. A narrow, deliberately conservative
exclusion list (`NON_FLOOR_REFERENCE_KEYWORDS`) fixes this — now
correctly shows 3 floors, 9.5m travel. "Roof S.S.L" was deliberately
left included rather than also excluded: forcing an exact match to
"this is a G+1 building" on one example's evidence risks the opposite
mistake (undercounting a genuine lift-accessible roof) on a different
building.

**No numbered car ID at all** — the villa's single lift is labelled
just `"LIFT"` (and `"LIFT LOBBY"` for its lobby), with nothing to
number in a one-lift building. `detect_bare_lift_label()` handles this
as a fallback — only when no numbered labels exist, confirmed not to
override genuine numbered labels when both are present.

**Dimensions embedded directly in the label** — the warehouse's lift is
labelled `"LIFT 2.0X1.6"`, stating the car's plan dimensions in the
label itself rather than as a separate callout elsewhere.
`extract_car_dimensions_from_label()` handles this new pattern.

## Proposing minimum pit depth and headroom when the drawing doesn't state them

None of the three real drawings tested this session (office, warehouse,
villa) stated pit depth or headroom overhead clearance anywhere —
confirming this is the common case, not an exception worth ignoring.
`src/traffic_engine/minimum_requirements.py` proposes a minimum for
both, sourced to the KONE MonoSpace 500 Planning Guide (the same source
already used for the load/speed envelope elsewhere in this project):

- **Pit depth**: linearly interpolated between the guide's three
  published data points (1050mm @ 1.0m/s, 1200mm @ 1.6m/s, 1550mm @
  1.75m/s / with counterweight safety gear) — explicitly flagged as
  interpolation, not an independently verified value for every speed.
- **Headroom**: Car Height + 1500mm, disclosed as a representative
  minimum within the guide's real stated range of CH+1300 to
  CH+1580mm (which varies by ceiling type, entrance configuration, and
  frame type) — never presented as one precisely verified number.

Every value returned carries `is_proposed_not_extracted: True` and a
note explaining exactly how it was derived. Wired into the dashboard
per lift group, whenever a rated speed is known.

## Fifth real drawing: a hospital — two more confirmed bugs, both from cross-drawing conflicts

A hospital (lift details only) drawing revealed two further real bugs —
both from the SAME underlying pattern: a signal that was reliable on
one real drawing turned out to mean something different, or collide
with something else, on another.

**Building type misclassified due to keyword priority, not just
collision** — the hospital's actual title clearly states "HAMDAN BIN
RASHID CANCER HOSPITAL", but the SAME title block also mentions "Local
Support Office" and "Lead Design Office" (consultant role titles,
describing who did the work, not what the building is). Since
`infer_building_type` checked "commercial" (via the generic "OFFICE"
keyword) before "hospital" in plain dict order, the drawing was
misclassified as commercial — confirmed directly before fixing.

Fixed with an explicit `PRIORITY_ORDER`: specific, essentially
unambiguous building names (HOSPITAL, HOTEL, WAREHOUSE, VILLA) are now
checked before the generic, collision-prone COMMERCIAL/RESIDENTIAL
keywords, on the reasoning that a hospital project is very unlikely to
have an unrelated "commercial" mention, but very likely to have generic
consultant-role text mentioning "office" somewhere.

**A far more serious bug: floor references counted as lift cars** — the
hospital's lift plans read `"PUBLIC / VISITOR ELEVATORS [P1~P3] FLOOR
PLAN @ G, L1~L3, L5~L7"`. The actual car IDs are **P1, P2, P3**
(3 cars). But `"L1~L3, L5~L7"` — floor LEVEL references describing
which floors that plan sheet shows — use the exact same "L + digit"
shape my car-ID pattern was built around from the office building,
where "L01".."L08" genuinely WERE car IDs. Confirmed directly: the
original pattern alone reported **9 "cars"** against this drawing
(every floor reference it found), when there are actually 3.

This is the clearest evidence yet that "L + digit" is not a reliable
car-ID signal on its own — it means "Lift" on one real drawing and
"Level" on another. Fixed by adding `extract_bracket_range_cars()`,
which recognises the more specific `"ELEVATORS [P1~P3]"` bracket-range
notation and gives it priority over the generic pattern whenever
present — the generic pattern remains the fallback for drawings (like
the office building) that don't use bracket notation at all.

**Genuinely new, extractable patterns spotted but not yet built**,
worth returning to:
- `"1600 KG - 21 PERSONS"` — a clean, directly-parseable rated
  load/capacity statement, a pattern not seen in any of the previous
  four drawings.
- `"CW 2100"` / `"CD 1600"` — abbreviated Car Width/Car Depth labels
  with values immediately adjacent, unlike this same drawing's "SHAFT
  WIDTH"/"SHAFT DEPTH" column headers, which appear to have their
  actual numeric values in a separate table column the plain text
  stream doesn't preserve adjacent to the label — the same spatial/
  table extraction problem already solved for the warehouse's shaft
  width, not yet attempted for this drawing's apparent level-schedule
  table (`G`/`L4`/`L8`/`R` + `FFL`, values not adjacent in plain text).

## Sixth real drawing: a hotel — confirms the office building was the outlier, not the rule

A hotel drawing (10 pages, lift details) surfaced two more real bugs —
and, more importantly, confirmed a pattern first suspected from the
hospital: **the office building's "L01".."L08" car-ID convention is
looking like the exception across real drawings, not the default.**
Both the hospital and this hotel independently use "L" + digit to mean
FLOOR level, with genuine car IDs using other prefixes entirely (P1-P3
for the hospital; HPL1-HPL4 and RPL1-RPL4 for this hotel).

**A physically impossible 215m travel, traced to a safety note, not a
level** — the drawing states `"HEIGHT OF 2.50M ABOVE THE FLOOR OF
LOWEST SERVING FLOOR"` (a general note about pit separation screen
height), which got captured as a level reading purely because it
contains the word "FLOOR" — one of `VALUE_FIRST_KEYWORDS`. Confirmed
directly before fixing: this produced a fake 187.9m gap, more than
double this hotel's real 25.3m travel. Fixed with
`is_likely_prose_fragment()` — a captured name containing common
sentence-connector words (OF, ABOVE, THE, FROM, WITH...) is rejected,
since no genuine level label confirmed across any real drawing tested
so far ("FIRST FLOOR F.F.L", "Main Villa", "Gate Level") contains any
of them.

**Car count wrong again, in a new way** — this hotel expresses its
served-floor range as `"HPL1 & HPL4 - L1(G), L3, L4, L5, L6 ~ L16"`,
while its actual car range appears elsewhere as `"ELEVATORS - HPL1 ~
HPL4"` (a dash-separated variant of the hospital's bracketed
`"[P1~P3]"`). A first attempt at excluding floor-range "L" tokens only
stripped the ONE token directly touching the "~" character, missing
every other comma-separated item in the same list — confirmed directly
(L1, L3, L4, L5, L16 all survived) before switching to a
**proximity-based** exclusion: any "L" + digit match within 80
characters of a `~` anywhere in the text is treated as a floor
reference, not a car ID, regardless of exactly how far from the tilde
it sits in the list. Also confirmed and hard-excluded: a bare "L"
prefix is now rejected as a car-range prefix even when it directly
follows the word "ELEVATORS" (`"ELEVATORS - L1 ~ L8"` correctly returns
no cars) — every genuine car-range prefix confirmed across every real
drawing tested (P, HPL, RPL) is never bare "L".

## Seventh real drawing (two files, one project): a marine management facility from a new consultant

Two lift-detail sheets from a genuinely new consultant ("SILA MARINE
MANAGEMENT CENTER", a small G+1 institutional building — a building
type not covered by any existing keyword, correctly returning
`building_type: None` rather than guessing). This drawing's cleaner,
more structured style surfaced a subtle bug the five previous drawings
never triggered, plus two genuinely new, valuable capabilities.

**A car-ID false positive from level ABBREVIATIONS, not floor
references this time** — the previous fixes handled floor-range lists
("L1~L8") corrupting car counts. This drawing revealed a different
mechanism entirely: `"S.S.L"` and `"F.F.L"` (both real, common level
abbreviations) end in the letter "L", and when immediately followed by
their OWN elevation value on the next line (`"S.S.L\n8.400 m"`), the
car-ID pattern's separator — which allowed any whitespace, including a
newline — misread this as car label "L8". Confirmed directly: matches
were found for "L8", "L4", and "L0" that don't exist anywhere as
genuine car labels in this drawing. Fixed by restricting the separator
to same-line space/hyphen only, since every genuine car label confirmed
across all seven real drawings tested is written as a single
contiguous token on one line.

**A third level-naming convention** — distinct from both the office
building's short-code convention (`"P3_PODIUM3"` then `"12200.000"`)
and the villa's value-first convention (`"+5.80 FIRST FLOOR F.F.L"`):
this drawing spells the name out in full with NO short code at all,
alone on its own line, with the value on the line immediately after
(`"GROUND FLOOR FFL"` then `"+0.30 m"`). `extract_spelled_out_name_first_levels()`
handles this, reusing the same keyword list and prose-fragment/unit-
inference logic already built for the value-first path. Also confirmed:
`"ROAD LEVEL"` is a site/civil reference here too (exactly like the
villa's `"Gate Level"`), now added to the non-floor exclusion list.

**A new, genuinely simple dimension-extraction win** — unlike the
warehouse's `"CLEAR SHAFT WIDTH"`, where the label and its value were
nowhere near each other in the text stream and needed OCR bounding-box
analysis to associate, this drawing states seven dimensions (cabin
width/depth, shaft width/depth, door width/height, and — for the first
time — an explicit **pit depth**) as a clean label immediately followed
by its value on the very next line. `extract_labeled_dimensions()`
pulls all seven directly via plain text, no OCR needed. Confirmed
correctly distinguishing genuinely different values between the
project's two lifts (LIFT-01: 2300×2000mm shaft, 2000mm pit; LIFT-02:
1800×2300mm shaft, 1000mm pit) while correctly recognising identical
shared values (both cabins 1500×1200mm).

## Real bugs found and fixed along the way

This project has been built with a test-first, stress-test-everything
approach — every stage was deliberately pushed with realistic messiness
before being considered "done." Worth knowing about for your dissertation's
evaluation methodology section:

1. **"PL" substring collision** — the abbreviation for passenger lift ("PL")
   was matching inside unrelated words like "PLAN". Fixed with word-boundary
   matching.
2. **"SL01"-style labels not matching** — after fixing #1 too strictly, real
   lift labels with no separator before digits (e.g. "SL01") stopped
   matching. Fixed with a boundary rule that allows a digit suffix.
3. **"EL"/"PL" discipline code collision** — found via distractor testing:
   these elevator abbreviations are also standard AEC discipline codes
   (Electrical, Plumbing), causing every electrical/plumbing drawing in a
   project folder to be wrongly flagged relevant. Fixed by requiring a
   digit suffix specifically for these codes.
3. **"FIRE" / "FF" false positives** — also found via distractor testing:
   "FIRE" alone matched ordinary fire-suppression/fire-fighting MEP
   drawings (a completely different, common drawing category from
   firefighter lifts), and "FF" fuzzy-matched against "Finished Floor" (a
   common level marker). Both removed from the keyword lexicon in favour
   of more specific terms (FFL, FIREFIGHTER, EN81-72, FIRE LIFT).
4. **Extraction accuracy collapse under format variation** — the first
   version of the parameter extractor scored 100% against clean,
   consistently-formatted text, but dropped to ~30% once realistic format
   variation was introduced (different offices phrasing floor height as
   "F-F HT: 3700mm" vs "FLOOR TO FLOOR: 3.70m" vs "F/F HEIGHT = 3700").
   Rewrote the extraction to use label alternation and unit normalisation
   rather than one fixed pattern per field.
5. **Apparent "errors" that were actually correct** — some floor-to-floor
   mismatches after fix #4 were the parser correctly reading a value that
   the drawing itself only stated to 2-decimal-place metre precision
   (losing a few mm). This isn't an extraction error — it's realistic
   precision loss in that drawing convention. Fixed by adding a small
   tolerance to the *evaluation*, not the extractor. Worth discussing in
   your dissertation as a genuine example of ambiguous ground truth.
6. **Floor line count corrupted by nearby text (visual channel)** — a
   naive gray-intensity scan for floor lines also matched anti-aliased
   text pixels near the header/footer that happened to fall in the same
   intensity range, inflating the count. A spacing-regularity filter was
   tried first but proved fragile (it could be fooled when there were
   only a few real lines, or when a spurious match's spacing coincidentally
   matched the real pattern). Fixed properly by bounding the floor-line
   scan to the shaft's own detected vertical extent instead.
7. **Shaft vertical extent itself corrupted by text, then by the title
   block** — detecting the shaft's top/bottom border via a single-column
   scan picked up nearby label and dimension text (which sits in the same
   column). Fixed by requiring a candidate row to be dark across most of
   the shaft's *width*, not just at one column — but this then picked up
   the title block's own border, which is also wide enough to pass that
   check and sits further down the page. Fixed by taking the first two
   distinct row clusters specifically (the shaft's real top and bottom),
   not the overall topmost/bottommost match.

## Eighth+ real drawing(s): a full multi-file office project, plus independent re-confirmation of the warehouse and villa

A full drawing package was obtained for the office tower project
already described above as "Drawing 1" ("P3_PODIUM3" podium-naming
convention), now available as its actual
multi-sheet package (lift pit details, 5 sheets of passenger-lift
detail plans/sections, and a separate VT (vertical transportation)
consultant package with its own pit/shaft/machine-room drawings and a
stacking chart) rather than a single lift-detail sheet. A new villa
project and a new warehouse project (each as a full multi-sheet
architectural set — floor plans, sections, a setting-out plan) were
also obtained, independently exercising the same value-first level
convention and `"LIFT 2.0X1.6"` label-embedded-dimension pattern
already documented above for Drawings 2 and 3 — both reproduced
exactly, with no new fix required, which is itself a useful
confirmation that those two fixes generalise rather than having been
overfit to the single file each was originally built against.

**A new, silent triage failure found and fixed — the same root cause
as the layered-PDF bug above, in a different module.** One file in
this batch (a 41-page, 83MB A1 architecture set) hit the same
Optional-Content-Group parsing issue already documented for the main
text extractor, but in `file_triage.py`, which has no pdfplumber
fallback at all: PyMuPDF returned next to no text for every page
*without raising a Python exception*, which the triage stage's
`except Exception` could not catch — it doesn't fail loudly, it just
quietly returns "not relevant." Confirmed directly: this exact file
was silently marked not elevator-relevant, indistinguishable from a
genuinely irrelevant drawing. Fixed by detecting the pattern (a
multi-page PDF with next to no extracted text is a suspected read
failure, not a confirmed negative) and falling back to pdfplumber —
but pdfplumber turned out to be **catastrophically slow** on this
same file (92.7 seconds for a single page, confirmed by direct
timing, and still returned nothing), so the fallback is wall-clock
time-boxed to 10 seconds via `signal.alarm` rather than bounded by
page count, which would not have helped. When both tiers fail, the
file is now flagged with an explicit `error` explaining the
uncertainty, rather than silently reported as "not relevant." 3 new
regression tests cover the recovery path, the timeout path, and confirm
a single blank page (the ordinary case, e.g. a cover sheet) does not
trigger this fallback machinery at all.

**A genuine extraction win: 7 real lift cars correctly identified from
one drawing.** The VT consultant's own pit-plan drawing labels seven
passenger cars `PL-01` through `PL-07` plus an eighth `MU01`
(multi-utility) car; `real_car_labels` correctly returned
`['PL1', 'PL2', 'PL3', 'PL4', 'PL5', 'PL6', 'PL7']` — the most cars
correctly extracted from a single real drawing so far. `MU01` was
correctly NOT force-matched into this list (its prefix doesn't match
any known car-ID pattern) rather than being wrongly merged in, which
is the right failure mode, but is itself a gap worth closing: `MU`
(multi-utility) is a new, unrecognised car-ID prefix.

**A new false-positive category: civil/survey spot elevations misread
as building floor levels.** The warehouse project's setting-out plan
contains a dense grid of civil engineering spot-height annotations for
site grading (`27.41`, `27.40`, `27.50`...), completely unrelated to
lift travel. These were captured as level readings under the *stale*
label of the last genuine floor level seen earlier in the document
("M FFL"), rather than being correctly excluded — confirmed directly:
elevations of 26800–28000mm appended to the real building's genuine
0–3110mm range, producing a fabricated 23.69m "gap" that the existing
anomaly detector did flag as suspicious, but for the wrong reason (it
read as a possible OCR misread, not as two unrelated data sources
being merged). This is a real, not-yet-fixed gap: the level extractor
has no way to know a civil site plan's coordinate grid is a
different *kind* of number from a building's floor levels, since both
are written as a bare `+XX.XX` value. A fix would need to scope level
extraction to pages/regions that are genuinely lift-plan or
section drawings, not just any page containing an `FFL`-labelled
value anywhere in the document — worth returning to, not attempted
here.

**Also confirmed, not yet built:** the same VT drawing's shaft-section
sheet shows a full stacking chart with 13 named levels (Basement 01/02,
Ground, Podium 1–4, Level 5, Roof, Upper Roof) in a format distinct
from every level-naming convention documented so far (a table-based
callout rather than inline drawing text); only 3 of these 13 were
picked up by the existing level extractors. A fourth level-naming
convention, specific to this table format, would be needed to close
this gap — noted here rather than attempted, since the existing
value-first/spelled-out-name extractors were built against inline
drawing text, not tabular stacking charts.

**A second, related gap found and fixed while testing this batch: the
contradiction detector couldn't tell "no contradiction" from "the spec
couldn't be read at all."** The same real specification document
(`Elevator_specs.pdf` — a genuine table-format spec sheet, not this
project's synthetic format) was paired with one of the real drawings
above to test contradiction detection end-to-end for the first time
against non-synthetic input on both sides. `spec_reader.py`'s patterns
are built for the synthetic spec format only (`"Total travel: X m"`,
etc.) and matched nothing at all in the real sheet — every field came
back `None`. Confirmed directly: this produced `has_contradiction:
False, flags: []`, identical to a genuinely clean project, when in
truth nothing was actually compared. This directly contradicts the
project's own stated design principle (flag ambiguity, don't guess)
and was fixed: `ContradictionResult` now carries a
`spec_extraction_uncertain` flag, set when both compared fields come
back empty from the spec document, with an explicit message
distinguishing "verified no contradiction" from "could not verify."
3 new regression tests cover this, plus confirming a spec with even
one field readable is correctly NOT treated as a total read failure.
A genuine real-spec-format reader, symmetric to the real-drawing-format
fallback already built for drawings, would still be needed to actually
compare a real specification's values — this fix only ensures the gap
is reported honestly rather than mistaken for a clean result.

**Two more files from this same upload, briefly:** a 62-page interior/ID
drawing set from a different, unnamed project processed cleanly in
10.4 seconds with real content on 32 of its 62 pages — but a thin
result rather than a rich one: a single generic `"LIFT"` car label, one
level reading, and correctly no building-type guess (no type-identifying
keyword was present in the sampled text, and the classifier didn't
invent one). Not every successfully-read real drawing yields dense
structured data, and this is a useful, honest contrast to the VT
drawing's 7-car result above. Separately, the 41-page file that
triggered the triage fix above remains unreadable in practical time
even with that fix in place — pdfplumber's 92.7-second-per-page rate
would need roughly 25 minutes minimum for this file at 5 pages sampled,
let alone all 41; this is recorded as a real, unresolved limitation of
the current text-layer-only approach for this specific category of
CAD export, not something the timeout mechanism was meant to solve.

## Ninth finding: a specification table with no selectable text at all — vector-outlined, not missing

Requested after a user review of the dashboard's real-drawing output:
per-lift specification details (load capacity, speed, stops/landing
doors, travel, power) were completely absent from the real-drawing
results view, even though the VT (vertical transportation) consultant's
own drawing clearly has a "BRIEF SPECIFICATION" table with exactly this
data. Investigating why revealed something more specific than "not yet
extracted": the table's heading and every surrounding note on that page
**are** in the PDF's normal text layer (confirmed directly — the string
"BRIEF SPECIFICATION" is found by a plain PyMuPDF text search), but the
table's own data values are not, in either PyMuPDF or pdfplumber output.
This is the same "text flattened to vector curves" phenomenon already
documented in `text_extractor.py`'s OCR fallback, but here it affects
only one table on an otherwise perfectly readable page, so the existing
near-empty-page OCR trigger never fires — this page has thousands of
characters of normal text, just not this one table's.

Confirmed by rendering the page and reading the table by eye: the data
is genuinely there and legible, just not as extractable text. Full-page
OCR was tried first and confirmed too low-resolution to read reliably —
an A1 sheet's worth of pixels spent mostly on drawing geometry leaves the
one small table's text tiny even at a generous page DPI. The fix:
locate the table's heading via the (fast, reliable) text layer to get
its on-page coordinates, crop generously around just that region, and
OCR only the crop at high DPI. This recovered the table with no errors
against the real drawing tested.

A new module, `real_lift_specification_extractor.py`, implements this
and handles a genuine table-reading wrinkle found along the way: OCR
splits this table's rows inconsistently — some fields land on one line
with their values ("CAPACITY 1350 Kgs / 18 PERSONS 1600 Kgs / 21
PERSONS"), others split label and values across several lines (LIFT
DESIGNATION's three group labels each on their own line). Both are
handled by first grouping OCR'd lines into per-field blocks (everything
between one recognised label and the next), then parsing each block's
values — using a known value-shape regex (e.g. `\d+\s*Kgs?\s*/\s*\d+\s*
PERSONS`) where whitespace alone doesn't reliably mark the column
boundary, which turned out to matter: OCR did not preserve a wide gap
between "PERSONS" and the next cell's "1600", so a pure whitespace
split silently produced one merged value instead of two.

Lift groups that turn out to share every extracted field value are
merged into one displayed row (e.g. "PL-01 ~ PL-07") rather than
repeated — verified against the real drawing, where all seven passenger
cars share identical values and the multi-utility car MU01 correctly
stays on its own row with its own values. This directly satisfies the
dashboard requirement as asked: show shared specs once, not per car.

Scope, stated plainly: this has been built against and verified on one
real drawing's table format. The field-label list and value-shape
patterns are specific to what that table actually contains; a
differently-labelled specification table (a different consultant's
template, for instance) would need its labels added to `FIELD_LABELS`
to be recognised at all. Cabin finishes are not extracted by this
module because they were not present in the one table format tested —
that level of interior detail tends to live on ID/interior drawings,
not a VT consultant's brief specification sheet. 6 new unit tests cover
the parsing/grouping logic directly (the actual unit of correctness
here, since OCR quality itself isn't something a unit test can verify).

## Sourcing real data (eventually)

There's no ready-made public dataset labelled for elevator specification.
EMAAR-style developer sites only publish simplified sales floor plans (no
shaft dimensions, no lift labels). Realistic places to look:

- **Public procurement / tender portals** (e.g. Dubai's eSupply) — real
  technical drawing packages, though most require supplier registration.
- **CubiCasa5k** (https://github.com/CubiCasa/CubiCasa5k) — general
  floor-plan dataset, useful for a floor-plan-understanding baseline,
  though it doesn't label elevators specifically.

## Roadmap status — all originally-scoped WBS stages complete

1. ✅ File triage
2. ✅ Synthetic dataset generator
3. ✅ Text-layer parameter extraction
4. ✅ Visual/CV channel (shaft outlines, floor count, shaft width)
5. ✅ Merged text + visual cross-validation
6. ✅ Rule-based lift group classifier
7. ✅ Traffic study engine (CIBSE Guide D uppeak RTT method)
8. ✅ Specification recommender (rule-based search, documented as a
   scope decision in place of the proposal's XGBoost model — see above)
9. ✅ Contradiction detection (total travel + shaft width; floor count
   scope-limited — see above)
10. ✅ Dashboard — Streamlit app tying everything together

## Honest next steps beyond this session's scope

- Real drawing data — extended real-drawing validation. Eight-plus
  building projects have now been tested (see the real-drawing
  sections above), which is a solid qualitative validation base, but
  still nowhere near the 60-100 annotated packages the original
  proposal scoped for training a supervised model — that remains the
  actual gap, not "no real drawings tested at all."
- ML-based specification recommender — the proposal's originally-scoped
  XGBoost model, once real historical data or enough rule-based-search
  outputs exist to train on meaningfully (see the recommender's module
  docstring).
- Exact numeric validation of the traffic study engine against the
  actual CIBSE Guide D document (see that module's docstring for what
  was and wasn't possible to verify in this session).
- Floor-count contradiction detection, once a consistent floor-counting
  convention is resolved between the drawing and specification-document
  extractors (see the contradiction detector's documented scope limit).
- Multi-page / multi-drawing project folders — everything here processes
  one drawing at a time; the original 16GB-project-folder triage-then-
  deep-process pipeline design (Stage 1) hasn't been exercised at that
  scale.
