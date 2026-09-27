# Elevator Keyword Lexicon

This is the fuzzy-matching keyword set used by the file triage stage
(`src/triage/file_triage.py`) to flag which files/pages in a project
folder are elevator-relevant. Expand this list as you encounter new
naming conventions in real drawings.

## Lift group types

| Type | Keywords / abbreviations |
|---|---|
| Passenger | LIFT, ELEVATOR, EL, PL, PASSENGER, ELEV |
| Service | SERVICE, SL, SE, SV, S/L, SERV, SVC |
| Firefighter | FF, FFL, FIRE, FIREMAN, EN81-72, F/F, FIRE LIFT |
| Goods | GOODS, GL, G/L, FREIGHT |
| Bed / stretcher | BED, STRETCHER, STR, HOSPITAL |

## Structural / drawing keywords

SHAFT, HOISTWAY, PIT, OVERRUN, MACHINE ROOM, MRL, CORE

## Drawing type keywords

SECTION, ELEVATION, TYPICAL FLOOR, FLOOR PLAN, LIFT DETAILS,
LIFT SECTION, SHAFT DIMENSION

## Known collision risk: short abbreviations vs discipline codes

`EL`, `PL`, `GL`, `SL` are also standard AEC discipline codes
(Electrical, Plumbing, etc.) and commonly appear as isolated tokens in
filenames (e.g. `...-EL-318-...`). The matcher requires these specific
codes to be followed by a digit (as in a real lift label like `EL01`)
rather than any word boundary — see `DIGIT_SUFFIX_REQUIRED` in
`file_triage.py`. Found via distractor testing — see
`tests/test_file_triage.py` for the regression tests.

## Notes

- Matching should be fuzzy (e.g. `rapidfuzz`), not exact — real drawings
  contain typos and inconsistent abbreviations (e.g. "DETALS" instead
  of "DETAILS").
- Case-insensitive matching, punctuation stripped before comparison.
- Bilingual (English/Arabic) drawings are common in the UAE — if you
  encounter Arabic labels for these terms, add them here too.
