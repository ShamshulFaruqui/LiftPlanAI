"""
File Triage — Stage 1 of the LiftPlan AI pipeline.

Scans a project folder (which may contain hundreds of PDFs across
multiple gigabytes) and flags which files are elevator-relevant,
using fuzzy keyword matching against filenames and embedded PDF text.

This stage requires NO machine learning and NO labelled dataset —
it works on any PDF folder, which makes it the ideal first component
to build and test before any model training begins.

Usage:
    python file_triage.py --input data/raw --output data/processed/triage_report.csv
"""

import argparse
import csv
import re
import signal
import sys
from pathlib import Path
from dataclasses import dataclass, field

try:
    import pymupdf as fitz  # PyMuPDF (modern import name)
except ImportError:
    try:
        import fitz  # fallback for older PyMuPDF versions
    except ImportError:
        sys.exit("Missing dependency: run `pip install -r requirements.txt` first (needs PyMuPDF).")

try:
    from rapidfuzz import fuzz
except ImportError:
    sys.exit("Missing dependency: run `pip install -r requirements.txt` first (needs rapidfuzz).")


# ---------------------------------------------------------------------------
# Keyword lexicon — see docs/keyword_lexicon.md for the full explanation.
# Keep this in sync with that file as you expand it.
# ---------------------------------------------------------------------------
KEYWORD_LEXICON = {
    "passenger": ["LIFT", "ELEVATOR", "EL", "PL", "PASSENGER", "ELEV"],
    "service": ["SERVICE", "SL", "SE", "SV", "S/L", "SERV", "SVC"],
    "firefighter": ["FFL", "FIREMAN", "EN81-72", "F/F", "FIRE LIFT", "FIREFIGHTER"],
    "goods": ["GOODS", "GL", "G/L", "FREIGHT"],
    "bed": ["BED", "STRETCHER", "STR", "HOSPITAL"],
    "structural": ["SHAFT", "HOISTWAY", "PIT", "OVERRUN", "MACHINE ROOM", "MRL", "CORE"],
    "drawing_type": ["SECTION", "ELEVATION", "TYPICAL FLOOR", "FLOOR PLAN",
                      "LIFT DETAILS", "LIFT SECTION", "SHAFT DIMENSION"],
}

# Fuzzy match threshold (0-100). Lower = more lenient (catches more typos,
# but more false positives). 80 is a reasonable starting point.
FUZZY_THRESHOLD = 80

# How many characters of PDF text to scan per page (keeps large files fast).
MAX_CHARS_PER_PAGE = 4000

# Short 2-letter codes that collide with standard AEC discipline codes in
# drawing filenames — "EL" is also the code for Electrical, "PL" for
# Plumbing, "GL" can appear as a level/grid reference. For these specific
# keywords, require a digit immediately after (mimicking a real lift label
# like "EL01") rather than accepting any non-alphanumeric boundary — this
# is what distinguishes a lift label from a discipline code in the same
# filename position (e.g. "...-EL-303-..." is Electrical, "EL01" is a lift).
DIGIT_SUFFIX_REQUIRED = {"EL", "PL", "GL", "SL"}


@dataclass
class TriageResult:
    filepath: str
    is_relevant: bool
    matched_in_filename: list = field(default_factory=list)
    matched_in_text: dict = field(default_factory=dict)  # {page_num: [keywords]}
    page_count: int = 0
    error: str = ""


def _flatten_keywords():
    """Return a single flat list of (category, keyword) pairs."""
    return [(cat, kw) for cat, kws in KEYWORD_LEXICON.items() for kw in kws]


def fuzzy_contains(text: str, keyword: str, threshold: int = FUZZY_THRESHOLD) -> bool:
    """
    Check whether `keyword` appears in `text` with fuzzy tolerance.

    Real drawing labels commonly attach digits directly to the keyword
    with no separator (e.g. "SL01", "EL02", "FFL01"), so a strict
    \\b...\\b word-boundary match is too strict — letters and digits are
    both "word characters" in regex, so there's no boundary between them.

    Instead: the keyword must NOT be preceded by a letter (so "PL" does
    not match inside "PLAN"), and must be followed by either a digit,
    a non-alphanumeric character, or the end of the string (so "SL01"
    matches "SL", but "PLAN" does not match "PL").

    A small set of short abbreviations (see DIGIT_SUFFIX_REQUIRED) also
    collide with standard AEC discipline codes (EL=Electrical, PL=Plumbing)
    that appear in filenames as isolated tokens like "-EL-303-". For these,
    a digit suffix is REQUIRED — a non-alphanumeric or end-of-string
    boundary is not enough — since that's what separates a genuine lift
    label ("EL01") from a discipline code ("EL" alone).

    Falls back to token-level fuzzy matching to catch typos like "DETALS".
    """
    text_upper = text.upper()
    keyword_upper = keyword.upper()

    if keyword_upper in DIGIT_SUFFIX_REQUIRED:
        pattern = r"(?<![A-Z])" + re.escape(keyword_upper) + r"(?=[0-9])"
    else:
        pattern = r"(?<![A-Z])" + re.escape(keyword_upper) + r"(?=[0-9]|[^A-Z0-9]|$)"
    if re.search(pattern, text_upper):
        return True

    # Fuzzy path: check word-level tokens against the keyword (catches typos).
    # Skipped for short keywords (<=3 chars) since fuzzy matching on short
    # abbreviations produces too many false positives — e.g. "FFL" (firefighter
    # lift) fuzzy-matching against "FF" (the common "Finished Floor" level
    # marker), which differ by only one character.
    if len(keyword_upper.replace(" ", "")) <= 3:
        return False

    tokens = text_upper.split()
    kw_token_count = max(1, len(keyword_upper.split()))
    for i in range(len(tokens) - kw_token_count + 1):
        window = " ".join(tokens[i:i + kw_token_count])
        if fuzz.ratio(window, keyword_upper) >= threshold:
            return True
    return False


def check_filename(filepath: Path) -> list:
    """Return list of matched keywords found in the filename itself."""
    matches = []
    name = filepath.stem.replace("_", " ").replace("-", " ")
    for category, keyword in _flatten_keywords():
        if fuzzy_contains(name, keyword, threshold=85):  # stricter for short filenames
            matches.append(f"{category}:{keyword}")
    return matches


def check_pdf_text(filepath: Path) -> tuple:
    """
    Open the PDF and scan each page's embedded text layer for keyword matches.
    Returns (matches_by_page, page_count, error_message).
    """
    matches_by_page = {}
    total_chars = 0
    page_count = 0
    pymupdf_error = ""
    try:
        doc = fitz.open(filepath)
        page_count = doc.page_count
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()[:MAX_CHARS_PER_PAGE]
            total_chars += len(text.strip())
            if not text.strip():
                continue  # likely a scanned/rasterised page with no text layer
            page_matches = []
            for category, keyword in _flatten_keywords():
                if fuzzy_contains(text, keyword):
                    page_matches.append(f"{category}:{keyword}")
            if page_matches:
                matches_by_page[page_num + 1] = page_matches  # 1-indexed for humans
        doc.close()
    except Exception as e:
        pymupdf_error = str(e)

    # PyMuPDF can silently degrade on some real CAD-exported PDFs (a known
    # Optional-Content-Group parsing issue — see extract_text() in
    # text_extractor.py, fixed there for the same failure found on an
    # earlier real drawing) WITHOUT raising a Python exception at all: it
    # just returns next-to-nothing for every page. Left unguarded, that is
    # indistinguishable from a genuinely irrelevant multi-page drawing —
    # confirmed directly against a real 41-page architecture set that hit
    # exactly this path and was silently marked "not relevant" before this
    # fix. A page count above 1 with almost no extracted text is treated as
    # a suspected read failure, not a confirmed negative, and falls back to
    # pdfplumber the same way the main extractor already does.
    if not matches_by_page and total_chars < 20 and page_count > 1:
        # pdfplumber's per-page parsing is far slower than PyMuPDF's on
        # dense, large-format CAD exports — confirmed directly against a
        # real 41-page A1 architecture set (83MB) that hit this exact
        # fallback path: a SINGLE page took 92.7 seconds to parse and still
        # returned zero characters. A page-count limit alone doesn't bound
        # this (5 pages at that rate is 7+ minutes), so the attempt is wall
        # clock time-boxed instead — if pdfplumber can't produce anything
        # from the first page or two within a few seconds, this file gets
        # the same treatment as any other unreadable one: flagged uncertain
        # via `error`, not silently marked "not relevant".
        FALLBACK_TIMEOUT_S = 10
        fallback_text = ""
        fallback_error = None
        has_alarm = hasattr(signal, "SIGALRM")
        if has_alarm:
            def _on_timeout(signum, frame):
                raise TimeoutError(f"pdfplumber fallback exceeded {FALLBACK_TIMEOUT_S}s")
            old_handler = signal.signal(signal.SIGALRM, _on_timeout)
            signal.alarm(FALLBACK_TIMEOUT_S)
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as doc:
                fallback_text = "\n".join(
                    (p.extract_text() or "") for p in doc.pages[:5]
                )
        except Exception as e2:
            fallback_error = str(e2)
        finally:
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        if fallback_error:
            return matches_by_page, page_count, (
                f"possible read failure: near-empty text from a {page_count}-page PDF via PyMuPDF"
                + (f" ({pymupdf_error})" if pymupdf_error else "")
                + f"; pdfplumber fallback also failed ({fallback_error})"
            )

        if fallback_text.strip():
            page_matches = []
            for category, keyword in _flatten_keywords():
                if fuzzy_contains(fallback_text, keyword):
                    page_matches.append(f"{category}:{keyword}")
            if page_matches:
                # Whole-fallback-range match — no per-page breakdown from
                # this path, unlike the primary PyMuPDF loop above.
                matches_by_page[1] = page_matches
            return matches_by_page, page_count, ""
        return matches_by_page, page_count, (
            f"possible read failure: near-empty text from a {page_count}-page PDF via PyMuPDF"
            + (f" ({pymupdf_error})" if pymupdf_error else "")
            + "; pdfplumber fallback (first 5 pages) also found nothing"
        )

    return matches_by_page, page_count, pymupdf_error


def triage_file(filepath: Path) -> TriageResult:
    filename_matches = check_filename(filepath)
    text_matches, page_count, error = check_pdf_text(filepath)

    is_relevant = bool(filename_matches) or bool(text_matches)

    return TriageResult(
        filepath=str(filepath),
        is_relevant=is_relevant,
        matched_in_filename=filename_matches,
        matched_in_text=text_matches,
        page_count=page_count,
        error=error,
    )


def triage_folder(input_dir: Path) -> list:
    """Recursively scan a folder for PDFs and triage each one."""
    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found under {input_dir}")
        return []

    print(f"Found {len(pdf_files)} PDF file(s). Scanning...\n")
    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        result = triage_file(pdf_path)
        status = "RELEVANT" if result.is_relevant else "skip"
        print(f"[{i}/{len(pdf_files)}] {status:>9} — {pdf_path.name}")
        results.append(result)
    return results


def write_report(results: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "is_relevant", "page_count",
                          "filename_matches", "relevant_pages", "error"])
        for r in results:
            relevant_pages = ";".join(str(p) for p in r.matched_in_text.keys())
            writer.writerow([
                r.filepath,
                r.is_relevant,
                r.page_count,
                ";".join(r.matched_in_filename),
                relevant_pages,
                r.error,
            ])
    print(f"\nReport written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Stage 1: File Triage")
    parser.add_argument("--input", type=str, default="data/raw",
                         help="Folder containing PDF drawings to scan")
    parser.add_argument("--output", type=str, default="data/processed/triage_report.csv",
                         help="Where to write the triage report CSV")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists():
        sys.exit(f"Input folder not found: {input_dir}")

    results = triage_folder(input_dir)
    if results:
        relevant_count = sum(1 for r in results if r.is_relevant)
        print(f"\n{relevant_count}/{len(results)} files flagged as elevator-relevant.")
        write_report(results, output_path)


if __name__ == "__main__":
    main()
