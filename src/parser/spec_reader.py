"""
Specification Document Reader — extracts parameters from client
specification PDFs (as distinct from architectural drawings), for
cross-referencing against drawing-extracted parameters in the
contradiction detection stage.

Usage:
    python spec_reader.py --input ../../data/synthetic/specs/SYN-001_SPEC.pdf
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        sys.exit("Missing dependency: run `pip install -r requirements.txt` first (needs PyMuPDF).")


@dataclass
class SpecParameters:
    source_file: str
    building_type: str = None
    floor_count: int = None
    total_travel_m: float = None
    shaft_width_mm: int = None
    requirements: list = None


def extract_spec_parameters(pdf_path: Path) -> SpecParameters:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    result = SpecParameters(source_file=str(pdf_path), requirements=[])

    m = re.search(r"Building type:\s*([a-zA-Z\-]+)", text, re.IGNORECASE)
    if m:
        result.building_type = m.group(1).strip().lower()

    m = re.search(r"Number of floors:\s*(\d+)", text, re.IGNORECASE)
    if m:
        result.floor_count = int(m.group(1))

    m = re.search(r"Total travel:\s*([\d.]+)\s*m", text, re.IGNORECASE)
    if m:
        result.total_travel_m = float(m.group(1))

    m = re.search(r"Primary shaft width:\s*(\d+)\s*mm", text, re.IGNORECASE)
    if m:
        result.shaft_width_mm = int(m.group(1))

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("-"):
            result.requirements.append(line.lstrip("- ").strip())

    return result


def main():
    parser = argparse.ArgumentParser(description="LiftPlan AI — Specification Document Reader")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    result = extract_spec_parameters(Path(args.input))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
