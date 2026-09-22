"""
Tests for the real-drawing batch test harness. Since this tool's whole
point is running against drawings not yet available, these tests confirm
its mechanics (empty folder handling, CSV structure) using the synthetic
dataset as a stand-in — not that it produces specific "correct" values,
since that's exactly what it doesn't assume.

Run with: pytest tests/test_real_drawing_test.py -v
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.real_drawing_test import run_batch_test

SYNTHETIC_DRAWINGS = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "drawings"


def test_empty_folder_does_not_crash(tmp_path, capsys):
    run_batch_test(tmp_path, tmp_path / "report.csv")
    captured = capsys.readouterr()
    assert "No PDF files found" in captured.out
    assert not (tmp_path / "report.csv").exists()


def test_batch_run_produces_valid_csv_with_expected_columns(tmp_path):
    if not SYNTHETIC_DRAWINGS.exists() or not any(SYNTHETIC_DRAWINGS.glob("*LIFT*")):
        return  # dataset not generated in this environment; skip silently

    output = tmp_path / "report.csv"
    run_batch_test(SYNTHETIC_DRAWINGS, output)

    assert output.exists()
    with open(output) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) > 0
        expected_columns = {"file", "triage_relevant", "building_type",
                             "floor_to_floor_mm", "total_travel_m", "group_count",
                             "errors", "flags"}
        assert expected_columns.issubset(set(reader.fieldnames))


def test_irrelevant_drawings_are_marked_not_triage_relevant(tmp_path):
    if not SYNTHETIC_DRAWINGS.exists():
        return
    output = tmp_path / "report.csv"
    run_batch_test(SYNTHETIC_DRAWINGS, output)

    with open(output) as f:
        rows = list(csv.DictReader(f))
    structural_rows = [r for r in rows if "-ST-" in r["file"]]
    if structural_rows:
        assert all(r["triage_relevant"] == "False" for r in structural_rows)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


def test_triage_read_error_is_distinguished_from_genuinely_irrelevant(tmp_path, capsys):
    # Real gap found from real-drawing testing: a file that threw a MuPDF
    # error during triage was silently marked "not relevant" with no
    # indication it was actually unreadable, indistinguishable from a
    # correctly-skipped structural/MEP drawing.
    import shutil
    from unittest.mock import patch
    import src.triage.file_triage as ft
    from src.dashboard.real_drawing_test import run_batch_test

    if not SYNTHETIC_DRAWINGS.exists():
        return
    sample = next(SYNTHETIC_DRAWINGS.glob("*LIFT*"), None)
    if sample is None:
        return

    shutil.copy(sample, tmp_path / "unreadable.pdf")

    def failing_check(filepath):
        return {}, 0, "MuPDF error: format error: No default Layer config"

    with patch.object(ft, "check_pdf_text", side_effect=failing_check):
        run_batch_test(tmp_path, tmp_path / "report.csv")

    captured = capsys.readouterr()
    assert "could not be read" in captured.out
    assert "No default Layer config" in captured.out
