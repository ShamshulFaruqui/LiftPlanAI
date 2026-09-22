"""
Tests for the visual (CV) parameter extractor (Stage 2b).
Run with: pytest tests/test_visual_extractor.py -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser.visual_extractor import (
    cluster_columns, detect_shaft_vertical_extent, MM_PER_PIXEL,
    extract_visual_parameters,
)

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "drawings"


# --- cluster_columns: basic clustering behaviour ---

def test_cluster_columns_merges_adjacent():
    # A single anti-aliased line spread across a few adjacent pixels should
    # collapse into one cluster, not be counted multiple times.
    result = cluster_columns(np.array([100, 101, 102, 200, 201]))
    assert len(result) == 2


def test_cluster_columns_empty_input():
    assert cluster_columns(np.array([])) == []


def test_cluster_columns_respects_gap_threshold():
    # Points further apart than the gap threshold should NOT be merged.
    result = cluster_columns(np.array([100, 110]), gap_threshold=3)
    assert len(result) == 2


# --- pixel-to-mm calibration sanity check ---

def test_mm_per_pixel_matches_known_calibration():
    # A shaft rendered at 2100mm width (per generate_dataset.py's scale of
    # shaft_width_mm/40 points) should convert back close to 2100mm at the
    # module's fixed RENDER_DPI. Verified against a real measured sample
    # during development: 109-110px measured -> ~2100mm expected.
    measured_px = 109.5
    mm = measured_px * MM_PER_PIXEL
    assert abs(mm - 2100) < 60  # allows for pixel-measurement rounding


# --- detect_shaft_vertical_extent: the title-block contamination bug ---

def test_shaft_vertical_extent_ignores_title_block_border():
    # Real bug found during development: a naive "last dark row" approach
    # picked up the title block's own border (which spans a similarly wide
    # x-range as the shaft) instead of the shaft's actual bottom edge,
    # since both pass a width-coverage check. Construct a synthetic image
    # with a shaft border at rows 200/800 and an unrelated wide dark band
    # (mimicking the title block) at row 1500, and confirm only the first
    # two clusters (the real shaft borders) are returned.
    img = np.full((2000, 500), 255, dtype=np.uint8)
    img[200, 100:300] = 0    # shaft top border
    img[800, 100:300] = 0    # shaft bottom border
    img[1500, 50:450] = 0    # unrelated wide dark band (title block, wider x-range)
    top, bottom = detect_shaft_vertical_extent(img, 100, 300)
    assert top == 200
    assert bottom == 800


def test_shaft_vertical_extent_ignores_narrow_text():
    # Text characters are narrow relative to the shaft width and should
    # fail the width-coverage check, unlike a genuine full-width border.
    img = np.full((2000, 500), 255, dtype=np.uint8)
    img[200, 100:300] = 0     # shaft top border (full width)
    img[210, 100:110] = 0     # a narrow "text character" just below the border
    img[800, 100:300] = 0     # shaft bottom border (full width)
    top, bottom = detect_shaft_vertical_extent(img, 100, 300)
    assert top == 200
    assert bottom == 800


# --- End-to-end checks against real generated synthetic drawings ---
# Skipped gracefully if the dataset hasn't been generated in this environment.

def _dataset_available():
    return SYNTHETIC_DIR.exists() and any(SYNTHETIC_DIR.glob("*LIFT*"))


def test_end_to_end_shaft_and_floor_count_on_real_sample():
    if not _dataset_available():
        return  # dataset not generated in this environment; skip silently
    sample = next(SYNTHETIC_DIR.glob("*158-1003*"), None)
    if sample is None:
        return
    result = extract_visual_parameters(sample)
    assert result.shaft_count == 2
    assert result.implied_total_levels == 12


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
