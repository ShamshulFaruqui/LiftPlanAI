"""
Estimates building-level parameters (floor-to-floor height, floor count)
from real-drawing level readings, for the specific purpose of letting the
specification recommender (spec_engine/recommender.py) run on a real
drawing that has no specification table of its own — see pipeline.py for
where this feeds in, and Chapter 7 of the dissertation for why this is
framed as a best-effort estimate, not a fix for the underlying level-
misread issue it works around.

Why this exists: asked directly after a user review noted that a real
drawing with no "BRIEF SPECIFICATION" table (Section 5.7) got no
specification detail of any kind, even though enough was extracted to
make a reasonable estimate (building type, and — usually — a set of
level readings). The gap is real: level readings can include the
civil-survey misread already documented as a limitation, and a naive
count or gap calculation is not robust to it (confirmed directly, see
below).

Why "largest cluster of readings" does NOT work: tested directly against
a real drawing carrying exactly this contamination, where a civil-survey
spot-height grid on the same page contributed MORE distinct readings (9)
than the building's own genuine floor levels (8) — the larger cluster
was the wrong one. Picking the cluster whose readings sit closest to a
local zero (median absolute elevation) instead correctly identified the
real building levels in that case, on the reasoning that architectural
drawings near-universally reference floor levels from a project datum
close to 0, while a civil/survey elevation grid on the same sheet is a
much larger, site- or sea-level-referenced number. This is a heuristic,
not a guarantee: a real project that references its own floor levels
from an absolute site datum (rather than a local 0) would defeat it.
"""

from dataclasses import dataclass
import statistics

GAP_OUTLIER_MULTIPLE = 3  # matches the existing anomaly detector's own threshold

# A real building's floor-to-floor height is bounded in practice (from a
# low-ceiling residential floor to a tall industrial/warehouse floor).
# Used as a plausibility check on the estimate below: a computed value
# outside this range is far more likely to mean the "building" cluster
# still contains non-floor annotations (door thresholds, ramp levels,
# paving heights) than a genuine floor-to-floor height — confirmed
# directly against a real drawing whose ground-floor plan carries many
# such small level annotations, where the cluster this module correctly
# excluded of civil-survey noise still produced an implausible 0.15m
# "floor height" from those finer annotations. Rather than pass that
# through with confidence, the estimator refuses in this case.
PLAUSIBLE_FLOOR_HEIGHT_M = (2.2, 15.0)


@dataclass
class BuildingClusterEstimate:
    floor_height_m: float
    num_floors: int
    cluster_elevations_mm: list
    warning: str


def estimate_building_cluster(elevation_mm_values: list) -> "BuildingClusterEstimate | None":
    """
    Given a list of raw elevation readings (mm, possibly containing
    unrelated contaminating data), returns a best-effort estimate of the
    building's own floor-to-floor height and floor count, or None if
    there are too few distinct readings to estimate anything from (fewer
    than 3 — two floors give one gap, which is not enough to distinguish
    a real pattern from noise).
    """
    values = sorted(set(elevation_mm_values))
    if len(values) < 3:
        return None

    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        return None

    clusters = [[values[0]]]
    for i, g in enumerate(gaps):
        if g > GAP_OUTLIER_MULTIPLE * median_gap:
            clusters.append([])
        clusters[-1].append(values[i + 1])

    # Prefer the cluster whose readings sit closest to a local zero —
    # see module docstring for why this, rather than cluster size, is
    # the more reliable signal against the one real contamination case
    # this has been tested against.
    def _score(cluster):
        return statistics.median(abs(v) for v in cluster)

    best = min(clusters, key=_score)

    if len(best) < 2:
        return None

    cluster_gaps = [best[i + 1] - best[i] for i in range(len(best) - 1)]
    floor_height_m = round(statistics.median(cluster_gaps) / 1000, 2)

    if not (PLAUSIBLE_FLOOR_HEIGHT_M[0] <= floor_height_m <= PLAUSIBLE_FLOOR_HEIGHT_M[1]):
        # Refuse rather than pass through an implausible estimate — see
        # module docstring. This is a real, expected outcome for some
        # drawings, not a bug: it means the readings available are not
        # clean enough for this estimation approach, and no recommendation
        # should be computed from them.
        return None

    num_floors = len(best)

    warning = (
        f"Estimated from {num_floors} of {len(values)} extracted level readings "
        f"(the {num_floors} judged most likely to be genuine floor levels, by "
        "proximity to a local zero datum) - a best-effort heuristic, not a "
        "verified floor count. See the dissertation, Chapter 7, for its known limits."
    )

    return BuildingClusterEstimate(
        floor_height_m=floor_height_m,
        num_floors=num_floors,
        cluster_elevations_mm=best,
        warning=warning,
    )
