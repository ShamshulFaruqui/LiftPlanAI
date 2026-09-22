"""
Tests for the synthetic dataset generator's building-type diversity
(warehouse and villa, added to support broader ML training data beyond
the original 5 typologies).

Run with: pytest tests/test_generate_dataset.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic.generate_dataset import (
    build_project, BUILDING_TYPES, FLOOR_COUNT_RANGE, BASEMENT_COUNT_RANGE,
)


def test_all_seven_building_types_present():
    assert set(BUILDING_TYPES) == {
        "residential", "commercial", "hotel", "hospital", "mixed-use",
        "warehouse", "villa",
    }


# --- Villa: single small car, no service, no firefighter ---

def test_villa_has_exactly_one_passenger_group():
    villa_projects = [build_project(i) for i in range(1, 40)]
    villa_projects = [p for p in villa_projects if p.building_type == "villa"]
    assert len(villa_projects) > 0, "seed/range didn't produce any villas to test — widen the sample"
    for p in villa_projects:
        assert len(p.groups) == 1
        assert p.groups[0]["lift_type"] == "passenger"
        assert p.groups[0]["num_cars"] == 1


def test_villa_never_has_service_or_firefighter_group():
    villa_projects = [build_project(i) for i in range(1, 40)]
    villa_projects = [p for p in villa_projects if p.building_type == "villa"]
    for p in villa_projects:
        lift_types = {g["lift_type"] for g in p.groups}
        assert "service" not in lift_types
        assert "firefighter" not in lift_types


def test_villa_uses_home_lift_scale_load():
    villa_projects = [build_project(i) for i in range(1, 40)]
    villa_projects = [p for p in villa_projects if p.building_type == "villa"]
    for p in villa_projects:
        assert p.groups[0]["rated_load_kg"] <= 450


# --- Warehouse: goods-focused, not the default service group ---

def test_warehouse_gets_goods_group_more_often_than_service():
    warehouse_projects = [build_project(i) for i in range(1, 60)]
    warehouse_projects = [p for p in warehouse_projects if p.building_type == "warehouse"]
    assert len(warehouse_projects) > 0, "seed/range didn't produce any warehouses to test"

    with_goods = sum(1 for p in warehouse_projects if any(g["lift_type"] == "goods" for g in p.groups))
    with_service = sum(1 for p in warehouse_projects if any(g["lift_type"] == "service" for g in p.groups))
    assert with_goods > with_service


def test_warehouse_never_gets_a_plain_service_group():
    # The building-type branch replaces the default service-group logic
    # entirely for warehouses — it should never fall through to it.
    warehouse_projects = [build_project(i) for i in range(1, 60)]
    warehouse_projects = [p for p in warehouse_projects if p.building_type == "warehouse"]
    for p in warehouse_projects:
        lift_types = {g["lift_type"] for g in p.groups}
        assert "service" not in lift_types


# --- Real bug found and fixed: force_short ignoring per-type floor ranges ---

def test_force_short_respects_warehouse_floor_range():
    # Real bug found via testing a generated batch: force_short used a
    # flat (2,5) floor-count range for every building type, which let a
    # "short" warehouse (realistic range 1-3) come out with up to 5
    # floors — silently violating the type's own documented bounds.
    warehouse_type_min, warehouse_type_max = FLOOR_COUNT_RANGE["warehouse"]
    for i in range(1, 60):
        p = build_project(i, force_short=True)
        if p.building_type == "warehouse":
            assert warehouse_type_min <= p.floor_count <= warehouse_type_max


def test_force_short_respects_villa_floor_range():
    villa_type_min, villa_type_max = FLOOR_COUNT_RANGE["villa"]
    for i in range(1, 60):
        p = build_project(i, force_short=True)
        if p.building_type == "villa":
            assert villa_type_min <= p.floor_count <= villa_type_max


# --- Travel cap: confirm new types stay within the KONE envelope ---

def test_warehouse_and_villa_never_exceed_kone_travel_cap():
    for i in range(1, 80):
        p = build_project(i)
        if p.building_type in ("warehouse", "villa"):
            assert p.total_travel_m <= 75.0


def test_warehouse_and_villa_have_restricted_basement_range():
    assert BASEMENT_COUNT_RANGE["warehouse"] == (0, 1)
    assert BASEMENT_COUNT_RANGE["villa"] == (0, 1)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
