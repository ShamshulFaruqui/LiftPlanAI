"""
Evaluation harness - LiftPlan AI v2
===================================

Runs an extraction system over a dataset split and scores it against the
generator's ground truth. Reports precision / recall / F1 for:

  T1  triage            - is this a lift drawing? (binary, drawing level)
  T2  lift detection    - each lift on the sheet found, by label (instance level)
  T3  lift type         - passenger / service / firefighter / goods, on detected lifts
                          (macro-F1 + confusion matrix)
  T4  per-lift fields   - car width, car depth, door width, shaft width, shaft depth,
                          rated load, rated speed
  T5  building fields   - number of stops, total travel, typical floor-to-floor,
                          pit depth, overhead, building type
  T6  compliance        - "firefighter lift required but missing" flag (binary)

Field scoring convention (T4/T5): a field is only scored if it is actually
SHOWN on the sheet (e.g. shaft depth is omitted on ~30% of shafts, load /
speed only exist when the sheet carries a lift schedule, building type only
when the title states it). For a scored field: correct value = TP; wrong
value = FP + FN; no value = FN. For a field NOT shown, a returned value is an
FP (hallucination) and abstaining is correct. Tolerances: exact for mm
dimensions and loads, +/-0.01 m/s speed, +/-0.02 m travel.

Results are broken down by subset: seen vs unseen drafting conventions, and
vector vs scanned sheets.

Usage:
    python -m src.evaluation.evaluate --data data/synthetic_v2 --split dev
    python -m src.evaluation.evaluate --data data/synthetic_v2 --split test --out results/test
"""

import argparse
import json
import sys
import time
from collections import defaultdict, Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.parser.layout_extractor import extract_layout, canonical_label  # noqa: E402

TYPES = ["passenger", "service", "firefighter", "goods"]
LIFT_FIELDS = ["car_w", "car_d", "door_w", "shaft_w", "shaft_d", "rated_load_kg", "rated_speed_ms"]
BUILDING_FIELDS = ["num_stops", "total_travel_m", "typical_floor_to_floor_mm",
                   "pit_depth_mm", "overhead_mm", "building_type"]


class PRF:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, tp=0, fp=0, fn=0):
        self.tp += tp; self.fp += fp; self.fn += fn

    @property
    def p(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def r(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self):
        return 2 * self.p * self.r / (self.p + self.r) if self.p + self.r else 0.0

    def dict(self):
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.p, 4), "recall": round(self.r, 4), "f1": round(self.f1, 4)}


def close(field, pred, gold):
    if pred is None or gold is None:
        return False
    if field == "total_travel_m":
        return abs(pred - gold) <= 0.02
    if field == "rated_speed_ms":
        return abs(pred - gold) <= 0.01
    return pred == gold


def gt_typical_applicable(g):
    lv = [l["ffl_m"] for l in g["levels"]]
    diffs = [round((b - a) * 1000 / 10) * 10 for a, b in zip(lv, lv[1:])]
    return diffs.count(g["typical_floor_to_floor_mm"]) >= 2


def score_field(bucket, field, pred, gold, shown):
    if shown:
        if pred is None:
            bucket[field].add(fn=1)
        elif close(field, pred, gold):
            bucket[field].add(tp=1)
        else:
            bucket[field].add(fp=1, fn=1)
    elif pred is not None:
        bucket[field].add(fp=1)


class Scorer:
    def __init__(self):
        self.triage = PRF()
        self.detect = PRF()
        self.fields = defaultdict(PRF)
        self.compliance = PRF()
        self.type_pairs = []   # (gold, pred) on matched lifts
        self.n = 0

    def add(self, g, p):
        self.n += 1
        is_lift = g["is_lift_drawing"]
        pred_lift = bool(p.lifts)
        self.triage.add(tp=int(is_lift and pred_lift), fp=int(pred_lift and not is_lift),
                        fn=int(is_lift and not pred_lift))
        if not is_lift:
            self.detect.add(fp=len(p.lifts))
            return
        gold = {canonical_label(l["label"]): l for l in g["lifts"]}
        pred = {canonical_label(l.label): l for l in p.lifts}
        for key, gl in gold.items():
            pl = pred.get(key)
            if pl is None:
                self.detect.add(fn=1)
                for f in LIFT_FIELDS:
                    shown = self._lift_shown(g, gl, f)
                    if shown:
                        self.fields[f].add(fn=1)
                continue
            self.detect.add(tp=1)
            self.type_pairs.append((gl["lift_type"], pl.lift_type))
            for f in LIFT_FIELDS:
                score_field(self.fields, f, getattr(pl, f), gl[f], self._lift_shown(g, gl, f))
        self.detect.add(fp=sum(1 for k in pred if k not in gold))
        bshown = {
            "num_stops": True, "total_travel_m": True, "pit_depth_mm": True, "overhead_mm": True,
            "typical_floor_to_floor_mm": gt_typical_applicable(g),
            "building_type": g.get("building_type_stated", True),
        }
        for f in BUILDING_FIELDS:
            score_field(self.fields, f, getattr(p, f), g[f], bshown[f])
        gold_flag = g["ffl_required"] and not g["ffl_present"]
        pred_flag = bool(p.ffl_required) and not p.ffl_present
        self.compliance.add(tp=int(gold_flag and pred_flag), fp=int(pred_flag and not gold_flag),
                            fn=int(gold_flag and not pred_flag))

    @staticmethod
    def _lift_shown(g, gl, f):
        if f == "shaft_d":
            return gl.get("shaft_d_shown", True)
        if f in ("rated_load_kg", "rated_speed_ms"):
            return g.get("spec_table_present", False)
        return True

    def type_report(self):
        per = {}
        for t in TYPES:
            s = PRF()
            for gold, pred in self.type_pairs:
                s.add(tp=int(gold == t and pred == t), fp=int(pred == t and gold != t),
                      fn=int(gold == t and pred != t))
            per[t] = s.dict()
        present = [t for t in TYPES if per[t]["tp"] + per[t]["fn"] > 0]
        macro = sum(per[t]["f1"] for t in present) / len(present) if present else 0.0
        acc = sum(g == p for g, p in self.type_pairs) / len(self.type_pairs) if self.type_pairs else 0.0
        conf = {g: dict(Counter(p for gg, p in self.type_pairs if gg == g)) for g in TYPES}
        return {"per_class": per, "macro_f1": round(macro, 4), "accuracy": round(acc, 4), "confusion": conf}

    def report(self):
        field_rep = {f: self.fields[f].dict() for f in LIFT_FIELDS + BUILDING_FIELDS}
        micro = PRF()
        for f in LIFT_FIELDS + BUILDING_FIELDS:
            micro.add(self.fields[f].tp, self.fields[f].fp, self.fields[f].fn)
        return {"drawings": self.n, "T1_triage": self.triage.dict(), "T2_lift_detection": self.detect.dict(),
                "T3_lift_type": self.type_report(), "T4_T5_fields": field_rep,
                "T4_T5_fields_micro": micro.dict(), "T6_ffl_compliance_flag": self.compliance.dict()}


def run(data_root, split, out_dir=None, limit=None, cache=True, vector_only=False):
    root = Path(data_root) / split
    gt = json.loads((root / "ground_truth.json").read_text())
    if vector_only:  # fast dev iteration: skip scanned sheets (OCR)
        gt = [g for g in gt if not g.get("scanned")]
    if limit:
        gt = gt[:limit]
    cache_dir = root / "ocr_cache" if cache else None
    subsets = {"all": Scorer(), "seen": Scorer(), "unseen": Scorer(), "vector": Scorer(),
               "scanned": Scorer(), "distractors_only": Scorer()}
    preds = []
    t0 = time.time()
    for i, g in enumerate(gt, 1):
        p = extract_layout(root / "drawings" / g["filename"], cache_dir)
        preds.append({"drawing_id": g["drawing_id"], **asdict(p)})
        subsets["all"].add(g, p)
        if g["is_lift_drawing"]:
            subsets["unseen" if g["style"].get("unseen") else "seen"].add(g, p)
            subsets["scanned" if g.get("scanned") else "vector"].add(g, p)
        else:
            subsets["distractors_only"].add(g, p)
        if i % 50 == 0:
            print(f"  {i}/{len(gt)}  ({time.time() - t0:.0f}s)", flush=True)
    report = {k: v.report() for k, v in subsets.items() if v.n}
    report["meta"] = {"split": split, "n": len(gt), "seconds": round(time.time() - t0, 1)}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"metrics_{split}.json").write_text(json.dumps(report, indent=1))
        (out / f"predictions_{split}.json").write_text(json.dumps(preds, indent=1, default=str))
    return report


def headline(rep):
    a = rep["all"]
    rows = [("T1 triage", a["T1_triage"]["f1"]), ("T2 lift detection", a["T2_lift_detection"]["f1"]),
            ("T3 lift type (macro)", a["T3_lift_type"]["macro_f1"]),
            ("T4/T5 fields (micro)", a["T4_T5_fields_micro"]["f1"]),
            ("T6 compliance flag", a["T6_ffl_compliance_flag"]["f1"])]
    return "\n".join(f"  {n:<24} F1 = {v:.3f}" for n, v in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_v2")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--vector-only", action="store_true")
    a = ap.parse_args()
    rep = run(a.data, a.split, a.out, a.limit, vector_only=a.vector_only)
    print(headline(rep))
    for sub in ("seen", "unseen", "vector", "scanned"):
        if sub in rep:
            r = rep[sub]
            print(f"  [{sub:<7}] detect F1={r['T2_lift_detection']['f1']:.3f}  "
                  f"type F1={r['T3_lift_type']['macro_f1']:.3f}  fields F1={r['T4_T5_fields_micro']['f1']:.3f}")


if __name__ == "__main__":
    main()
