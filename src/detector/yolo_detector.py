"""
Learned detector (YOLOv8) for lift-drawing elements
====================================================

Trains a deep-learning object detector to find, on a rendered drawing
sheet:  0 = shaft, 1 = car, 2 = level_marker.

This is the project's trained deep-learning component. Labels come for free
from the synthetic generator (exact bounding boxes of every drawn element),
so the dev split trains the model and the held-out test split - including
drafting conventions never seen in dev, and scanned sheets - evaluates it.

Three sub-commands:

  prepare   render every lift drawing to a PNG and write YOLO-format labels
            (dev -> train/val 90/10 split; test -> test)
  train     fine-tune a pretrained YOLOv8n on train, validating on val
  evaluate  score the trained model on the held-out TEST split: precision,
            recall, F1 and mAP@0.5 per class, plus a count-level check
            (number of cars detected == number of lifts on the sheet)

Usage (from the project root, in the venv that has ultralytics + torch):
    python -m src.detector.yolo_detector prepare
    python -m src.detector.yolo_detector train --epochs 40
    python -m src.detector.yolo_detector evaluate --weights runs/detect/liftplan/weights/best.pt

CPU-only machines: training at --imgsz 1280 for 40 epochs on ~450 images
takes a few hours; --imgsz 960 --epochs 25 is a reasonable faster setting.
Report whichever setting was actually used.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import pymupdf as fitz

CLASSES = ["shaft", "car", "level_marker"]
DATA = Path("data/synthetic_v2")
YOLO_ROOT = Path("data/yolo")
RENDER_WIDTH_PX = 1600


def render_png(pdf_path: Path, out_png: Path):
    page = fitz.open(str(pdf_path))[0]
    zoom = RENDER_WIDTH_PX / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
    pix.save(str(out_png))


def prepare(val_fraction=0.1, seed=0):
    if YOLO_ROOT.exists():
        shutil.rmtree(YOLO_ROOT)
    rng = random.Random(seed)
    for split in ("dev", "test"):
        gt = json.loads((DATA / split / "ground_truth.json").read_text())
        lift = [g for g in gt if g["is_lift_drawing"]]
        if split == "dev":
            rng.shuffle(lift)
            n_val = int(len(lift) * val_fraction)
            parts = {"val": lift[:n_val], "train": lift[n_val:]}
        else:
            parts = {"test": lift}
        for part, items in parts.items():
            (YOLO_ROOT / "images" / part).mkdir(parents=True, exist_ok=True)
            (YOLO_ROOT / "labels" / part).mkdir(parents=True, exist_ok=True)
            for g in items:
                stem = Path(g["filename"]).stem
                render_png(DATA / split / "drawings" / g["filename"], YOLO_ROOT / "images" / part / f"{stem}.png")
                shutil.copy(DATA / split / "labels" / f"{stem}.txt", YOLO_ROOT / "labels" / part / f"{stem}.txt")
            print(f"{part}: {len(items)} images")
    (YOLO_ROOT / "data.yaml").write_text(
        f"path: {YOLO_ROOT.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"wrote {YOLO_ROOT / 'data.yaml'}")


def train(epochs, imgsz, model_name, workers=2):
    # Absolute project path: newer ultralytics versions nest a relative
    # project under their own runs/detect folder (runs/detect/runs/detect/...),
    # which would break the --weights path used by the evaluate step.
    from ultralytics import YOLO
    model = YOLO(model_name)   # COCO-pretrained weights, fine-tuned on drawings
    model.train(data=str(YOLO_ROOT / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=4, workers=workers,
                project=str(Path("runs/detect").resolve()), name="liftplan", exist_ok=True,
                fliplr=0.0, mosaic=0.5, degrees=1.0, seed=0, deterministic=True, plots=True)


def evaluate(weights, imgsz, out_dir, workers=2):
    from ultralytics import YOLO
    model = YOLO(weights)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = model.val(data=str(YOLO_ROOT / "data.yaml"), split="test", imgsz=imgsz, batch=4, workers=workers,
                  project=str(out.resolve()), name="yolo_test", exist_ok=True, plots=True)
    per_class = {}
    for i, c in enumerate(CLASSES):
        p, r, ap50, ap = m.box.class_result(i)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[c] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                        "f1": round(float(f1), 4), "mAP50": round(float(ap50), 4),
                        "mAP50_95": round(float(ap), 4)}
    # count-level check on the test split: cars detected vs lifts on the sheet,
    # broken down by scanned / vector and seen / unseen conventions
    gt = {Path(g["filename"]).stem: g for g in
          json.loads((DATA / "test" / "ground_truth.json").read_text()) if g["is_lift_drawing"]}
    counts = {"all": [0, 0], "scanned": [0, 0], "vector": [0, 0], "seen": [0, 0], "unseen": [0, 0]}
    for res in model.predict(source=str(YOLO_ROOT / "images" / "test"), imgsz=imgsz, conf=0.25,
                             stream=True, verbose=False):
        stem = Path(res.path).stem
        g = gt[stem]
        n_cars = int((res.boxes.cls == CLASSES.index("car")).sum())
        ok = int(n_cars == len(g["lifts"]))
        for key in ("all", "scanned" if g["scanned"] else "vector",
                    "unseen" if g["style"].get("unseen") else "seen"):
            counts[key][0] += ok
            counts[key][1] += 1
    report = {"per_class": per_class,
              "car_count_accuracy": {k: round(a / n, 4) if n else None for k, (a, n) in counts.items()},
              "imgsz": imgsz, "weights": str(weights)}
    (out / "yolo_test_metrics.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--model", default="yolov8n.pt")
    # Data-loading worker processes. Each is a full copy of Python + torch;
    # ultralytics defaults to 8, which exhausted RAM on a Windows laptop
    # (cv2 "Insufficient memory" in a worker). 2 keeps a 4 GB GPU busy.
    t.add_argument("--workers", type=int, default=2)
    e = sub.add_parser("evaluate")
    e.add_argument("--weights", required=True)
    e.add_argument("--imgsz", type=int, default=1280)
    e.add_argument("--out", default="results/yolo")
    e.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "train":
        train(a.epochs, a.imgsz, a.model, a.workers)
    else:
        evaluate(a.weights, a.imgsz, a.out, a.workers)


if __name__ == "__main__":
    main()
