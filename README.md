# LiftPlan AI

Automated elevator specification from architectural drawings - MSc Individual Project (CST4275),
Middlesex University Dubai. Shamshul Huda Faruqui (M00908731). Supervisor: Dr. Siddhaling Urolagin.

No real project drawings, specifications or client data are included in, or used by, this code.
All data is synthetic and regenerated from fixed seeds.

## Setup

    python -m venv venv
    venv\Scripts\activate            (Windows)   |   source venv/bin/activate   (macOS/Linux)
    pip install -r requirements.txt
    # Tesseract OCR must be installed and on PATH (used for sheets without a text layer)

## Run the dashboard

    streamlit run src/dashboard/app.py

## Regenerate the dataset (v2, ~160 MB per split, resumable)

    python -m src.synthetic.generate_realistic --split dev  --count 500 --distractors 100
    python -m src.synthetic.generate_realistic --split test --count 500 --distractors 100

Output goes to data/synthetic_v2/<split>/ (drawings, YOLO labels, ground_truth.json).
Each sheet has its own seed, so an interrupted run can simply be re-run to continue.

## Evaluate (precision / recall / F1)

    python -m src.evaluation.evaluate --split test --out results/test_full
    python -m src.evaluation.evaluate --split dev  --out results/dev --vector-only   # fast, no OCR

The first full run OCRs the ~96 scanned test sheets (~20 s each); results are cached.

## Train and evaluate the YOLOv8 detector (needs ultralytics + torch)

    pip install ultralytics
    python -m src.detector.yolo_detector prepare
    python -m src.detector.yolo_detector train --epochs 40            # CPU: try --imgsz 960 --epochs 25
    python -m src.detector.yolo_detector evaluate --weights runs/detect/liftplan/weights/best.pt

Metrics are written to results/yolo/yolo_test_metrics.json (Table 6.7 of the dissertation).
If training stops with an out-of-memory error in a dataloader worker, lower `--workers`
(default 2; `--workers 0` uses no extra processes) on both `train` and `evaluate`.

## Tests

    python -m pytest -q          # 321 tests

## Layout

    src/synthetic/generate_realistic.py   v2 dataset generator (lift-detail sheets + ground truth + YOLO labels)
    src/parser/layout_extractor.py        layout-aware extractor (labels, dimensions, levels, schedule)
    src/evaluation/evaluate.py            evaluation harness
    src/detector/yolo_detector.py         YOLOv8 prepare / train / evaluate
    src/traffic_engine/traffic_study.py   CIBSE Guide D traffic study (validated, see tests/test_traffic_worked_example.py)
    src/spec_engine/recommender.py        specification recommender
    src/dashboard/                        Streamlit dashboard and pipeline orchestration
    results/                              test-split metrics, traffic-fix impact, pipeline comparison
