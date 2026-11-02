"""Measure batch-1 preprocessing, inference and postprocessing at sweet spots."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, project_path, require_local_venv

configure_project_environment()

MODELS = {
    "yolo11n": {"label": "YOLO11n", "source": "yolov13_src"},
    "yolov12n": {"label": "YOLOv12n", "source": "yolov12_src"},
    "yolov13n": {"label": "YOLOv13n", "source": "yolov13_src"},
    "yolov13s": {"label": "YOLOv13s", "source": "yolov13_src"},
}
SEEDS = (42, 100, 2026, 777, 999)
BUDGETS = (5, 10, 20, 30, 60, 120)
IMAGE_SIZE = 416
WARMUP_FRAMES = 32
MEASURED_FRAMES = 500
OUTPUT = project_path("results/fullscale_experiments/latency_records.json")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(value: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT)


def select_sweet_spots() -> dict[str, dict]:
    records = {}
    for model in MODELS:
        seeds = {}
        for seed in SEEDS:
            path = project_path(f"results/fullscale_experiments/records/{model}/seed-{seed}.json", must_exist=True)
            record = load_json(path)
            if record.get("status") != "complete":
                raise RuntimeError(f"Training record is not complete: {path}")
            seeds[seed] = record
        means = {}
        for budget in BUDGETS:
            values = []
            for record in seeds.values():
                row = next((m for m in record["milestones"] if m["budget_minutes"] == budget), None)
                if row and row.get("status") == "complete" and row.get("validation", {}).get("map50_95") is not None:
                    values.append(row["validation"]["map50_95"])
            if len(values) == len(SEEDS):
                means[budget] = sum(values) / len(values)
        if not means:
            raise RuntimeError(f"No complete validation milestones for {model}")
        peak = max(means.values())
        sweet_budget = next(b for b in BUDGETS if b in means and peak - means[b] <= 0.005)
        records[model] = {
            "label": MODELS[model]["label"],
            "mean_validation_map50_95_by_budget": {str(k): v for k, v in means.items()},
            "peak_validation_map50_95": peak,
            "sweet_spot_minutes": sweet_budget,
        }
    return records


def benchmark_one(model_key: str, seed: int, budget: int) -> dict:
    require_local_venv()
    spec = MODELS[model_key]
    training_record = load_json(project_path(f"results/fullscale_experiments/records/{model_key}/seed-{seed}.json", must_exist=True))
    if training_record.get("status") != "complete":
        raise RuntimeError("Cannot benchmark latency before the training trajectory completes")
    milestone = next((row for row in training_record["milestones"] if row["budget_minutes"] == budget), None)
    if not milestone or milestone.get("status") != "complete" or not milestone.get("checkpoint"):
        raise RuntimeError(f"Missing selected checkpoint for {model_key} seed {seed} at {budget}m")
    checkpoint = project_path(milestone["checkpoint"], must_exist=True)

    import numpy as np

    sys.path.insert(0, str(project_path(spec["source"], must_exist=True)))
    from ultralytics import YOLO
    import ultralytics

    manifest = load_json(project_path("dataset/fullscale_person/manifest.json", must_exist=True))
    test_images = sorted(
        str(project_path(row["image"], must_exist=True))
        for row in manifest["split_manifest"]["test"]
    )
    rng = random.Random(2026)
    sample_size = min(WARMUP_FRAMES + MEASURED_FRAMES, len(test_images))
    selected_images = rng.sample(test_images, sample_size)
    warmup = min(WARMUP_FRAMES, max(0, sample_size - 1))

    model = YOLO(str(checkpoint))
    stream = model.predict(
        source=selected_images,
        imgsz=IMAGE_SIZE,
        batch=1,
        device="0",
        half=False,
        stream=True,
        verbose=False,
        save=False,
        plots=False,
    )
    per_frame = {"preprocess": [], "inference": [], "postprocess": []}
    seen = 0
    for result in stream:
        if seen >= warmup:
            speed = result.speed or {}
            for key in per_frame:
                per_frame[key].append(float(speed.get(key, 0.0)))
        seen += 1
    if seen != sample_size or not per_frame["inference"]:
        raise RuntimeError(f"Latency prediction yielded {seen} frames; expected {sample_size}")
    stages = {key: float(np.median(values)) for key, values in per_frame.items() if values}
    totals = [sum(per_frame[key][idx] for key in per_frame) for idx in range(len(per_frame["inference"]))]
    latency_ms = float(np.median(totals))
    return {
        "model_key": model_key,
        "model_name": spec["label"],
        "seed": seed,
        "sweet_spot_minutes": budget,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "imgsz": IMAGE_SIZE,
        "batch": 1,
        "timing_precision": "fp32",
        "device": "NVIDIA GeForce RTX 5060",
        "warmup_frames": warmup,
        "measured_frames": seen - warmup,
        "stage_median_ms_per_frame": stages,
        "latency_median_ms_per_frame": latency_ms,
        "fps_from_median_latency": 1000 / latency_ms if latency_ms > 0 else None,
        "timing_scope": "Ultralytics preprocessing + inference + postprocessing; excludes image loading/decoding and application I/O",
        "ultralytics_version": ultralytics.__version__,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--budget-minutes", type=int, choices=BUDGETS)
    parser.add_argument("--all", action="store_true", help="Select model sweet spots, then benchmark every model/seed")
    args = parser.parse_args()
    require_local_venv()

    if args.all:
        spots = select_sweet_spots()
        data = {"study_id": "fullscale-20261011", "sweet_spots": spots, "records": []}
        if OUTPUT.exists():
            previous = load_json(OUTPUT)
            data["records"] = [r for r in previous.get("records", []) if r.get("model_key") in spots and r.get("sweet_spot_minutes") == spots[r["model_key"]]["sweet_spot_minutes"] and r.get("timing_precision") == "fp32"]
        for model_key, spec in MODELS.items():
            budget = spots[model_key]["sweet_spot_minutes"]
            for seed in SEEDS:
                if any(r["model_key"] == model_key and r["seed"] == seed for r in data["records"]):
                    continue
                print(f"Latency benchmark: {spec['label']} seed={seed} at {budget}m", flush=True)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--model", model_key,
                    "--seed", str(seed),
                    "--budget-minutes", str(budget),
                ]
                environment = os.environ.copy()
                result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, check=False)
                if result.returncode:
                    return result.returncode
                # The child writes its record atomically to a single-result file.
                one_path = project_path("results/fullscale_experiments/latency_one.json", must_exist=True)
                data["records"].append(load_json(one_path))
                save_json(data)
        save_json(data)
        print(f"Latency records saved: {OUTPUT.relative_to(PROJECT_ROOT)}", flush=True)
        return 0

    if not args.model or args.seed is None or args.budget_minutes is None:
        parser.error("provide --all, or all of --model, --seed, and --budget-minutes")
    result = benchmark_one(args.model, args.seed, args.budget_minutes)
    one_path = project_path("results/fullscale_experiments/latency_one.json")
    one_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
