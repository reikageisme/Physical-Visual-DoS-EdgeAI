"""Measure representative edge-style batch-1 latency with repeatable validation."""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from pipeline_utils import configure_project_environment, materialize_ultralytics_dataset_yaml, project_path, require_local_venv

configure_project_environment()

REPETITIONS = 3
PLATEAU_TOLERANCE = 0.005


def num(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def save_records(records, path):
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = list(dict.fromkeys(key for row in records for key in row))
    with path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def main():
    require_local_venv()
    import torch
    import ultralytics
    from ultralytics import YOLO

    out_dir = project_path("results/time_budget")
    records_path = out_dir / "milestone_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    data_yaml = project_path("dataset/adv_tshirt_patch_detection.yaml", must_exist=True)
    runtime_yaml = materialize_ultralytics_dataset_yaml(data_yaml)
    eval_dir = out_dir / "latency_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for model_key in ("yolo11n", "yolov13n"):
        model_rows = [r for r in records if r.get("model_key") == model_key and r.get("status") == "complete"]
        history_path = project_path(model_rows[0]["history_csv"], must_exist=True)
        with history_path.open("r", encoding="utf-8-sig", newline="") as source:
            history = list(csv.DictReader(source))
        scores = [num(row.get("metrics/mAP50-95(B)")) for row in history]
        best = max(score for score in scores if score is not None)
        first_near_best = next(row for row in history if num(row.get("metrics/mAP50-95(B)")) is not None and num(row.get("metrics/mAP50-95(B)")) >= best - PLATEAU_TOLERANCE)
        max_epoch = int(float(first_near_best["epoch"]))
        eligible = [r for r in model_rows if int(r["nominal_budget_minutes"]) >= int(float(first_near_best["time"]) / 60) and int(r["selected_epoch"]) >= max_epoch]
        chosen = min(eligible, key=lambda r: int(r["nominal_budget_minutes"])) if eligible else max(model_rows, key=lambda r: int(r["nominal_budget_minutes"]))
        checkpoint = project_path(chosen["checkpoint"], must_exist=True)
        model = YOLO(str(checkpoint))
        repetitions = []
        for repetition in range(1, REPETITIONS + 1):
            metrics = model.val(
                data=str(runtime_yaml), split="test", imgsz=416, batch=1, workers=0,
                device="0", plots=False, project=str(eval_dir),
                name=f"latency-{model_key}-sweetspot-r{repetition}", exist_ok=False, verbose=False,
            )
            speed = {key: num(value) for key, value in (metrics.speed or {}).items()}
            total = sum(speed.get(key, 0.0) or 0.0 for key in ("preprocess", "inference", "postprocess"))
            repetitions.append({"repetition": repetition, "speed_ms": speed, "total_latency_ms": total})
        components = {}
        for component in ("preprocess", "inference", "postprocess"):
            values = [rep["speed_ms"].get(component) for rep in repetitions]
            values = [value for value in values if value is not None]
            components[component] = statistics.median(values) if values else None
        median_latency = sum(value or 0.0 for value in components.values())
        result = {
            "model_key": model_key,
            "model_name": chosen.get("model_name"),
            "checkpoint": chosen["checkpoint"],
            "checkpoint_epoch": chosen["selected_epoch"],
            "nominal_milestone_minutes": chosen["nominal_budget_minutes"],
            "batch_size": 1,
            "repetitions": REPETITIONS,
            "median_speed_ms": components,
            "latency_ms_per_frame": median_latency,
            "fps": 1000.0 / median_latency if median_latency > 0 else None,
            "raw_repetitions": repetitions,
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)
        for row in model_rows:
            row["latency_benchmark"] = result
            row["test"]["validator_speed_batch4_ms"] = row["test"].get("speed_ms")
            row["test"]["speed_ms"] = components
            row["test"]["latency_ms_per_frame"] = median_latency
            row["test"]["fps"] = result["fps"]
        print(f"{model_key}: latency={median_latency:.3f} ms/frame; FPS={result['fps']:.1f}; checkpoint={chosen['checkpoint']}")

    measurement = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_environment": ".venv",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "dataset_split": "test",
        "batch_size": 1,
        "repetitions_per_model": REPETITIONS,
        "latency_definition": "median preprocess + inference + postprocess per image; FPS=1000/latency",
        "models": results,
    }
    save_records(records, records_path)
    save_records([measurement], out_dir / "latency_measurements.json")


if __name__ == "__main__":
    main()
