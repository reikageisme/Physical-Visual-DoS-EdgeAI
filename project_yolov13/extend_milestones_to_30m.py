"""Add a 30-minute checkpoint evaluation to completed fixed-schedule runs."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from pipeline_utils import configure_project_environment, materialize_ultralytics_dataset_yaml, project_path, require_local_venv

configure_project_environment()


def num(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def loss(row, prefix):
    vals = {key: num(row.get(f"{prefix}/{key}_loss")) for key in ("box", "cls", "dfl")}
    vals["total"] = sum(vals.values()) if all(v is not None for v in vals.values()) else None
    return vals


def write_records(records, out_dir):
    (out_dir / "milestone_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = list(dict.fromkeys(key for row in records for key in row))
    with (out_dir / "milestone_records.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def main():
    require_local_venv()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    out_dir = project_path("results/time_budget")
    path = out_dir / "milestone_records.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    dataset_yaml = project_path("dataset/adv_tshirt_patch_detection.yaml", must_exist=True)
    runtime_yaml = materialize_ultralytics_dataset_yaml(dataset_yaml)
    eval_dir = out_dir / "eval_milestones"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for model_key in ("yolo11n", "yolov13n"):
        if any(r.get("model_key") == model_key and r.get("nominal_budget_minutes") == 30 and r.get("status") == "complete" for r in records):
            continue
        prior = next(r for r in records if r.get("model_key") == model_key and r.get("status") == "complete")
        history_path = project_path(prior["history_csv"], must_exist=True)
        with history_path.open("r", encoding="utf-8-sig", newline="") as source:
            history = list(csv.DictReader(source))
        available = [r for r in history if (num(r.get("time")) or 0) <= 30 * 60]
        if not available:
            available = [history[0]]
        endpoint = available[-1]
        selected = max(available, key=lambda r: num(r.get("metrics/mAP50-95(B)")) if num(r.get("metrics/mAP50-95(B)")) is not None else -1)
        epoch = int(float(selected["epoch"]))
        checkpoint = history_path.parent / "weights" / f"epoch{epoch - 1}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        relative_checkpoint = str(checkpoint.relative_to(project_path(".")))
        cached = next((r for r in records if r.get("model_key") == model_key and r.get("checkpoint") == relative_checkpoint), None)
        if cached:
            test_metrics = cached["test"]
            parameters_m = cached["parameters_m"]
            flops_g = cached["flops_g"]
        else:
            weights_rel = "weights/yolo11n.pt" if model_key == "yolo11n" else "weights/yolov13n.pt"
            model = YOLO(str(checkpoint))
            parameters_m = sum(parameter.numel() for parameter in model.model.parameters()) / 1_000_000
            flops_g = get_flops(model.model, imgsz=416)
            metrics = model.val(
                data=str(runtime_yaml), split="test", imgsz=416, batch=4, workers=0, device="0",
                plots=False, project=str(eval_dir), name=f"milestones-20261010-{model_key}-30m-test",
                exist_ok=False, verbose=False,
            )
            speed = {key: num(value) for key, value in (metrics.speed or {}).items()}
            latency = sum(speed.get(key, 0.0) or 0.0 for key in ("preprocess", "inference", "postprocess"))
            test_metrics = {
                "precision": num(metrics.box.mp), "recall": num(metrics.box.mr),
                "map50": num(metrics.box.map50), "map50_95": num(metrics.box.map),
                "speed_ms": speed, "latency_ms_per_frame": latency or None,
                "fps": 1000.0 / latency if latency > 0 else None,
            }
        row = {
            "study_id": prior.get("study_id"), "model_key": model_key, "model_name": prior.get("model_name"),
            "status": "complete", "run_name": prior["run_name"], "started_at_utc": prior.get("started_at_utc"),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(), "nominal_budget_minutes": 30,
            "time_budget_hours_initial_grid_replaced": True,
            "training_wall_seconds": prior.get("training_wall_seconds"),
            "training_actual_seconds": num(history[-1].get("time")), "training_epochs_completed": len(history),
            "training_stop_reason": prior.get("training_stop_reason"),
            "epoch_endpoint": int(float(endpoint["epoch"])), "endpoint_elapsed_seconds": num(endpoint.get("time")),
            "endpoint_train_loss": loss(endpoint, "train"), "endpoint_val_loss": loss(endpoint, "val"),
            "selected_epoch": epoch, "selected_elapsed_seconds": num(selected.get("time")),
            "selected_val": {
                "precision": num(selected.get("metrics/precision(B)")), "recall": num(selected.get("metrics/recall(B)")),
                "map50": num(selected.get("metrics/mAP50(B)")), "map50_95": num(selected.get("metrics/mAP50-95(B)")),
            },
            "test": test_metrics, "parameters_m": parameters_m, "flops_g": flops_g,
            "checkpoint": relative_checkpoint, "log": prior.get("log"), "history_csv": prior.get("history_csv"),
            "dataset_yaml": prior.get("dataset_yaml"),
        }
        records.append(row)
        print(f"Added {model_key} @ 30m: epoch={epoch}, t={row['selected_elapsed_seconds']:.1f}s, val mAP95={row['selected_val']['map50_95']:.5f}")

    records.sort(key=lambda r: (r.get("model_key", ""), int(r.get("nominal_budget_minutes", 0))))
    write_records(records, out_dir)
    env_path = out_dir / "milestone_environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8"))
    env["time_milestones_minutes"] = [5, 10, 15, 20, 25, 30]
    env["milestone_extension_reason"] = "YOLO11n fixed-schedule pilot curve continued improving at 24.7 minutes; extended upper bound after observing the main run."
    env_path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
