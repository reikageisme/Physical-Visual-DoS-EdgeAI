"""Train one fixed-schedule trajectory per model and evaluate time milestones."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from pipeline_utils import (
    PROJECT_ROOT,
    configure_project_environment,
    materialize_ultralytics_dataset_yaml,
    project_path,
    require_local_venv,
    validate_dataset_yaml,
)

configure_project_environment()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


STUDY_ID = "20261010"
TIME_MILESTONES_MINUTES = (5, 10, 15, 20, 25, 30)
MODEL_SPECS = {
    "yolo11n": ("YOLO11n (Baseline)", "weights/yolo11n.pt"),
    "yolov13n": ("YOLOv13n", "weights/yolov13n.pt"),
}
IMAGE_SIZE = 416
BATCH_SIZE = 4
MAX_EPOCHS = 50
PATIENCE = 10


def as_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_history(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def loss_summary(row, prefix):
    if not row:
        return {key: None for key in ("box", "cls", "dfl", "total")}
    vals = {key: as_float(row.get(f"{prefix}/{key}_loss")) for key in ("box", "cls", "dfl")}
    vals["total"] = sum(vals.values()) if all(value is not None for value in vals.values()) else None
    return vals


def metric_summary(metrics):
    box = metrics.box
    speed = {key: as_float(value) for key, value in (metrics.speed or {}).items()}
    latency = sum(speed.get(key, 0.0) or 0.0 for key in ("preprocess", "inference", "postprocess"))
    return {
        "precision": as_float(box.mp),
        "recall": as_float(box.mr),
        "map50": as_float(box.map50),
        "map50_95": as_float(box.map),
        "speed_ms": speed,
        "latency_ms_per_frame": latency or None,
        "fps": 1000.0 / latency if latency > 0 else None,
    }


def train_arguments(data_yaml: Path, run_name: str):
    return {
        "data": str(data_yaml),
        "epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch": BATCH_SIZE,
        "imgsz": IMAGE_SIZE,
        "save": True,
        "save_period": 1,
        "cache": False,
        "device": "0",
        "workers": 0,
        "project": str(project_path("runs/train")),
        "name": run_name,
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "SGD",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "cos_lr": False,
        "close_mosaic": 5,
        "resume": False,
        "amp": True,
        "val": True,
        "plots": True,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 2.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "auto_augment": None,
        "erasing": 0.0,
        "fraction": 1.0,
    }


def checkpoint_for_epoch(run_dir: Path, csv_epoch: int) -> Path:
    # Ultralytics writes zero-based epochN.pt files while results.csv is one-based.
    return run_dir / "weights" / f"epoch{csv_epoch - 1}.pt"


def save_records(rows, out_dir):
    (out_dir / "milestone_records.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (out_dir / "milestone_records.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def main() -> int:
    require_local_venv()
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    dataset_yaml = project_path("dataset/adv_tshirt_patch_detection.yaml", must_exist=True)
    validate_dataset_yaml(dataset_yaml, require_test=True)
    runtime_yaml = materialize_ultralytics_dataset_yaml(dataset_yaml)
    for _, weight in MODEL_SPECS.values():
        project_path(weight, must_exist=True)

    out_dir = project_path("results/time_budget")
    logs_dir = out_dir / "logs"
    eval_dir = out_dir / "eval_milestones"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    hardware = {
        "study_id": STUDY_ID,
        "python": sys.version,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None,
        "dataset_yaml": str(dataset_yaml.relative_to(PROJECT_ROOT)),
        "seed": 42,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_max": MAX_EPOCHS,
        "patience": PATIENCE,
        "time_milestones_minutes": list(TIME_MILESTONES_MINUTES),
        "protocol": "one fixed-schedule run per model; save every epoch; select the best validation checkpoint available by each elapsed-time milestone",
    }
    (out_dir / "milestone_environment.json").write_text(
        json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records = []
    for model_key, (model_label, weight_rel) in MODEL_SPECS.items():
        run_name = f"milestones-{STUDY_ID}-{model_key}-fixed50"
        run_dir = project_path(Path("runs/train") / run_name)
        log_path = logs_dir / f"{run_name}.log"
        started_at = datetime.now(timezone.utc).isoformat()
        print(f"\n=== {run_name} ===\ncap snapshots={TIME_MILESTONES_MINUTES} minutes; epochs={MAX_EPOCHS}; patience={PATIENCE}")
        print(f"Logs: {log_path.relative_to(PROJECT_ROOT)}")
        started = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                tee = Tee(sys.stdout, log)
                with redirect_stdout(tee), redirect_stderr(tee):
                    print(f"run={run_name} start={started_at} model={model_label}")
                    print(f"weight={project_path(weight_rel)} dataset={dataset_yaml}")
                    model = YOLO(str(project_path(weight_rel)))
                    model.train(**train_arguments(runtime_yaml, run_name))

                    history_path = run_dir / "results.csv"
                    history = read_history(history_path)
                    if not history:
                        raise RuntimeError(f"No epoch history was produced: {history_path}")
                    last_time = as_float(history[-1].get("time")) or 0.0
                    if len(history) < MAX_EPOCHS:
                        stop_reason = "patience early stopping"
                    else:
                        stop_reason = "50-epoch maximum"

                    milestone_cache = {}
                    for budget_minutes in TIME_MILESTONES_MINUTES:
                        available = [r for r in history if (as_float(r.get("time")) or 0.0) <= budget_minutes * 60]
                        if not available:
                            available = [history[0]]
                        endpoint = available[-1]
                        selected = max(
                            available,
                            key=lambda r: as_float(r.get("metrics/mAP50-95(B)")) if as_float(r.get("metrics/mAP50-95(B)")) is not None else -1.0,
                        )
                        selected_epoch = int(float(selected["epoch"]))
                        checkpoint = checkpoint_for_epoch(run_dir, selected_epoch)
                        if not checkpoint.exists():
                            raise FileNotFoundError(f"Expected epoch checkpoint was not saved: {checkpoint}")

                        checkpoint_key = str(checkpoint.relative_to(PROJECT_ROOT))
                        if checkpoint_key not in milestone_cache:
                            trained = YOLO(str(checkpoint))
                            parameters_m = sum(parameter.numel() for parameter in trained.model.parameters()) / 1_000_000
                            flops_g = get_flops(trained.model, imgsz=IMAGE_SIZE)
                            eval_name = f"{run_name}-{budget_minutes}m-test"
                            metrics = trained.val(
                                data=str(runtime_yaml),
                                split="test",
                                imgsz=IMAGE_SIZE,
                                batch=BATCH_SIZE,
                                workers=0,
                                device="0",
                                plots=False,
                                project=str(eval_dir),
                                name=eval_name,
                                exist_ok=False,
                                verbose=False,
                            )
                            milestone_cache[checkpoint_key] = {
                                "test": metric_summary(metrics),
                                "parameters_m": parameters_m,
                                "flops_g": flops_g,
                            }
                        cached = milestone_cache[checkpoint_key]
                        record = {
                            "study_id": STUDY_ID,
                            "model_key": model_key,
                            "model_name": model_label,
                            "status": "complete",
                            "run_name": run_name,
                            "started_at_utc": started_at,
                            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                            "nominal_budget_minutes": budget_minutes,
                            "time_budget_hours_initial_grid_replaced": True,
                            "training_wall_seconds": round(time.perf_counter() - started, 3),
                            "training_actual_seconds": last_time,
                            "training_epochs_completed": len(history),
                            "training_stop_reason": stop_reason,
                            "epoch_endpoint": int(float(endpoint["epoch"])),
                            "endpoint_elapsed_seconds": as_float(endpoint.get("time")),
                            "endpoint_train_loss": loss_summary(endpoint, "train"),
                            "endpoint_val_loss": loss_summary(endpoint, "val"),
                            "selected_epoch": selected_epoch,
                            "selected_elapsed_seconds": as_float(selected.get("time")),
                            "selected_val": {
                                "precision": as_float(selected.get("metrics/precision(B)")),
                                "recall": as_float(selected.get("metrics/recall(B)")),
                                "map50": as_float(selected.get("metrics/mAP50(B)")),
                                "map50_95": as_float(selected.get("metrics/mAP50-95(B)")),
                            },
                            "test": cached["test"],
                            "parameters_m": cached["parameters_m"],
                            "flops_g": cached["flops_g"],
                            "checkpoint": checkpoint_key,
                            "log": str(log_path.relative_to(PROJECT_ROOT)),
                            "history_csv": str(history_path.relative_to(PROJECT_ROOT)),
                            "dataset_yaml": str(dataset_yaml.relative_to(PROJECT_ROOT)),
                        }
                        records.append(record)
                        save_records(records, out_dir)
                        print(
                            f"MILESTONE_SUMMARY={json.dumps(record, ensure_ascii=False, allow_nan=False)}"
                        )
                    print(f"TRAINING_SUMMARY model={model_key} epochs={len(history)} train_seconds={last_time:.3f} stop={stop_reason}")
        except Exception as exc:
            failure = {
                "study_id": STUDY_ID,
                "model_key": model_key,
                "model_name": model_label,
                "status": "failed",
                "run_name": run_name,
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "log": str(log_path.relative_to(PROJECT_ROOT)),
            }
            records.append(failure)
            save_records(records, out_dir)
            print(f"FAILED {model_label}: {failure['error']}", file=sys.stderr)

    return 0 if all(row.get("status") == "complete" for row in records) and len(records) == len(MODEL_SPECS) * len(TIME_MILESTONES_MINUTES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
