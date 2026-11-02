"""Run short, matched pilots to place useful training-time milestones."""

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


PILOT_ID = "20261010"
PILOT_MINUTES = 10
MAX_EPOCHS = 50
PATIENCE = 10
IMAGE_SIZE = 416
BATCH_SIZE = 4
MODELS = {
    "yolo11n": ("YOLO11n", "weights/yolo11n.pt"),
    "yolov13n": ("YOLOv13n", "weights/yolov13n.pt"),
}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_history(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        return rows, None, None
    best = max(rows, key=lambda row: number(row.get("metrics/mAP50-95(B)")) or -1)
    return rows, rows[-1], best


def loss_summary(row, prefix):
    if not row:
        return None
    losses = {name: number(row.get(f"{prefix}/{name}_loss")) for name in ("box", "cls", "dfl")}
    losses["total"] = sum(losses.values()) if all(value is not None for value in losses.values()) else None
    return losses


def train_arguments(data_yaml: Path, run_name: str):
    return {
        "data": str(data_yaml),
        "epochs": MAX_EPOCHS,
        "time": PILOT_MINUTES / 60.0,
        "patience": PATIENCE,
        "batch": BATCH_SIZE,
        "imgsz": IMAGE_SIZE,
        "save": True,
        "save_period": -1,
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


def main() -> int:
    require_local_venv()
    import torch
    from ultralytics import YOLO

    data_yaml = project_path("dataset/adv_tshirt_patch_detection.yaml", must_exist=True)
    validate_dataset_yaml(data_yaml, require_test=True)
    runtime_yaml = materialize_ultralytics_dataset_yaml(data_yaml)
    for _, weight in MODELS.values():
        project_path(weight, must_exist=True)

    out_dir = project_path("results/time_budget/pilot")
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    hardware = {
        "pilot_id": PILOT_ID,
        "cap_minutes": PILOT_MINUTES,
        "epochs_max": MAX_EPOCHS,
        "patience": PATIENCE,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_yaml": str(data_yaml.relative_to(PROJECT_ROOT)),
        "seed": 42,
    }
    (out_dir / "environment.json").write_text(json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8")

    for model_key, (model_name, weight_rel) in MODELS.items():
        run_name = f"milestone-pilot-{PILOT_ID}-{model_key}-10m"
        run_dir = project_path(Path("runs/train") / run_name)
        log_path = logs_dir / f"{run_name}.log"
        record = {
            "pilot_id": PILOT_ID,
            "model_key": model_key,
            "model_name": model_name,
            "budget_minutes": PILOT_MINUTES,
            "status": "running",
            "run_name": run_name,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str((run_dir / "weights/best.pt").relative_to(PROJECT_ROOT)),
            "log": str(log_path.relative_to(PROJECT_ROOT)),
            "history_csv": str((run_dir / "results.csv").relative_to(PROJECT_ROOT)),
        }
        started = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                tee = Tee(sys.stdout, log)
                with redirect_stdout(tee), redirect_stderr(tee):
                    print(f"=== {run_name} | epochs={MAX_EPOCHS} cap={PILOT_MINUTES}m | {record['started_at_utc']} ===")
                    print(f"weight={project_path(weight_rel)} data={data_yaml}")
                    model = YOLO(str(project_path(weight_rel)))
                    model.train(**train_arguments(runtime_yaml, run_name))
                    history, final_epoch, best_epoch = read_history(run_dir / "results.csv")
                    record.update({
                        "epochs_completed": len(history),
                        "actual_train_seconds": number(final_epoch.get("time")) if final_epoch else None,
                        "wall_seconds_including_load_and_train": round(time.perf_counter() - started, 3),
                        "best_epoch": int(best_epoch["epoch"]) if best_epoch else None,
                        "best_epoch_elapsed_seconds": number(best_epoch.get("time")) if best_epoch else None,
                        "best_epoch_map50": number(best_epoch.get("metrics/mAP50(B)")) if best_epoch else None,
                        "best_epoch_map50_95": number(best_epoch.get("metrics/mAP50-95(B)")) if best_epoch else None,
                        "final_train_loss": loss_summary(final_epoch, "train"),
                        "final_val_loss": loss_summary(final_epoch, "val"),
                        "stop_reason": "patience early stopping" if "EarlyStopping: Training stopped early" in log_path.read_text(encoding="utf-8", errors="ignore") else "10-minute cap or epoch limit",
                        "status": "complete",
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    print("PILOT_SUMMARY=" + json.dumps(record, ensure_ascii=False, allow_nan=False))
        except Exception as exc:
            record.update({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds_including_load_and_train": round(time.perf_counter() - started, 3),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(f"Pilot failed for {model_name}: {record['error']}", file=sys.stderr)
        rows.append(record)
        (out_dir / "pilot_records.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if all(row["status"] == "complete" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
