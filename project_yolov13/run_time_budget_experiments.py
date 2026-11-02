"""Run reproducible wall-clock capped fine-tuning trials for YOLO11n and YOLOv13n."""

from __future__ import annotations

import csv
import json
import math
import os
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
BUDGETS_HOURS = (1, 2, 4, 8)
MODELS = {
    "yolo11n": "weights/yolo11n.pt",
    "yolov13n": "weights/yolov13n.pt",
}
IMAGE_SIZE = 416
BATCH_SIZE = 4
MAX_EPOCHS = 600
PATIENCE = 10


def as_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def read_history(path: Path) -> tuple[list[dict], dict | None, dict | None]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        return rows, None, None
    best_row = max(rows, key=lambda row: as_float(row.get("metrics/mAP50-95(B)")) or -1)
    return rows, rows[-1], best_row


def loss_summary(row: dict | None, prefix: str) -> dict:
    if not row:
        return {"box": None, "cls": None, "dfl": None, "total": None}
    values = {key: as_float(row.get(f"{prefix}/{key}_loss")) for key in ("box", "cls", "dfl")}
    values["total"] = sum(values.values()) if all(value is not None for value in values.values()) else None
    return values


def metric_summary(metrics) -> dict:
    box = metrics.box
    speed = {key: as_float(value) for key, value in (metrics.speed or {}).items()}
    total_latency = sum(value for value in speed.values() if value is not None)
    return {
        "precision": as_float(box.mp),
        "recall": as_float(box.mr),
        "map50": as_float(box.map50),
        "map50_95": as_float(box.map),
        "speed_ms": speed,
        "latency_ms": speed.get("inference"),
        "fps_from_total_validator_ms": 1000.0 / total_latency if total_latency > 0 else None,
    }


def save_records(records: list[dict], output_dir: Path) -> None:
    json_path = output_dir / "time_budget_records.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if not records:
        return
    columns = list(dict.fromkeys(key for row in records for key in row))
    with (output_dir / "time_budget_records.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def train_arguments(data_yaml: Path, model_name: str, run_name: str, budget_hours: int) -> dict:
    return {
        "data": str(data_yaml),
        "epochs": MAX_EPOCHS,
        "time": float(budget_hours),
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
    from ultralytics.utils.torch_utils import get_flops

    output_dir = project_path("results/time_budget")
    logs_dir = output_dir / "logs"
    eval_dir = output_dir / "eval"
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = project_path("dataset/adv_tshirt_patch_detection.yaml", must_exist=True)
    validate_dataset_yaml(data_yaml, require_test=True)
    runtime_data_yaml = materialize_ultralytics_dataset_yaml(data_yaml)
    for model_path in MODELS.values():
        project_path(model_path, must_exist=True)

    records_path = output_dir / "time_budget_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    done = {(row.get("model_key"), row.get("budget_hours")) for row in records if row.get("status") == "complete"}

    hardware = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None,
        "study_id": STUDY_ID,
        "dataset_yaml": str(data_yaml.relative_to(PROJECT_ROOT)),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "seed": 42,
    }
    (output_dir / "environment.json").write_text(json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(hardware, ensure_ascii=False))

    for model_key, weight_rel in MODELS.items():
        for budget_hours in BUDGETS_HOURS:
            if (model_key, budget_hours) in done:
                print(f"SKIP completed {model_key} at {budget_hours}h")
                continue

            run_name = f"timebudget-{STUDY_ID}-{model_key}-{budget_hours}h"
            run_dir = project_path(Path("runs/train") / run_name)
            log_path = logs_dir / f"{run_name}.log"
            row = {
                "study_id": STUDY_ID,
                "model_key": model_key,
                "model_name": "YOLO11n" if model_key == "yolo11n" else "YOLOv13n",
                "budget_hours": budget_hours,
                "status": "running",
                "run_name": run_name,
                "checkpoint": str((run_dir / "weights/best.pt").relative_to(PROJECT_ROOT)),
                "log": str(log_path.relative_to(PROJECT_ROOT)),
                "data_yaml": str(data_yaml.relative_to(PROJECT_ROOT)),
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "actual_train_seconds": None,
                "epochs_completed": None,
                "stop_reason": None,
            }
            stdout_original, stderr_original = sys.stdout, sys.stderr
            started = time.perf_counter()
            try:
                with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
                    tee = Tee(stdout_original, log_handle)
                    with redirect_stdout(tee), redirect_stderr(tee):
                        print(f"\n=== {run_name} | cap={budget_hours}h | {datetime.now(timezone.utc).isoformat()} ===")
                        print(f"weight={project_path(weight_rel)}")
                        model = YOLO(str(project_path(weight_rel)))
                        model.train(**train_arguments(runtime_data_yaml, model_key, run_name, budget_hours))

                        history, final_epoch, best_epoch = read_history(run_dir / "results.csv")
                        actual_train_seconds = as_float(final_epoch.get("time")) if final_epoch else None
                        row.update(
                            {
                                "actual_train_seconds": actual_train_seconds,
                                "wall_seconds_including_load_and_train": round(time.perf_counter() - started, 3),
                                "epochs_completed": len(history),
                                "final_epoch": int(final_epoch["epoch"]) if final_epoch and final_epoch.get("epoch") else None,
                                "final_train_loss": loss_summary(final_epoch, "train"),
                                "final_val_loss": loss_summary(final_epoch, "val"),
                                "best_epoch": int(best_epoch["epoch"]) if best_epoch and best_epoch.get("epoch") else None,
                                "best_epoch_map50": as_float(best_epoch.get("metrics/mAP50(B)")) if best_epoch else None,
                                "best_epoch_map50_95": as_float(best_epoch.get("metrics/mAP50-95(B)")) if best_epoch else None,
                                "history_csv": str((run_dir / "results.csv").relative_to(PROJECT_ROOT)),
                            }
                        )

                        early_stop_phrase = "EarlyStopping: Training stopped early" in log_path.read_text(encoding="utf-8", errors="ignore")
                        if early_stop_phrase:
                            row["stop_reason"] = "patience early stopping"
                        elif actual_train_seconds is not None and actual_train_seconds >= budget_hours * 3600:
                            row["stop_reason"] = "wall-clock cap"
                        else:
                            row["stop_reason"] = "training ended before wall-clock cap; inspect log"

                        best_path = run_dir / "weights" / "best.pt"
                        trained = YOLO(str(best_path))
                        row["parameters_m"] = sum(parameter.numel() for parameter in trained.model.parameters()) / 1_000_000
                        row["flops_g"] = get_flops(trained.model, imgsz=IMAGE_SIZE)
                        row["evaluations"] = {}
                        for split in ("val", "test"):
                            metrics = trained.val(
                                data=str(runtime_data_yaml),
                                split=split,
                                imgsz=IMAGE_SIZE,
                                batch=BATCH_SIZE,
                                workers=0,
                                device="0",
                                plots=False,
                                project=str(eval_dir),
                                name=f"{run_name}-{split}",
                                exist_ok=False,
                                verbose=False,
                            )
                            row["evaluations"][split] = metric_summary(metrics)

                        row["status"] = "complete"
                        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                        print("TRIAL_SUMMARY=" + json.dumps(row, ensure_ascii=False, allow_nan=False))
            except Exception as exc:  # record failures with their logs and continue to the next cap
                row.update(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                        "wall_seconds_including_load_and_train": round(time.perf_counter() - started, 3),
                    }
                )
                with log_path.open("a", encoding="utf-8") as log_handle:
                    log_handle.write(f"\nFAILED: {row['error']}\n")
                print(f"FAILED {run_name}: {row['error']}")
            finally:
                sys.stdout, sys.stderr = stdout_original, stderr_original

            records.append(row)
            save_records(records, output_dir)
            print(f"RECORDED {run_name}: status={row['status']} stop={row.get('stop_reason')} epochs={row.get('epochs_completed')}")

    return 0 if all(row.get("status") == "complete" for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
