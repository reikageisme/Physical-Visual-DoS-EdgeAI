"""Train one full-scale seed trajectory and evaluate its time snapshots."""

from __future__ import annotations

import argparse
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

MODELS = {
    "yolo11n": {"label": "YOLO11n", "weights": "weights/yolo11n.pt", "source": "yolov13_src"},
    "yolov12n": {"label": "YOLOv12n", "weights": "weights/yolov12n.pt", "source": "yolov12_src"},
    "yolov13n": {"label": "YOLOv13n", "weights": "weights/yolov13n.pt", "source": "yolov13_src"},
    "yolov13s": {"label": "YOLOv13s", "weights": "weights/yolov13s.pt", "source": "yolov13_src"},
}
SEEDS = (42, 100, 2026, 777, 999)
MILESTONES_MINUTES = (5, 10, 20, 30, 60, 120)
MAX_EPOCHS = 300
MAX_TRAIN_SECONDS = 120 * 60
IMAGE_SIZE = 416
BATCH_SIZE = 8


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


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def json_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_model_api(source_dir: Path):
    # The official YOLOv12 and YOLOv13 repositories carry their own Ultralytics
    # forks. Load each architecture in its matching source tree in this process.
    sys.path.insert(0, str(source_dir))
    from ultralytics import YOLO
    import ultralytics

    return YOLO, ultralytics


def train_arguments(data_yaml: Path, run_name: str, seed: int, project_dir: Path):
    return {
        "data": str(data_yaml),
        "epochs": MAX_EPOCHS,
        "patience": MAX_EPOCHS + 1,
        "batch": BATCH_SIZE,
        "imgsz": IMAGE_SIZE,
        "save": True,
        "save_period": 1,
        "cache": False,
        "device": "0",
        "workers": 0,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "SGD",
        "verbose": True,
        "seed": seed,
        "deterministic": True,
        "single_cls": False,
        "cos_lr": False,
        "close_mosaic": 0,
        "resume": False,
        "amp": True,
        "val": True,
        "plots": False,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
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
    # Ultralytics names its saved checkpoint epochN.pt with a zero-based N.
    return run_dir / "weights" / f"epoch{csv_epoch - 1}.pt"


def metric_summary(metrics) -> dict:
    box = metrics.box
    speed = {key: number(value) for key, value in (metrics.speed or {}).items()}
    latency = sum(speed.get(key, 0.0) or 0.0 for key in ("preprocess", "inference", "postprocess"))
    return {
        "precision": number(box.mp),
        "recall": number(box.mr),
        "map50": number(box.map50),
        "map50_95": number(box.map),
        "validator_speed_ms_batch8": speed,
        "validator_latency_ms_per_frame_batch8": latency or None,
    }


def evaluate_checkpoints(model_key: str, model_spec: dict, run_name: str, run_dir: Path, history: list[dict], runtime_yaml: Path, output_root: Path, YOLO) -> list[dict]:
    eval_root = output_root / "eval" / model_key
    checkpoint_cache: dict[str, dict] = {}
    milestones = []
    for budget in MILESTONES_MINUTES:
        eligible = [row for row in history if (number(row.get("time")) or 0) <= budget * 60]
        if not eligible:
            milestones.append({
                "budget_minutes": budget,
                "status": "no_completed_epoch_at_or_before_budget",
                "selected_epoch": None,
                "selected_elapsed_seconds": None,
                "validation": None,
                "test": None,
                "checkpoint": None,
            })
            continue
        selected = max(eligible, key=lambda row: number(row.get("metrics/mAP50-95(B)")) or -1.0)
        endpoint = max(eligible, key=lambda row: number(row.get("epoch")) or -1)
        selected_epoch = int(float(selected["epoch"]))
        checkpoint = checkpoint_for_epoch(run_dir, selected_epoch)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing per-epoch checkpoint: {checkpoint}")
        key = str(checkpoint)
        if key not in checkpoint_cache:
            evaluated = YOLO(str(checkpoint))
            params_m = sum(param.numel() for param in evaluated.model.parameters()) / 1_000_000
            from ultralytics.utils.torch_utils import get_flops

            flops_g = get_flops(evaluated.model, imgsz=IMAGE_SIZE)
            eval_name = f"{run_name}-{budget}m-test"
            metrics = evaluated.val(
                data=str(runtime_yaml),
                split="test",
                imgsz=IMAGE_SIZE,
                batch=BATCH_SIZE,
                workers=0,
                device="0",
                plots=False,
                project=str(eval_root),
                name=eval_name,
                exist_ok=True,
                verbose=False,
            )
            checkpoint_cache[key] = {"test": metric_summary(metrics), "parameters_m": params_m, "flops_g": flops_g}
            del evaluated, metrics
        cached = checkpoint_cache[key]
        milestones.append({
            "budget_minutes": budget,
            "status": "complete",
            "selected_epoch": selected_epoch,
            "selected_elapsed_seconds": number(selected.get("time")),
            "validation": {
                "precision": number(selected.get("metrics/precision(B)")),
                "recall": number(selected.get("metrics/recall(B)")),
                "map50": number(selected.get("metrics/mAP50(B)")),
                "map50_95": number(selected.get("metrics/mAP50-95(B)")),
            },
            "endpoint_epoch": int(float(endpoint["epoch"])),
            "endpoint_elapsed_seconds": number(endpoint.get("time")),
            "endpoint_loss": {
                "train_box": number(endpoint.get("train/box_loss")),
                "train_cls": number(endpoint.get("train/cls_loss")),
                "train_dfl": number(endpoint.get("train/dfl_loss")),
                "val_box": number(endpoint.get("val/box_loss")),
                "val_cls": number(endpoint.get("val/cls_loss")),
                "val_dfl": number(endpoint.get("val/dfl_loss")),
            },
            "test": cached["test"],
            "parameters_m": cached["parameters_m"],
            "flops_g": cached["flops_g"],
            "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        })
    return milestones


def execute(model_key: str, seed: int, cap_minutes: int) -> int:
    require_local_venv()
    if model_key not in MODELS:
        raise ValueError(f"Unsupported model key: {model_key}")
    if seed not in SEEDS:
        raise ValueError(f"Unsupported seed: {seed}")
    spec = MODELS[model_key]
    source_dir = project_path(spec["source"], must_exist=True)
    weights = project_path(spec["weights"], must_exist=True)
    dataset_yaml = project_path("dataset/data.yaml", must_exist=True)
    validate_dataset_yaml(dataset_yaml, require_test=True)
    runtime_yaml = materialize_ultralytics_dataset_yaml(dataset_yaml)

    output_root = project_path("results/fullscale_experiments")
    record_dir = output_root / "records" / model_key
    record_path = record_dir / f"seed-{seed}.json"
    log_path = output_root / "logs" / f"{model_key}-seed{seed}.log"
    base_run_name = f"fullscale-{model_key}-seed{seed}-{cap_minutes}m"
    run_name = base_run_name
    run_dir = project_path(Path("runs/train") / run_name)
    record_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"Already complete: {record_path.relative_to(PROJECT_ROOT)}", flush=True)
            return 0
    if run_dir.exists():
        history_path = run_dir / "results.csv"
        reusable_full_run = False
        if history_path.is_file():
            previous_history = read_history(history_path)
            previous_elapsed = number(previous_history[-1].get("time")) if previous_history else None
            reusable_full_run = previous_elapsed is not None and previous_elapsed >= cap_minutes * 60 * 0.97
        if not reusable_full_run:
            raise RuntimeError(
                f"Found an incomplete prior run at {run_dir}; preserving it and stopping instead of retrying automatically."
            )

    started_utc = datetime.now(timezone.utc).isoformat()
    started_wall = time.perf_counter()
    run_metadata = {
        "study_id": "fullscale-20261011",
        "model_key": model_key,
        "model_name": spec["label"],
        "seed": seed,
        "status": "running",
        "started_at_utc": started_utc,
        "run_name": run_name,
        "dataset_yaml": "dataset/data.yaml",
        "model_source": spec["source"],
        "weights": spec["weights"],
        "milestones_minutes": list(MILESTONES_MINUTES),
        "requested_training_cap_minutes": cap_minutes,
        "training_config": {
            "epochs": MAX_EPOCHS,
            "patience": MAX_EPOCHS + 1,
            "batch": BATCH_SIZE,
            "imgsz": IMAGE_SIZE,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "amp": True,
            "deterministic": True,
            "workers": 0,
            "mosaic": 1.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "mixup": 0.0,
            "seed": seed,
        },
    }
    json_write(record_path, run_metadata)

    with log_path.open("a", encoding="utf-8", buffering=1) as logfile:
        tee = Tee(sys.stdout, logfile)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"\n=== {run_name} ===")
            print(f"start={started_utc}; weights={weights}; source={source_dir}; data={runtime_yaml}")
            try:
                import torch

                YOLO, ultralytics = load_model_api(source_dir)
                if not (run_dir / "results.csv").is_file():
                    model = YOLO(str(weights))
                    clock = {"cap_reached": False, "training_start": None}

                    def stop_at_cap(trainer):
                        elapsed = time.time() - trainer.train_time_start
                        clock["training_start"] = trainer.train_time_start
                        if elapsed >= cap_minutes * 60:
                            clock["cap_reached"] = True
                            trainer.stop = True
                            print(f"\nTIME_CAP reached after {elapsed:.1f}s; stopping after saved epoch {trainer.epoch + 1}.")

                    model.add_callback("on_fit_epoch_end", stop_at_cap)
                    model.train(**train_arguments(runtime_yaml, run_name, seed, project_path("runs/train")))
                    del model
                else:
                    print("Training history exists; reusing it and continuing snapshot evaluation.")

                history_path = run_dir / "results.csv"
                history = read_history(history_path)
                if not history:
                    raise RuntimeError(f"No epoch rows found in {history_path}")
                last_elapsed = number(history[-1].get("time")) or 0.0
                actual_wall = time.perf_counter() - started_wall
                milestone_rows = evaluate_checkpoints(model_key, spec, run_name, run_dir, history, runtime_yaml, output_root, YOLO)
                hardware = {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__,
                    "ultralytics_module": str(Path(ultralytics.__file__).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None,
                }
                final = {
                    **run_metadata,
                    "status": "complete",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "actual_training_seconds": last_elapsed,
                    "process_wall_seconds_before_snapshot_test": actual_wall,
                    "epochs_completed": len(history),
                    "stop_reason": "time cap after epoch" if last_elapsed >= cap_minutes * 60 * 0.97 else "epoch or external stop",
                    "hardware_software": hardware,
                    "results_csv": str((run_dir / "results.csv").relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "log": str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "milestones": milestone_rows,
                }
                json_write(record_path, final)
                print(f"Completed {model_key} seed={seed}; epochs={len(history)}; training_seconds={last_elapsed:.1f}")
                return 0
            except Exception as exc:
                failed = {**run_metadata, "status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc)}
                json_write(record_path, failed)
                raise


def preflight(model_key: str) -> int:
    require_local_venv()
    spec = MODELS[model_key]
    source_dir = project_path(spec["source"], must_exist=True)
    weights = project_path(spec["weights"], must_exist=True)
    import torch

    YOLO, ultralytics = load_model_api(source_dir)
    model = YOLO(str(weights))
    params_m = sum(param.numel() for param in model.model.parameters()) / 1_000_000
    print(json.dumps({
        "model": model_key,
        "label": spec["label"],
        "weights": str(weights.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source": str(source_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_module": str(Path(ultralytics.__file__).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "task": model.task,
        "classes_in_checkpoint": len(model.names),
        "parameters_m": round(params_m, 4),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, ensure_ascii=False, indent=2))
    del model
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--cap-minutes", type=int, default=120)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        return preflight(args.model)
    if args.seed is None:
        parser.error("--seed is required unless --preflight-only is set")
    return execute(args.model, args.seed, args.cap_minutes)


if __name__ == "__main__":
    raise SystemExit(main())
