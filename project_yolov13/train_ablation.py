"""Fine-tune one controlled YOLOv13 architecture ablation on the fixed dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from pipeline_utils import (
    PROJECT_ROOT,
    configure_project_environment,
    materialize_ultralytics_dataset_yaml,
    load_yaml,
    project_path,
    require_local_venv,
    sha256_file,
    source_revision,
    utc_timestamp,
    validate_dataset_yaml,
    write_csv,
    write_json,
)

configure_project_environment()

from ablation_utils import ABLATION_VARIANTS, apply_ablation
from train_yolov13 import Tee, metric_summary, released_checkpoint


ABLATION_FIELDS = [
    "variant", "precision", "recall", "map50", "map50_95", "params_m",
    "flops_g", "latency_ms", "device", "split", "seed", "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--data", help="Override data_yaml with a YAML file inside this project.")
    parser.add_argument("--scale", choices=["n", "s", "l", "x"], default="s")
    parser.add_argument("--variant", choices=ABLATION_VARIANTS, required=True)
    parser.add_argument("--device", help="Override device, for example 0 or cpu.")
    parser.add_argument("--name", help="Run folder name under runs/train/.")
    parser.add_argument("--output", default="results/ablation.csv", help="Project-local CSV for validation ablation measurements.")
    return parser.parse_args()


def upsert_ablation_row(row: dict, output: str | Path) -> Path:
    path = project_path(output)
    existing = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
    existing = [
        item for item in existing
        if not (item.get("variant") == row["variant"] and item.get("split") == row["split"])
    ]
    existing.append(row)
    return write_csv(path, existing, ABLATION_FIELDS)


def main() -> int:
    args = parse_args()
    require_local_venv()
    try:
        import torch
        from ultralytics import YOLO
        from benchmark_eval import get_model_costs
    except ImportError as exc:
        raise SystemExit(f"Missing project dependency {exc.name!r}. Run setup_env.bat first.") from exc

    config = load_yaml(args.config)
    data_path = project_path(args.data or config.get("data_yaml", "dataset/data.yaml"), must_exist=True)
    validate_dataset_yaml(data_path, require_test=True)
    runtime_data_path = materialize_ultralytics_dataset_yaml(data_path)
    scale = args.scale
    run_name = args.name or f"yolov13-{scale}-{args.variant}"
    run_dir = project_path(Path("runs/train") / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = released_checkpoint(scale)
    device = args.device or str(config.get("device", "auto"))
    if device.lower() == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    with (run_dir / "console.log").open("a", encoding="utf-8") as log_handle:
        sys.stdout = Tee(saved_stdout, log_handle)
        sys.stderr = Tee(saved_stderr, log_handle)
        try:
            started_at_utc = utc_timestamp()
            print(f"Project: {PROJECT_ROOT}")
            print(f"Dataset YAML: {data_path}")
            print(f"YOLOv13-{scale.upper()} ablation: {args.variant}")
            print(f"Initialization checkpoint: {checkpoint}")
            model = YOLO(str(checkpoint))
            ablation_counts = apply_ablation(model, args.variant)
            print(f"Applied architecture change: {json.dumps(ablation_counts)}")

            train_args = {
                key: value
                for key, value in config.items()
                if key not in {"data_yaml", "scale", "pretrained", "augment", "device", "run_prefix"}
            }
            train_args.update(config.get("augment", {}))
            train_args.update({
                "data": str(runtime_data_path),
                "project": str(project_path("runs/train")),
                "name": run_name,
                "exist_ok": True,
                "device": device,
                "plots": True,
                "save": True,
                "val": True,
                "workers": int(config.get("workers", 0)),
                "seed": int(config.get("seed", 42)),
                "deterministic": True,
                "verbose": True,
            })
            print("Training arguments:")
            print(json.dumps(train_args, indent=2, default=str))
            model.train(**train_args)

            best_checkpoint = project_path(Path("runs/train") / run_name / "weights/best.pt", must_exist=True)
            trained = YOLO(str(best_checkpoint))
            # Reapply switches after checkpoint reload to ensure validation uses
            # the same ablated forward paths even if a trainer strips Python flags.
            ablation_counts = apply_ablation(trained, args.variant)
            post_eval = []
            val_summary = None
            for split in ("val", "test"):
                metrics = trained.val(
                    data=str(runtime_data_path),
                    split=split,
                    imgsz=int(config.get("imgsz", 640)),
                    batch=int(config.get("batch", 1)),
                    workers=int(config.get("workers", 0)),
                    device=device,
                    plots=True,
                    project=str(project_path("results/post_train")),
                    name=f"{run_name}_{split}",
                    exist_ok=True,
                    verbose=True,
                )
                summary = metric_summary(metrics)
                post_eval.append({"split": split, **summary})
                if split == "val":
                    val_summary = summary
                print(f"{split} metrics: {json.dumps(summary, indent=2)}")

            params_m, flops_g = get_model_costs(trained, int(config.get("imgsz", 640)))
            if val_summary is None:
                raise RuntimeError("Validation metrics were not produced for the ablation run.")
            ablation_label = f"YOLOv13-{scale}-{args.variant}"
            val_speed = val_summary.get("speed_ms_per_image", {})
            latency_ms = val_speed.get("inference")
            row = {
                "variant": ablation_label,
                "precision": val_summary.get("precision"),
                "recall": val_summary.get("recall"),
                "map50": val_summary.get("map50"),
                "map50_95": val_summary.get("map50_95"),
                "params_m": params_m,
                "flops_g": flops_g,
                "latency_ms": latency_ms,
                "device": device,
                "split": "val",
                "seed": int(config.get("seed", 42)),
                "notes": f"Run={run_name}; checkpoint={best_checkpoint.relative_to(PROJECT_ROOT)}; ablation={json.dumps(ablation_counts)}",
            }
            csv_path = upsert_ablation_row(row, args.output)
            metadata = {
                "model": f"YOLOv13-{scale.upper()}",
                "variant": args.variant,
                "scale": scale,
                "started_at_utc": started_at_utc,
                "data_yaml": str(data_path.relative_to(PROJECT_ROOT)),
                "initial_checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                "checkpoint": str(best_checkpoint.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256_file(best_checkpoint),
                "source_revision": source_revision(),
                "python": sys.version,
                "torch": torch.__version__,
                "device": device,
                "ablation_modules": ablation_counts,
                "train_args": train_args,
                "post_train_evaluation": post_eval,
                "validation_ablation_row": row,
            }
            write_json(Path("runs/train") / run_name / "metadata.json", metadata)
            print(f"Best checkpoint: {best_checkpoint}")
            print(f"Ablation metrics appended to: {csv_path}")
        finally:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
