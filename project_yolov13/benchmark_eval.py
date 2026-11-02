"""Evaluate configured local checkpoints on fixed validation/test splits and measure CPU/GPU performance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline_utils import (
    PROJECT_ROOT,
    configure_project_environment,
    materialize_ultralytics_dataset_yaml,
    project_path,
    require_local_venv,
    sha256_file,
    utc_timestamp,
    validate_dataset_yaml,
    write_csv,
)

configure_project_environment()

FIELDS = [
    "model_name", "family", "scale", "split", "device", "precision", "recall",
    "map50", "map50_95", "params_m", "flops_g", "latency_ms", "fps",
    "imgsz", "batch", "dataset_yaml", "checkpoint", "checkpoint_sha256",
    "source", "timing_protocol", "measured_at_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--data", help="Override configured dataset YAML.")
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--devices", choices=["auto", "cpu", "gpu", "both"], default="auto")
    parser.add_argument("--output", default="results/benchmark_eval.csv")
    parser.add_argument("--skip-missing", action="store_true", help="Skip absent checkpoints instead of failing.")
    return parser.parse_args()


def devices_for(requested: str, torch) -> list[tuple[str, str]]:
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cpu":
        return [("cpu", "cpu")]
    if requested == "gpu":
        if not cuda_available:
            raise RuntimeError("GPU evaluation was requested, but torch.cuda.is_available() is false.")
        return [("GPU", "0")]
    if requested == "both":
        return [("CPU", "cpu")] + ([("GPU", "0")] if cuda_available else [])
    return [("CPU", "cpu")] + ([("GPU", "0")] if cuda_available else [])


def get_model_costs(model, imgsz: int) -> tuple[float | None, float | None]:
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    params_m = parameter_count / 1_000_000.0
    try:
        from ultralytics.utils.torch_utils import get_flops

        flops = float(get_flops(model.model, imgsz=imgsz))
        flops_g = flops if flops > 0 else None
    except Exception:
        flops_g = None
    return params_m, flops_g


def metric_values(metrics) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    if box is None:
        raise RuntimeError("Validation did not return detection metrics.")
    values = (
        float(box.mp),
        float(box.mr),
        float(box.map50),
        float(box.map),
        float(speed.get("inference")) if speed.get("inference") is not None else None,
    )
    return values


def main() -> int:
    args = parse_args()
    require_local_venv()
    try:
        import torch
        import yaml
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"Missing project dependency {exc.name!r}. Run setup_env.bat first.") from exc

    config_path = project_path(args.config, must_exist=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    data_path = project_path(args.data or config.get("data_yaml", "dataset/data.yaml"), must_exist=True)
    validate_dataset_yaml(data_path, require_test=("test" in args.splits))
    runtime_data_path = materialize_ultralytics_dataset_yaml(data_path)
    imgsz = int(config.get("imgsz", 640))
    batch = int(config.get("batch", 16))
    workers = int(config.get("workers", 0))
    devices = devices_for(args.devices, torch)
    output = project_path(args.output)
    partial_output = output.with_name(output.stem + ".partial" + output.suffix)

    rows: list[dict] = []
    models = config.get("models", [])
    if not models:
        raise ValueError(f"No models configured in {config_path}")

    for entry in models:
        checkpoint = project_path(entry["checkpoint"])
        if not checkpoint.is_file():
            message = f"Checkpoint missing for {entry.get('name', entry['checkpoint'])}: {checkpoint}"
            if args.skip_missing:
                print(f"SKIP: {message}")
                continue
            raise FileNotFoundError(message)

        print(f"Loading {entry['name']} from {checkpoint.relative_to(PROJECT_ROOT)}")
        base_model = YOLO(str(checkpoint))
        params_m, flops_g = get_model_costs(base_model, imgsz)
        checkpoint_hash = sha256_file(checkpoint)

        for split in args.splits:
            for device_label, device_arg in devices:
                result = base_model.val(
                    data=str(runtime_data_path),
                    split=split,
                    imgsz=imgsz,
                    batch=batch,
                    workers=workers,
                    device=device_arg,
                    conf=float(config.get("conf", 0.001)),
                    iou=float(config.get("iou", 0.7)),
                    plots=bool(config.get("plots", True)),
                    project=str(project_path("results/benchmark")),
                    name=f"{entry['name']}_{split}_{device_label.lower()}",
                    exist_ok=True,
                    verbose=True,
                )
                precision, recall, map50, map50_95, latency_ms = metric_values(result)
                fps = (1000.0 / latency_ms) if latency_ms and latency_ms > 0 else None
                row = {
                    "model_name": entry["name"],
                    "family": entry.get("family", ""),
                    "scale": entry.get("scale", ""),
                    "split": split,
                    "device": device_label,
                    "precision": precision,
                    "recall": recall,
                    "map50": map50,
                    "map50_95": map50_95,
                    "params_m": params_m,
                    "flops_g": flops_g,
                    "latency_ms": latency_ms,
                    "fps": fps,
                    "imgsz": imgsz,
                    "batch": batch,
                    "dataset_yaml": str(data_path.relative_to(PROJECT_ROOT)),
                    "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                    "checkpoint_sha256": checkpoint_hash,
                    "source": entry.get("source", ""),
                    "timing_protocol": "Ultralytics validator inference time per image; excludes model loading; batch size recorded.",
                    "measured_at_utc": utc_timestamp(),
                }
                rows.append(row)
                write_csv(partial_output, rows, FIELDS)
                print(
                    f"{entry['name']} | {split} | {device_label}: "
                    f"P={precision:.4f} R={recall:.4f} mAP50={map50:.4f} "
                    f"mAP50:95={map50_95:.4f} latency={latency_ms} ms/image FPS={fps}"
                )

    if not rows:
        raise RuntimeError("No models were evaluated. Supply checkpoints or remove --skip-missing.")
    write_csv(output, rows, FIELDS)
    partial_output.unlink(missing_ok=True)
    print(f"Saved {len(rows)} measurements to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
