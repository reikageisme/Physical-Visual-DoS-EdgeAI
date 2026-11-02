"""Fine-tune YOLOv8/10/11/12 checkpoints with the same settings as YOLOv13."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline_utils import (
    PROJECT_ROOT,
    configure_project_environment,
    materialize_ultralytics_dataset_yaml,
    project_path,
    require_local_venv,
    source_revision,
    utc_timestamp,
    validate_dataset_yaml,
    write_json,
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--data", help="Override data_yaml with a YAML file inside this project.")
    parser.add_argument("--scale", choices=["n", "s", "l", "x"], help="Shared model size.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["yolov8", "yolov10", "yolo11", "yolo12"],
        default=["yolov8", "yolov10", "yolo11", "yolo12"],
    )
    parser.add_argument("--device", help="Override device, for example 0 or cpu.")
    return parser.parse_args()


def main():
    args = parse_args()
    require_local_venv()
    try:
        import torch
        import yaml
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"Missing project dependency {exc.name!r}. Run setup_env.bat first.") from exc

    config = yaml.safe_load(project_path(args.config, must_exist=True).read_text(encoding="utf-8-sig")) or {}
    scale = args.scale or str(config.get("scale", "n")).lower()
    data_path = project_path(args.data or config.get("data_yaml", "dataset/data.yaml"), must_exist=True)
    validate_dataset_yaml(data_path, require_test=False)
    runtime_data_path = materialize_ultralytics_dataset_yaml(data_path)
    device = args.device or str(config.get("device", "auto"))
    run_prefix = str(config.get("run_prefix", ""))
    if device.lower() == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    for family in args.models:
        checkpoint_name = f"{family}{scale}"
        model_name = f"{run_prefix}{checkpoint_name}"
        checkpoint = project_path(f"weights/{checkpoint_name}.pt")
        source = str(checkpoint) if checkpoint.is_file() else f"{model_name}.pt"
        run_dir = project_path(Path("runs/train") / model_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "console.log"
        started_at_utc = utc_timestamp()
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        with log_path.open("a", encoding="utf-8") as log_handle:
            sys.stdout = Tee(saved_stdout, log_handle)
            sys.stderr = Tee(saved_stderr, log_handle)
            try:
                print(f"Fine-tuning {model_name} from {source}")
                print(f"Dataset YAML: {data_path}; device: {device}; source revision: {source_revision()}")
                model = YOLO(source)
                train_args = {
                    key: value
                    for key, value in config.items()
                    if key not in {"data_yaml", "scale", "pretrained", "augment", "device", "run_prefix"}
                }
                train_args.update(config.get("augment", {}))
                train_args.update(
                    {
                        "data": str(runtime_data_path),
                        "project": str(project_path("runs/train")),
                        "name": model_name,
                        "exist_ok": True,
                        "device": device,
                        "plots": True,
                        "save": True,
                        "val": True,
                        "workers": int(config.get("workers", 0)),
                        "seed": int(config.get("seed", 42)),
                        "deterministic": True,
                        "verbose": True,
                    }
                )
                model.train(**train_args)
                best = project_path(Path("runs/train") / model_name / "weights/best.pt", must_exist=True)
                write_json(
                    Path("runs/train") / model_name / "metadata.json",
                    {
                        "model": model_name,
                        "family": family,
                        "scale": scale,
                        "started_at_utc": started_at_utc,
                        "data_yaml": str(data_path.relative_to(PROJECT_ROOT)),
                        "checkpoint": str(best.relative_to(PROJECT_ROOT)),
                        "source_revision": source_revision(),
                        "python": sys.version,
                        "torch": torch.__version__,
                        "device": device,
                        "train_args": train_args,
                    },
                )
                print(f"Best checkpoint: {best}")
            finally:
                sys.stdout, sys.stderr = saved_stdout, saved_stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
