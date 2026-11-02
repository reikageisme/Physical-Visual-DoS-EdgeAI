"""Fine-tune an official iMoonLab YOLOv13 checkpoint on the fixed local dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--data", help="Override data_yaml with a YAML file inside this project.")
    parser.add_argument("--scale", choices=["n", "s", "l", "x"], help="YOLOv13 model size.")
    parser.add_argument("--weights", help="Use a local checkpoint inside this project instead of the released default.")
    parser.add_argument("--scratch", action="store_true", help="Initialize from the YOLOv13 architecture YAML, without pretrained weights.")
    parser.add_argument("--device", help="Override device, for example 0 or cpu.")
    parser.add_argument("--name", help="Run folder name under runs/train/.")
    return parser.parse_args()


def released_checkpoint(scale: str) -> Path:
    target = project_path(f"weights/yolov13{scale}.pt")
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".pt.part")
    url = f"https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13{scale}.pt"
    print(f"Downloading official YOLOv13-{scale.upper()} checkpoint into {target.relative_to(PROJECT_ROOT)}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "project-yolov13-research-pipeline"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if partial.stat().st_size < 100_000:
            raise RuntimeError("The downloaded checkpoint is unexpectedly small.")
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


def metric_summary(metrics) -> dict:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    return {
        "precision": float(box.mp) if box is not None else None,
        "recall": float(box.mr) if box is not None else None,
        "map50": float(box.map50) if box is not None else None,
        "map50_95": float(box.map) if box is not None else None,
        "speed_ms_per_image": {key: float(value) for key, value in speed.items()},
    }


def main() -> int:
    args = parse_args()
    require_local_venv()
    try:
        import torch
        import yaml
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"Missing project dependency {exc.name!r}. Run setup_env.bat first.") from exc

    config = load_yaml(args.config)
    scale = args.scale or str(config.get("scale", "n")).lower()
    if scale not in {"n", "s", "l", "x"}:
        raise ValueError("scale must be one of n, s, l, x.")
    data_path = project_path(args.data or config.get("data_yaml", "dataset/data.yaml"), must_exist=True)
    validate_dataset_yaml(data_path, require_test=True)
    runtime_data_path = materialize_ultralytics_dataset_yaml(data_path)

    if args.scratch and args.weights:
        raise ValueError("Choose either --scratch or --weights, not both.")
    if args.weights:
        checkpoint = project_path(args.weights, must_exist=True)
    elif bool(config.get("pretrained", True)) and not args.scratch:
        checkpoint = released_checkpoint(scale)
    else:
        checkpoint = None

    run_name = args.name or f"yolov13{scale}"
    run_dir = project_path(Path("runs/train") / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console.log"
    started_at_utc = utc_timestamp()

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    with log_path.open("a", encoding="utf-8") as log_handle:
        sys.stdout = Tee(saved_stdout, log_handle)
        sys.stderr = Tee(saved_stderr, log_handle)
        try:
            print(f"Project: {PROJECT_ROOT}")
            print(f"Dataset YAML: {data_path}")
            print(f"YOLOv13 scale: {scale}; source revision: {source_revision() or 'unknown'}")
            print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")

            if checkpoint is not None:
                print(f"Initialization checkpoint: {checkpoint}")
                model = YOLO(str(checkpoint))
            else:
                source_yaml = project_path("yolov13_src/ultralytics/cfg/models/v13/yolov13.yaml", must_exist=True)
                architecture = load_yaml(source_yaml)
                architecture["scale"] = scale
                generated_yaml = project_path(f"configs/generated/yolov13{scale}.yaml")
                generated_yaml.parent.mkdir(parents=True, exist_ok=True)
                generated_yaml.write_text(yaml.safe_dump(architecture, sort_keys=False), encoding="utf-8")
                print(f"Scratch architecture: {generated_yaml}")
                model = YOLO(str(generated_yaml))

            device = args.device or str(config.get("device", "auto"))
            if device.lower() == "auto":
                device = "0" if torch.cuda.is_available() else "cpu"
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
                }
            )
            print("Training arguments:")
            print(json.dumps(train_args, indent=2, default=str))
            model.train(**train_args)

            best_checkpoint = project_path(Path("runs/train") / run_name / "weights/best.pt", must_exist=True)
            trained = YOLO(str(best_checkpoint))
            post_eval = []
            for split in ("val", "test"):
                split_name = f"{run_name}_{split}"
                metrics = trained.val(
                    data=str(runtime_data_path),
                    split=split,
                    imgsz=int(config.get("imgsz", 640)),
                    batch=int(config.get("batch", 16)),
                    workers=int(config.get("workers", 0)),
                    device=device,
                    plots=True,
                    project=str(project_path("results/post_train")),
                    name=split_name,
                    exist_ok=True,
                    verbose=True,
                )
                summary = metric_summary(metrics)
                post_eval.append({"split": split, **summary})
                print(f"{split} metrics: {json.dumps(summary, indent=2)}")

            metadata = {
                "model": f"YOLOv13{scale}",
                "scale": scale,
                "started_at_utc": started_at_utc,
                "data_yaml": str(data_path.relative_to(PROJECT_ROOT)),
                "checkpoint": str(best_checkpoint.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256_file(best_checkpoint),
                "source_revision": source_revision(),
                "python": sys.version,
                "torch": torch.__version__,
                "torchvision": __import__("torchvision").__version__,
                "device": device,
                "train_args": train_args,
                "post_train_evaluation": post_eval,
            }
            write_json(Path("runs/train") / run_name / "metadata.json", metadata)
            print(f"Best checkpoint: {best_checkpoint}")
            print(f"Training plots and loss curves: {run_dir}")
            print(f"Validation/test confusion matrices and logs: {project_path('results/post_train')}")
        finally:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
