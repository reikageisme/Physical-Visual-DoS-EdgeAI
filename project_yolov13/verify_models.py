"""Download and verify official YOLOv13 n/s/l/x checkpoints inside this project."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_utils import (
    PROJECT_ROOT,
    configure_project_environment,
    require_local_venv,
    sha256_file,
    source_revision,
    utc_timestamp,
    write_json,
)

configure_project_environment()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", nargs="+", choices=["n", "s", "l", "x"], default=["n", "s", "l", "x"])
    args = parser.parse_args()
    require_local_venv()

    try:
        import torch
        from ultralytics import YOLO
        from train_yolov13 import released_checkpoint
    except ImportError as exc:
        raise SystemExit(f"Missing project dependency {exc.name!r}. Run setup_env.bat first.") from exc

    manifest = []
    for scale in args.scales:
        checkpoint = released_checkpoint(scale)
        model = YOLO(str(checkpoint))
        parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
        manifest.append(
            {
                "scale": scale,
                "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "parameters": parameter_count,
            }
        )
        print(
            f"YOLOv13-{scale.upper()}: loaded {checkpoint.relative_to(PROJECT_ROOT)}; "
            f"{parameter_count:,} parameters; {checkpoint.stat().st_size:,} bytes"
        )
        del model

    write_json(
        "weights/manifest.json",
        {
            "verified_at_utc": utc_timestamp(),
            "source_revision": source_revision(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "models": manifest,
        },
    )
    print("Saved project-local checkpoint manifest to weights/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())