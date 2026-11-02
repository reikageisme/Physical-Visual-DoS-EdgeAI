"""Smoke-check YOLOv13 inference paths for each implemented ablation variant."""

from __future__ import annotations

import argparse

from pipeline_utils import configure_project_environment, project_path, require_local_venv

configure_project_environment()


def tensors_in(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors_in(item, torch)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item, torch)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["n", "s", "l", "x"], default="n")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_local_venv()
    import torch
    from ultralytics import YOLO
    from ablation_utils import ABLATION_VARIANTS, apply_ablation

    if args.imgsz < 32 or args.imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32 and at least 32.")
    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU verification was requested, but CUDA is unavailable.")
    device = "cuda:0" if args.device == "gpu" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    checkpoint = project_path(f"weights/yolov13{args.scale}.pt", must_exist=True)
    images = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)

    for variant in ABLATION_VARIANTS:
        model = YOLO(str(checkpoint))
        counts = apply_ablation(model, variant)
        model.model.to(device).eval()
        with torch.inference_mode():
            output = model.model(images)
        tensors = list(tensors_in(output, torch))
        if not tensors or any(not torch.isfinite(tensor).all().item() for tensor in tensors):
            raise RuntimeError(f"{variant} produced no finite tensor outputs.")
        print(f"{variant}: finite forward pass; {counts}")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"Verified all YOLOv13-{args.scale.upper()} variants at {args.imgsz}px on {device}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
