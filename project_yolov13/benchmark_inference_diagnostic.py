"""Measure YOLOv13 inference throughput on a synthetic frame (not dataset evaluation)."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from pipeline_utils import (
    PROJECT_ROOT,
    configure_project_environment,
    project_path,
    require_local_venv,
    utc_timestamp,
    write_csv,
    write_json,
)

# Set all application caches before importing Ultralytics or PyTorch.
configure_project_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", nargs="+", choices=["n", "s", "l", "x"], default=["n", "s", "l", "x"])
    parser.add_argument("--devices", choices=["cpu", "gpu", "both"], default="both")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", default="results/local_inference_diagnostic.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_local_venv()
    if args.imgsz <= 0 or args.warmup < 0 or args.iterations < 1 or args.cpu_threads < 1:
        raise ValueError("imgsz, iterations and cpu-threads must be positive; warmup cannot be negative.")

    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    if args.devices in {"gpu", "both"} and not torch.cuda.is_available():
        raise RuntimeError("GPU measurements were requested, but CUDA is unavailable.")
    torch.set_num_threads(args.cpu_threads)
    devices = ["cpu"]
    if args.devices in {"gpu", "both"}:
        devices.append("0")

    frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    rows: list[dict] = []
    samples_by_model: dict[str, list[dict]] = {}
    measured_at = utc_timestamp()

    for scale in args.scales:
        checkpoint = project_path(Path("weights") / f"yolov13{scale}.pt", must_exist=True)
        profile_model = YOLO(str(checkpoint))
        params_m = sum(parameter.numel() for parameter in profile_model.model.parameters()) / 1_000_000
        flops_g = float(get_flops(profile_model.model, imgsz=args.imgsz))
        del profile_model

        for device in devices:
            model = YOLO(str(checkpoint))
            label = "CPU" if device == "cpu" else f"GPU ({torch.cuda.get_device_name(0)})"

            def predict() -> None:
                model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    device=device,
                    batch=1,
                    conf=0.25,
                    verbose=False,
                    save=False,
                )

            for _ in range(args.warmup):
                predict()
            if device != "cpu":
                torch.cuda.synchronize()

            timings_ms = []
            for _ in range(args.iterations):
                if device != "cpu":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                predict()
                if device != "cpu":
                    torch.cuda.synchronize()
                timings_ms.append((time.perf_counter() - started) * 1000)

            median_ms = statistics.median(timings_ms)
            model_name = f"YOLOv13{scale}"
            row = {
                "model": model_name,
                "scale": scale,
                "device": label,
                "runtime": "PyTorch FP32 / Ultralytics predict",
                "imgsz": args.imgsz,
                "batch": 1,
                "params_m": round(params_m, 4),
                "flops_g": round(flops_g, 4),
                "warmup_iterations": args.warmup,
                "timed_iterations": args.iterations,
                "synthetic_input": f"zero-valued {args.imgsz}x{args.imgsz} RGB image; inference-only diagnostic",
                "latency_median_ms": round(median_ms, 3),
                "latency_mean_ms": round(statistics.mean(timings_ms), 3),
                "latency_p90_ms": round(float(np.percentile(timings_ms, 90)), 3),
                "fps_from_median": round(1000 / median_ms, 2),
                "measured_at_utc": measured_at,
            }
            rows.append(row)
            samples_by_model[f"{model_name}:{label}"] = [
                {"iteration": index + 1, "latency_ms": round(value, 3)}
                for index, value in enumerate(timings_ms)
            ]
            print(
                f"{model_name} | {label}: median={row['latency_median_ms']} ms, "
                f"p90={row['latency_p90_ms']} ms, FPS={row['fps_from_median']}"
            )
            del model
            if device != "cpu":
                torch.cuda.empty_cache()

    output = project_path(args.output)
    write_csv(output.relative_to(PROJECT_ROOT), rows)
    write_json(
        output.with_suffix(".json").relative_to(PROJECT_ROOT),
        {
            "protocol": (
                "PyTorch FP32, Ultralytics predict, batch 1, synthetic all-zero RGB input; "
                f"{args.warmup} warmups and {args.iterations} timed iterations; GPU synchronized around each timed call. "
                "Inference-only diagnostic, not a dataset benchmark."
            ),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "results": rows,
            "latency_samples": samples_by_model,
        },
    )
    print(f"Saved inference-only diagnostic to {output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
