from __future__ import annotations
import csv
import gc
import statistics
import sys
import time
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, materialize_ultralytics_dataset_yaml, project_path, require_local_venv, sha256_file, utc_timestamp

configure_project_environment()
SCALES = ('n', 's', 'l')
FIELDS = [
    'model_name', 'family', 'scale', 'precision', 'recall', 'map50', 'map50_95',
    'params_m', 'flops_g', 'gpu_forward_median_ms', 'gpu_fps',
    'cpu_forward_median_ms', 'cpu_fps', 'imgsz', 'batch', 'device',
    'dataset', 'dataset_scope', 'checkpoint', 'checkpoint_sha256', 'status', 'measured_at_utc',
]


def forward_latency(core, torch, device: str, imgsz: int = 640) -> float:
    core.to(device).eval()
    image = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32, device=device)
    sync = torch.cuda.synchronize if device == 'cuda:0' else (lambda: None)
    values = []
    with torch.inference_mode():
        for _ in range(5):
            core(image)
        sync()
        for _ in range(30):
            sync()
            start = time.perf_counter()
            core(image)
            sync()
            values.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(values))


def main() -> int:
    require_local_venv()
    root = project_path('yolov12_v1_src', must_exist=True)
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for the transfer benchmark.')
    torch.set_num_threads(8)
    data_yaml = project_path('dataset/adv_tshirt_person_eval.yaml', must_exist=True)
    runtime_data_yaml = materialize_ultralytics_dataset_yaml(data_yaml)
    out_path = project_path('results/benchmark_yolov12_transfer.csv')
    rows = []
    for scale in SCALES:
        checkpoint = project_path(Path('weights') / f'yolov12{scale}.pt', must_exist=True)
        model = YOLO(str(checkpoint))
        core = model.model
        params_m = sum(p.numel() for p in core.parameters()) / 1_000_000.0
        try:
            flops_g = float(get_flops(core, imgsz=640))
            if flops_g <= 0:
                flops_g = None
        except Exception:
            flops_g = None
        gpu_ms = forward_latency(core, torch, 'cuda:0')
        cpu_ms = forward_latency(core, torch, 'cpu')
        core.to('cuda:0').eval()
        metrics = model.val(
            data=str(runtime_data_yaml), split='test', imgsz=640, batch=1, workers=0,
            device=0, classes=[0], conf=0.001, iou=0.7,
            plots=True, save_json=True,
            project=str(project_path('runs/adv_tshirt_transfer')),
            name=f'yolov12_{scale}', exist_ok=True, verbose=False,
        )
        box = metrics.box
        row = {
            'model_name': f'YOLOv12-{scale.upper()}', 'family': 'YOLOv12', 'scale': scale,
            'precision': float(box.mp), 'recall': float(box.mr), 'map50': float(box.map50), 'map50_95': float(box.map),
            'params_m': params_m, 'flops_g': flops_g,
            'gpu_forward_median_ms': gpu_ms, 'gpu_fps': 1000.0 / gpu_ms,
            'cpu_forward_median_ms': cpu_ms, 'cpu_fps': 1000.0 / cpu_ms,
            'imgsz': 640, 'batch': 1, 'device': 'NVIDIA CUDA GPU / CPU',
            'dataset': 'AdvT-shirt-1K', 'dataset_scope': 'Full corpus, zero-shot COCO-pretrained person detection; no training/tuning',
            'checkpoint': str(checkpoint.relative_to(PROJECT_ROOT)), 'checkpoint_sha256': sha256_file(checkpoint),
            'status': 'ok', 'measured_at_utc': utc_timestamp(),
        }
        rows.append(row)
        print(f"{row['model_name']}: P={row['precision']:.4f} R={row['recall']:.4f} mAP50={row['map50']:.4f} mAP50-95={row['map50_95']:.4f}")
        del metrics, model, core
        gc.collect()
        torch.cuda.empty_cache()
    with out_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} official YOLOv12 measurements to {out_path.relative_to(PROJECT_ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
