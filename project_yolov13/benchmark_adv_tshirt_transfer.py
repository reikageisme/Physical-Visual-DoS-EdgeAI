from __future__ import annotations
import argparse
import csv
import gc
import json
import statistics
import shutil
import time
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, materialize_ultralytics_dataset_yaml, project_path, require_local_venv, sha256_file, utc_timestamp, write_json

configure_project_environment()

SCALES = ('n', 's', 'l')
SPECS = {
    'YOLOv8': {s: f'yolov8{s}.pt' for s in SCALES},
    'YOLOv10': {s: f'yolov10{s}.pt' for s in SCALES},
    'YOLO11': {s: f'yolo11{s}.pt' for s in SCALES},
    'YOLOv13': {s: f'weights/yolov13{s}.pt' for s in SCALES},
}
FIELDS = [
    'model_name', 'family', 'scale', 'precision', 'recall', 'map50', 'map50_95',
    'params_m', 'flops_g', 'gpu_forward_median_ms', 'gpu_fps',
    'cpu_forward_median_ms', 'cpu_fps', 'imgsz', 'batch', 'device',
    'dataset', 'dataset_scope', 'checkpoint', 'checkpoint_sha256', 'status', 'measured_at_utc',
]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def forward_latency(core, torch, device: str, imgsz: int = 640) -> float:
    core.to(device).eval()
    sample = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32, device=device)
    sync = torch.cuda.synchronize if device == 'cuda:0' else (lambda: None)
    with torch.inference_mode():
        for _ in range(5):
            core(sample)
        sync()
        values: list[float] = []
        for _ in range(30):
            sync()
            start = time.perf_counter()
            core(sample)
            sync()
            values.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument('--scales', nargs='+', choices=SCALES, default=list(SCALES))
    parser.add_argument('--output', default='results/benchmark_adv_tshirt_transfer.csv')
    args = parser.parse_args()
    require_local_venv()
    import torch
    import numpy as np
    if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
        np.trapz = np.trapezoid
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required for the planned full-corpus transfer benchmark.')
    torch.set_num_threads(8)
    data_yaml = project_path('dataset/adv_tshirt_person_eval.yaml', must_exist=True)
    runtime_data_yaml = materialize_ultralytics_dataset_yaml(data_yaml)
    data_cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8-sig'))
    test_dir = project_path('dataset/external/adv_tshirt_person_eval/images/test', must_exist=True)
    if not any(test_dir.iterdir()):
        raise RuntimeError('Prepared external evaluation image directory is empty.')

    out_csv = project_path(args.output)
    partial = out_csv.with_name(out_csv.stem + '.partial.csv')
    rows: list[dict] = []
    failures: list[dict] = []
    for family in args.families:
        scale_map = SPECS[family]
        for scale in args.scales:
            asset = scale_map[scale]
            model_name = f'{family}-{scale.upper()}'
            supplied = project_path(asset) if asset.startswith('weights/') else None
            try:
                model = YOLO(str(supplied)) if supplied else YOLO(asset)
                if supplied is None:
                    loaded_path = Path(getattr(model, 'ckpt_path', asset)).resolve()
                    try:
                        loaded_path.relative_to(PROJECT_ROOT.resolve())
                    except ValueError as exc:
                        raise RuntimeError(f'Weight cache escaped project: {loaded_path}') from exc
                    if not loaded_path.is_file():
                        candidate = Path(asset)
                        if candidate.is_file():
                            loaded_path = candidate.resolve()
                        else:
                            raise FileNotFoundError(f'Could not resolve local checkpoint for {asset}')
                    checkpoint = project_path(Path('weights') / Path(asset).name)
                    if loaded_path != checkpoint.resolve() and not checkpoint.exists():
                        shutil.copy2(loaded_path, checkpoint)
                    elif loaded_path == checkpoint.resolve():
                        checkpoint = loaded_path
                    if not checkpoint.exists():
                        checkpoint = loaded_path
                else:
                    checkpoint = supplied
                if not checkpoint.is_file():
                    raise FileNotFoundError(f'Checkpoint not present inside project: {checkpoint}')

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
                # Restore model to the GPU before validation; the validator evaluates the same single COCO class.
                core.to('cuda:0').eval()
                metrics = model.val(
                    data=str(runtime_data_yaml), split='test', imgsz=640, batch=1, workers=0,
                    device=0, classes=[0], conf=0.001, iou=0.7,
                    plots=True, save_json=True,
                    project=str(project_path('runs/adv_tshirt_transfer')),
                    name=f'{family.lower()}_{scale}', exist_ok=True,
                    verbose=False,
                )
                box = metrics.box
                row = {
                    'model_name': model_name, 'family': family, 'scale': scale,
                    'precision': float(box.mp), 'recall': float(box.mr),
                    'map50': float(box.map50), 'map50_95': float(box.map),
                    'params_m': params_m, 'flops_g': flops_g,
                    'gpu_forward_median_ms': gpu_ms, 'gpu_fps': 1000.0 / gpu_ms,
                    'cpu_forward_median_ms': cpu_ms, 'cpu_fps': 1000.0 / cpu_ms,
                    'imgsz': 640, 'batch': 1, 'device': 'NVIDIA CUDA GPU / CPU',
                    'dataset': 'AdvT-shirt-1K',
                    'dataset_scope': 'Full corpus, zero-shot COCO-pretrained person detection; no training/tuning',
                    'checkpoint': str(checkpoint.relative_to(PROJECT_ROOT)),
                    'checkpoint_sha256': sha256_file(checkpoint),
                    'status': 'ok', 'measured_at_utc': utc_timestamp(),
                }
                rows.append(row)
                write_rows(partial, rows)
                print(f'{model_name}: P={row["precision"]:.4f} R={row["recall"]:.4f} mAP50={row["map50"]:.4f} mAP50-95={row["map50_95"]:.4f}')
                del metrics, model, core
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:
                failures.append({'model': model_name, 'asset': asset, 'error': f'{type(exc).__name__}: {exc}'})
                row = {field: '' for field in FIELDS}
                row.update({'model_name': model_name, 'family': family, 'scale': scale, 'dataset': 'AdvT-shirt-1K', 'status': 'failed', 'measured_at_utc': utc_timestamp()})
                rows.append(row)
                write_rows(partial, rows)
                print(f'{model_name}: FAILED ({type(exc).__name__}: {exc})')
                gc.collect()
                torch.cuda.empty_cache()

    write_rows(out_csv, rows)
    partial.unlink(missing_ok=True)
    manifest = {
        'measured_at_utc': utc_timestamp(),
        'dataset_manifest': 'results/external_transfer_dataset_manifest.json',
        'data_yaml': 'dataset/adv_tshirt_person_eval.yaml',
        'model_count': len(rows),
        'successful_models': sum(row['status'] == 'ok' for row in rows),
        'failed_models': failures,
        'evaluation': 'Ultralytics validation on full corpus, test split path; class 0 only (person), 640x640, batch=1, GPU, FP32.',
        'latency': 'Five warmups and 30 model-forward calls on a zero tensor, batch=1, 640x640, FP32; CUDA synchronization; median reported; preprocessing/postprocessing excluded.',
        'output_csv': args.output,
    }
    write_json(project_path('results/external_transfer_benchmark_manifest.json'), manifest)
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())




