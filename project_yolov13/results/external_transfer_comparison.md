# AdvT-shirt-1K zero-shot transfer benchmark

COCO-pretrained person detectors evaluated without fine-tuning on the full available AdvT-shirt-1K mirror (1,123 images; 4,403 person boxes). This external transfer set is distinct from the missing NCKH task dataset.

Latency is median model-forward time from 5 warm-ups and 30 timed FP32 calls at 640x640, batch 1; it excludes preprocessing, postprocessing, image transfer, and model loading. The exact same protocol was used for CPU and GPU. Metrics were computed on the complete corpus using the COCO person class.

| Model | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | FLOPs (G) | GPU latency (ms) | GPU FPS | CPU latency (ms) | CPU FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8-N | 88.69 | 79.59 | 89.17 | 73.97 | 3.157 | 8.86 | 4.43 | 225.9 | 39.65 | 25.2 |
| YOLOv10-N | 89.64 | 76.74 | 88.28 | 74.51 | 2.776 | 8.74 | 9.34 | 107.0 | 39.78 | 25.1 |
| YOLO11-N | 88.76 | 79.31 | 89.35 | 74.44 | 2.624 | 6.61 | 16.42 | 60.9 | 54.28 | 18.4 |
| YOLOv12-N | 91.27 | 79.35 | 90.04 | 76.20 | 2.603 | 6.65 | 27.04 | 37.0 | 73.17 | 13.7 |
| YOLOv13-N | 89.41 | 79.65 | 89.45 | 75.61 | 2.494 | 6.55 | 32.44 | 30.8 | 85.89 | 11.6 |
| YOLOv8-S | 90.77 | 86.04 | 93.52 | 82.16 | 11.167 | 28.81 | 11.46 | 87.3 | 133.35 | 7.5 |
| YOLOv10-S | 89.74 | 86.40 | 93.25 | 82.88 | 8.128 | 25.11 | 22.12 | 45.2 | 135.12 | 7.4 |
| YOLO11-S | 91.21 | 87.45 | 94.10 | 83.15 | 9.459 | 21.71 | 16.35 | 61.2 | 112.02 | 8.9 |
| YOLOv12-S | 91.68 | 88.14 | 94.43 | 84.12 | 9.286 | 21.69 | 25.90 | 38.6 | 142.92 | 7.0 |
| YOLOv13-S | 91.28 | 86.87 | 93.90 | 83.34 | 9.056 | 21.17 | 13.59 | 73.6 | 99.79 | 10.0 |
| YOLOv8-L | 91.50 | 92.61 | 96.44 | 88.69 | 43.692 | 165.73 | 19.36 | 51.7 | 664.53 | 1.5 |
| YOLOv10-L | 92.97 | 90.70 | 96.01 | 88.59 | 25.889 | 127.87 | 28.90 | 34.6 | 505.88 | 2.0 |
| YOLO11-L | 91.65 | 91.66 | 95.82 | 87.84 | 25.372 | 87.61 | 36.99 | 27.0 | 385.73 | 2.6 |
| YOLOv12-L | 92.35 | 91.10 | 95.70 | 87.97 | 26.455 | 89.76 | 51.62 | 19.4 | 565.68 | 1.8 |
| YOLOv13-L | 92.75 | 90.98 | 95.72 | 87.57 | 27.628 | 89.36 | 67.94 | 14.7 | 578.75 | 1.7 |

Models evaluated: 15. Full per-model hashes and run metadata are in `results/benchmark_adv_tshirt_transfer.csv`.
