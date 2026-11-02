# Literature reference: official MS COCO results

This is a separate literature-context table, not a direct substitute for the project-dataset benchmark. The paper's implementation details specify a Tesla T4 GPU with TensorRT FP16 for Table I latency; FPS is derived as 1000/latency.

| Model | Dataset | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Params (M) | FLOPs (G) | Reported latency (ms) | Latency device/protocol | Derived FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | MS COCO | -- | -- | 52.600 | 37.400 | 3.20 | 8.70 | 1.77 | Tesla T4 GPU; TensorRT FP16 | 565.0 |
| YOLOv10n | MS COCO | -- | -- | 53.800 | 38.500 | 2.30 | 6.70 | 1.84 | Tesla T4 GPU; TensorRT FP16 | 543.5 |
| YOLO11n | MS COCO | -- | -- | 54.200 | 38.600 | 2.60 | 6.50 | 1.53 | Tesla T4 GPU; TensorRT FP16 | 653.6 |
| YOLOv12n | MS COCO | -- | -- | 56.000 | 40.100 | 2.60 | 6.50 | 1.83 | Tesla T4 GPU; TensorRT FP16 | 546.4 |
| YOLOv13n | MS COCO | -- | -- | 57.800 | 41.600 | 2.50 | 6.40 | 1.97 | Tesla T4 GPU; TensorRT FP16 | 507.6 |
| YOLOv8s | MS COCO | -- | -- | 61.800 | 45.000 | 11.20 | 28.60 | 2.33 | Tesla T4 GPU; TensorRT FP16 | 429.2 |
| YOLOv10s | MS COCO | -- | -- | 63.000 | 46.300 | 7.20 | 21.60 | 2.53 | Tesla T4 GPU; TensorRT FP16 | 395.3 |
| YOLO11s | MS COCO | -- | -- | 62.600 | 45.800 | 9.40 | 21.50 | 2.56 | Tesla T4 GPU; TensorRT FP16 | 390.6 |
| YOLOv12s | MS COCO | -- | -- | 64.200 | 47.100 | 9.30 | 21.40 | 2.82 | Tesla T4 GPU; TensorRT FP16 | 354.6 |
| YOLOv13s | MS COCO | -- | -- | 65.200 | 48.000 | 9.00 | 20.80 | 2.98 | Tesla T4 GPU; TensorRT FP16 | 335.6 |
| YOLOv8l | MS COCO | -- | -- | 69.800 | 53.000 | 43.70 | 165.20 | 8.13 | Tesla T4 GPU; TensorRT FP16 | 123.0 |
| YOLOv10l | MS COCO | -- | -- | 70.100 | 53.200 | 24.40 | 120.30 | 7.31 | Tesla T4 GPU; TensorRT FP16 | 136.8 |
| YOLO11l | MS COCO | -- | -- | 69.200 | 52.300 | 25.30 | 86.90 | 6.23 | Tesla T4 GPU; TensorRT FP16 | 160.5 |
| YOLOv12l | MS COCO | -- | -- | 70.000 | 53.000 | 26.40 | 88.90 | 7.10 | Tesla T4 GPU; TensorRT FP16 | 140.8 |
| YOLOv13l | MS COCO | -- | -- | 70.900 | 53.400 | 27.60 | 88.40 | 8.63 | Tesla T4 GPU; TensorRT FP16 | 115.9 |
| YOLOv8x | MS COCO | -- | -- | 71.000 | 54.000 | 68.20 | 257.80 | 12.83 | Tesla T4 GPU; TensorRT FP16 | 77.9 |
| YOLOv10x | MS COCO | -- | -- | 71.300 | 54.400 | 29.50 | 160.40 | 10.70 | Tesla T4 GPU; TensorRT FP16 | 93.5 |
| YOLO11x | MS COCO | -- | -- | 71.000 | 54.200 | 56.90 | 194.90 | 11.35 | Tesla T4 GPU; TensorRT FP16 | 88.1 |
| YOLOv12x | MS COCO | -- | -- | 71.100 | 54.400 | 59.10 | 199.00 | 12.46 | Tesla T4 GPU; TensorRT FP16 | 80.3 |
| YOLOv13x | MS COCO | -- | -- | 72.000 | 54.800 | 64.00 | 199.20 | 14.67 | Tesla T4 GPU; TensorRT FP16 | 68.2 |
