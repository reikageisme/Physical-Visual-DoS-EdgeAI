# YOLOv13 for Physical Adversarial Patch Localization: A Controlled AdvT-shirt-1K Benchmark

**Report date:** 2026-10-05  
**Study status:** locally trained and evaluated on an external, source-cited physical adversarial patch dataset. The project-specific NCKH dataset remains unavailable.

## Abstract

Physical adversarial patches can disrupt object detection and motivate patch localization as a robustness task. We fine-tuned YOLOv8, YOLOv10, YOLO11, YOLOv12, and YOLOv13 Nano/Small/Large detectors on a deterministic split derived from AdvT-shirt-1K, using identical input size, optimizer, augmentation, seed, and training schedule. The local mirror contains 1,123 images and 1,548 patch boxes; the derived train/validation/test split contains 785/169/169 images. Using validation mAP@0.5:0.95 for model selection, YOLOv12s was selected and reached **83.81% mAP@0.5:0.95** and **99.41% mAP@0.5** on the held-out test split. The result is limited to patch localization on this external proxy and is not a result on the original physical–visual DoS dataset.

## Experimental setup

- **Hardware:** Intel(R) Core(TM) i5-14600KF; 31.79 GiB RAM; NVIDIA GeForce RTX 5060 (7.96 GiB); CUDA 12.8.
- **Software:** Python 3.14.7, PyTorch 2.11.0+cu128, Ultralytics 8.3.63, CUDA build 12.8; isolated environment at `.venv/`.
- **Dataset:** AdvT-shirt-1K, source mirror at `dataset/external/adv_tshirt_1k/`; license listed as Apache-2.0 by the dataset card. Class map: `{'0': 'patch'}`. Image counts: train/val/test = 785/169/169; patch instances = {'train': {'0': 1106}, 'val': {'0': 220}, 'test': {'0': 222}}. Native dimensions range from 856–2056 px wide × 1044–1598 px high; all models use 416 × 416 network input.
- **Split:** seed 42, 70/15/15 requested, stratified within the five source folders. This split is derived because the source does not provide train/validation/test partitions for this mirror.
- **Training:** pretrained initialization; 50 maximum epochs with patience 10; image size 416; batch 4; SGD; initial learning rate 0.01; linear LR decay to factor 0.01 with 2.0 warmup epochs; weight decay 0.0005; seed 42; deterministic mode. Shared augmentation: HSV (0.015/0.7/0.4), translation 0.1, scale 0.5, mosaic 1.0, horizontal flip 0.5; MixUp and CopyPaste disabled.
- **Measurement:** validation and test metrics were obtained on the same held-out split for every checkpoint. CPU/GPU latency is the evaluator-reported inference time per image at batch 4; FPS is computed as $1000/latency$. Parameters and FLOPs use the installed Ultralytics profiler at 416 pixels.
- **Path isolation:** the canonical `dataset/data.yaml` uses `path: ./dataset`; external data, images, labels, runtime data YAMLs, and Hugging Face caches stay below `dataset/`. `YOLO_CONFIG_DIR=./.config`; Ultralytics dataset/run/weight settings resolve to project-local `dataset/`, `runs/`, and `weights/`; pip, Torch, and temporary caches are under `.cache/`. The dataset tree contains no symlink/junction entries.

## Comprehensive benchmark

The table reports held-out **test** precision, recall, mAP, parameter count, FLOPs, CPU/GPU latency, and derived FPS. Model selection and same-scale baseline selection use validation mAP@0.5:0.95; the test set is reserved for final reporting. The machine-readable source is `results/adv_tshirt_patch_benchmark.csv`.

| Model | P (%) | R (%) | mAP50 (%) | mAP50:95 (%) | Params (M) | FLOPs (G) | CPU ms | CPU FPS | GPU ms | GPU FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11l | 98.81 | 98.20 | 99.45 | 83.91 | 25.31 | 36.87 | 108.12 | 9.25 | 5.09 | 196.28 |
| YOLO11n | 97.95 | 99.10 | 99.37 | 83.65 | 2.59 | 2.72 | 11.87 | 84.23 | 1.69 | 590.53 |
| YOLO11s | 99.10 | 98.65 | 99.47 | 82.18 | 9.43 | 9.10 | 30.70 | 32.57 | 2.34 | 427.76 |
| YOLOv10l | 96.42 | 97.06 | 99.04 | 82.68 | 25.77 | 53.74 | 140.35 | 7.13 | 6.30 | 158.81 |
| YOLOv10n | 95.77 | 98.65 | 99.32 | 83.02 | 2.71 | 3.55 | 13.68 | 73.09 | 2.01 | 497.88 |
| YOLOv10s | 96.03 | 96.85 | 99.14 | 81.65 | 8.07 | 10.46 | 33.61 | 29.75 | 2.54 | 393.68 |
| YOLOv12l | 98.84 | 98.20 | 99.45 | 84.25 | 26.92 | 38.23 | 124.92 | 8.01 | 6.74 | 148.46 |
| YOLOv12n | 97.24 | 99.55 | 99.41 | 82.52 | 2.55 | 2.73 | 13.35 | 74.93 | 2.49 | 402.31 |
| YOLOv12s | 97.76 | 98.41 | 99.41 | 83.81 | 9.22 | 9.07 | 33.61 | 29.75 | 2.91 | 343.73 |
| YOLOv13l | 98.21 | 98.61 | 99.44 | 83.84 | 27.57 | 37.61 | 144.76 | 6.91 | 8.53 | 117.25 |
| YOLOv13n | 99.52 | 98.65 | 99.49 | 79.49 | 2.46 | 2.69 | 16.89 | 59.19 | 3.08 | 324.38 |
| YOLOv13s | 98.21 | 99.10 | 99.45 | 83.96 | 9.55 | 9.15 | 38.95 | 25.67 | 3.16 | 316.55 |
| YOLOv8l | 98.20 | 98.10 | 99.42 | 83.94 | 43.63 | 69.88 | 165.67 | 6.04 | 6.42 | 155.88 |
| YOLOv8n | 98.18 | 99.10 | 99.47 | 83.70 | 3.01 | 3.46 | 11.87 | 84.27 | 1.30 | 769.73 |
| YOLOv8s | 96.63 | 100.00 | 99.33 | 83.72 | 11.14 | 12.10 | 34.81 | 28.73 | 2.12 | 472.52 |

```latex
% Requires \usepackage{booktabs,graphicx}
\begin{table*}[t]
\centering
\scriptsize
\caption{Fine-tuned patch localization on the derived AdvT-shirt-1K test split.}
\label{tab:advtshirt-patch-benchmark}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrrr}
\toprule
Model & P (\%) & R (\%) & mAP50 (\%) & mAP50:95 (\%) & Params (M) & FLOPs (G) & CPU ms & CPU FPS & GPU ms & GPU FPS \\
\midrule
YOLO11l & 98.81 & 98.20 & 99.45 & 83.91 & 25.31 & 36.87 & 108.12 & 9.25 & 5.09 & 196.28 \\
YOLO11n & 97.95 & 99.10 & 99.37 & 83.65 & 2.59 & 2.72 & 11.87 & 84.23 & 1.69 & 590.53 \\
YOLO11s & 99.10 & 98.65 & 99.47 & 82.18 & 9.43 & 9.10 & 30.70 & 32.57 & 2.34 & 427.76 \\
YOLOv10l & 96.42 & 97.06 & 99.04 & 82.68 & 25.77 & 53.74 & 140.35 & 7.13 & 6.30 & 158.81 \\
YOLOv10n & 95.77 & 98.65 & 99.32 & 83.02 & 2.71 & 3.55 & 13.68 & 73.09 & 2.01 & 497.88 \\
YOLOv10s & 96.03 & 96.85 & 99.14 & 81.65 & 8.07 & 10.46 & 33.61 & 29.75 & 2.54 & 393.68 \\
YOLOv12l & 98.84 & 98.20 & 99.45 & 84.25 & 26.92 & 38.23 & 124.92 & 8.01 & 6.74 & 148.46 \\
YOLOv12n & 97.24 & 99.55 & 99.41 & 82.52 & 2.55 & 2.73 & 13.35 & 74.93 & 2.49 & 402.31 \\
YOLOv12s & 97.76 & 98.41 & 99.41 & 83.81 & 9.22 & 9.07 & 33.61 & 29.75 & 2.91 & 343.73 \\
YOLOv13l & 98.21 & 98.61 & 99.44 & 83.84 & 27.57 & 37.61 & 144.76 & 6.91 & 8.53 & 117.25 \\
YOLOv13n & 99.52 & 98.65 & 99.49 & 79.49 & 2.46 & 2.69 & 16.89 & 59.19 & 3.08 & 324.38 \\
YOLOv13s & 98.21 & 99.10 & 99.45 & 83.96 & 9.55 & 9.15 & 38.95 & 25.67 & 3.16 & 316.55 \\
YOLOv8l & 98.20 & 98.10 & 99.42 & 83.94 & 43.63 & 69.88 & 165.67 & 6.04 & 6.42 & 155.88 \\
YOLOv8n & 98.18 & 99.10 & 99.47 & 83.70 & 3.01 & 3.46 & 11.87 & 84.27 & 1.30 & 769.73 \\
YOLOv8s & 96.63 & 100.00 & 99.33 & 83.72 & 11.14 & 12.10 & 34.81 & 28.73 & 2.12 & 472.52 \\
\bottomrule
\end{tabular}}
\end{table*}
```

## Efficiency trade-off on the local benchmark

The following figures plot the measured validation/test rows on CPU and GPU using the local Ultralytics validator at batch 4. They are hardware- and runtime-specific; they are not TensorRT measurements or an edge-device deployment test.

![Local test and validation mAP versus inference latency](adv_tshirt_patch_results/tradeoff_map_latency_local.png)

![Local test and validation mAP versus FLOPs](adv_tshirt_patch_results/tradeoff_map_flops_local.png)

At Nano scale, YOLOv13-N is 4.21 percentage points below the validation-selected YOLOv8-N baseline in test mAP@0.5:0.95 and has higher measured GPU latency (3.08 vs 1.30 ms/image). At Small scale, YOLOv13-S is 0.16 points above the validation-selected YOLOv12-S baseline, with latency of 3.16 vs 2.91 ms/image. At Large scale, YOLOv13-L is 0.41 points below YOLOv12-L and slower (8.53 vs 6.74 ms/image). These results do not show uniform YOLOv13 superiority on this proxy.

## Published benchmark context (separate dataset and protocol)

The YOLOv13 paper reports its MS COCO results and Table I latency using a Tesla T4 GPU with TensorRT FP16. This reference table is not directly comparable with the local AdvT-shirt proxy results, which use a different dataset, GPU, runtime and validator. Precision/recall are not reported in that paper table, so they are shown as unavailable. The paper motivates HyperACE for high-order cross-location/cross-scale feature relations, FullPAD for distributing enhanced representations through the pipeline, and depthwise-separable DS-C3k2 blocks for reducing model cost [Lei et al., YOLOv13](https://arxiv.org/abs/2506.17733).

| Model | Dataset | P (%) | R (%) | mAP50 (%) | mAP50:95 (%) | Params (M) | FLOPs (G) | Latency (ms) | Latency protocol | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | MS COCO | -- | -- | 52.60 | 37.40 | 3.20 | 8.70 | 1.77 | Tesla T4 GPU; TensorRT FP16 | 565.0 |
| YOLOv10n | MS COCO | -- | -- | 53.80 | 38.50 | 2.30 | 6.70 | 1.84 | Tesla T4 GPU; TensorRT FP16 | 543.5 |
| YOLO11n | MS COCO | -- | -- | 54.20 | 38.60 | 2.60 | 6.50 | 1.53 | Tesla T4 GPU; TensorRT FP16 | 653.6 |
| YOLOv12n | MS COCO | -- | -- | 56.00 | 40.10 | 2.60 | 6.50 | 1.83 | Tesla T4 GPU; TensorRT FP16 | 546.4 |
| YOLOv13n | MS COCO | -- | -- | 57.80 | 41.60 | 2.50 | 6.40 | 1.97 | Tesla T4 GPU; TensorRT FP16 | 507.6 |
| YOLOv8s | MS COCO | -- | -- | 61.80 | 45.00 | 11.20 | 28.60 | 2.33 | Tesla T4 GPU; TensorRT FP16 | 429.2 |
| YOLOv10s | MS COCO | -- | -- | 63.00 | 46.30 | 7.20 | 21.60 | 2.53 | Tesla T4 GPU; TensorRT FP16 | 395.3 |
| YOLO11s | MS COCO | -- | -- | 62.60 | 45.80 | 9.40 | 21.50 | 2.56 | Tesla T4 GPU; TensorRT FP16 | 390.6 |
| YOLOv12s | MS COCO | -- | -- | 64.20 | 47.10 | 9.30 | 21.40 | 2.82 | Tesla T4 GPU; TensorRT FP16 | 354.6 |
| YOLOv13s | MS COCO | -- | -- | 65.20 | 48.00 | 9.00 | 20.80 | 2.98 | Tesla T4 GPU; TensorRT FP16 | 335.6 |
| YOLOv8l | MS COCO | -- | -- | 69.80 | 53.00 | 43.70 | 165.20 | 8.13 | Tesla T4 GPU; TensorRT FP16 | 123.0 |
| YOLOv10l | MS COCO | -- | -- | 70.10 | 53.20 | 24.40 | 120.30 | 7.31 | Tesla T4 GPU; TensorRT FP16 | 136.8 |
| YOLO11l | MS COCO | -- | -- | 69.20 | 52.30 | 25.30 | 86.90 | 6.23 | Tesla T4 GPU; TensorRT FP16 | 160.5 |
| YOLOv12l | MS COCO | -- | -- | 70.00 | 53.00 | 26.40 | 88.90 | 7.10 | Tesla T4 GPU; TensorRT FP16 | 140.8 |
| YOLOv13l | MS COCO | -- | -- | 70.90 | 53.40 | 27.60 | 88.40 | 8.63 | Tesla T4 GPU; TensorRT FP16 | 115.9 |
| YOLOv8x | MS COCO | -- | -- | 71.00 | 54.00 | 68.20 | 257.80 | 12.83 | Tesla T4 GPU; TensorRT FP16 | 77.9 |
| YOLOv10x | MS COCO | -- | -- | 71.30 | 54.40 | 29.50 | 160.40 | 10.70 | Tesla T4 GPU; TensorRT FP16 | 93.5 |
| YOLO11x | MS COCO | -- | -- | 71.00 | 54.20 | 56.90 | 194.90 | 11.35 | Tesla T4 GPU; TensorRT FP16 | 88.1 |
| YOLOv12x | MS COCO | -- | -- | 71.10 | 54.40 | 59.10 | 199.00 | 12.46 | Tesla T4 GPU; TensorRT FP16 | 80.3 |
| YOLOv13x | MS COCO | -- | -- | 72.00 | 54.80 | 64.00 | 199.20 | 14.67 | Tesla T4 GPU; TensorRT FP16 | 68.2 |

## Ablation study

The ablations use YOLOv13-N, the same pretrained initialization, split, seed, schedule, and augmentation. Values are validation metrics; deltas compare each variant with the full model.

| Variant | P (%) | R (%) | mAP50 (%) | mAP50:95 (%) | Params (M) | FLOPs (G) | Latency (ms) | ΔmAP50:95 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv13-n-full | 98.47 | 97.27 | 99.29 | 81.07 | 2.45 | 2.62 | 5.44 | +0.00 |
| YOLOv13-n-no-hyperace | 98.88 | 98.64 | 99.42 | 80.67 | 2.36 | 2.56 | 6.21 | -0.39 |
| YOLOv13-n-no-fullpad | 97.31 | 98.50 | 99.25 | 83.19 | 2.45 | 2.62 | 6.38 | +2.13 |
| YOLOv13-n-no-dsc3k2 | 98.07 | 99.09 | 99.43 | 83.57 | 5.35 | 5.65 | 5.97 | +2.50 |

**Interpretation.** Validation selected YOLOv12s (val mAP@0.5:0.95 84.78%); its held-out test mAP@0.5:0.95 is 83.81%. Relative to the full model, the measured validation changes were: YOLOv13-n-no-hyperace: ΔmAP@0.5:0.95 -0.39 pp, Δlatency +0.77 ms, Δparameters -0.09 M; YOLOv13-n-no-fullpad: ΔmAP@0.5:0.95 +2.13 pp, Δlatency +0.93 ms, Δparameters -0.00 M; YOLOv13-n-no-dsc3k2: ΔmAP@0.5:0.95 +2.50 pp, Δlatency +0.53 ms, Δparameters +2.90 M. In this proxy, removing HyperACE slightly reduced mAP@0.5:0.95; removing FullPAD increased it but also increased latency; replacing DS-C3k2 increased both accuracy and model cost. This single-seed ablation is descriptive evidence, not proof of universal causal effects.

## Discussion and conclusion

- YOLOv13l: -0.41 pp test mAP@0.5:0.95 vs validation-selected same-scale baseline YOLOv12l
- YOLOv13n: -4.21 pp test mAP@0.5:0.95 vs validation-selected same-scale baseline YOLOv8n
- YOLOv13s: +0.16 pp test mAP@0.5:0.95 vs validation-selected same-scale baseline YOLOv12s
- HyperACE models adaptive high-order interactions among spatial feature locations; FullPAD routes feature information across backbone/neck stages; DS-C3k2 uses depthwise-separable computation to reduce cost. The local ablation table is the evidence for this dataset and scale. It must be read as a single-seed controlled comparison; a component is supported only when its measured delta is consistent with the architecture-level hypothesis.
- The exact NCKH task split remains empty. These results concern a separately derived external patch-localization proxy; they do not establish a gain for physical–visual DoS, NMS overload, or unseen patch designs. The source has no official split, the derived split is image-level within source folders, and models were trained with one random seed.

YOLOv13's architectural motivation is that HyperACE can model multi-location, higher-order feature relations; FullPAD distributes information across the pipeline; DS-C3k2 targets lower convolutional cost. The measured ablation—not the architectural description—determines whether these components improved this patch dataset. Any accuracy gain must be weighed against measured latency, FLOPs, and parameters before an edge deployment claim is made.

## Thảo luận và kết luận (Tiếng Việt)

Theo mAP@0.5:0.95 trên tập validation, cấu hình được chọn là **YOLOv12s**; kết quả kiểm tra độc lập của cấu hình này đạt mAP@0.5:0.95 **83.81%** và mAP@0.5 **99.41%**. Bảng đối sánh báo cáo thêm độ trễ CPU/GPU, FPS, số tham số và FLOPs để đánh giá chi phí suy luận cùng với độ chính xác.

HyperACE được thiết kế để tổng hợp quan hệ giữa nhiều vị trí đặc trưng theo cấu trúc bậc cao; FullPAD nhằm truyền và kết hợp thông tin xuyên suốt backbone/head; DS-C3k2 hướng tới giảm phép tính tích chập nhờ phép tích chập tách biệt theo chiều sâu. Đây là cơ sở kiến trúc, còn mức đóng góp thực nghiệm phải được xác định bằng các hàng ablation đo trên cùng dữ liệu. Kết quả một seed chỉ là bằng chứng ban đầu, chưa đủ để khẳng định ưu thế có ý nghĩa thống kê.

Chênh lệch mAP@0.5:0.95 trên tập test so với baseline cùng quy mô được chọn bằng validation: YOLOv13l: -0.41 điểm phần trăm so với YOLOv12l cùng quy mô; YOLOv13n: -4.21 điểm phần trăm so với YOLOv8n cùng quy mô; YOLOv13s: +0.16 điểm phần trăm so với YOLOv12s cùng quy mô.

Các số liệu hiện tại thuộc bộ phát hiện vị trí miếng vá AdvT-shirt-1K được chia lại, không phải tập dữ liệu NCKH gốc. Vì vậy chúng không chứng minh khả năng chống tấn công gây quá tải NMS hoặc tổng quát hóa sang thiết kế miếng vá mới. Cần bổ sung tập đề tài đã gán nhãn, chia test theo thiết kế/người chụp độc lập, chạy nhiều seed và đo trên thiết bị biên thực tế trước khi kết luận về triển khai.

## Scope and sources

The project's canonical `dataset/data.yaml` remains not ready: its original NCKH train/validation/test folders have no images or labels and the class name is still a placeholder. Accordingly, this report does not claim physical–visual DoS/NMS-overload efficacy. The current benchmark is an external patch-localization proxy. Its image-level split may contain correlated or near-duplicate views across partitions, and the single-seed protocol does not estimate variance. AdvT-shirt-1K reports 1,131 images, while the available mirror contains 1,123; the discrepancy is documented in `results/external_dataset_sources.md`.

- [AdvT-shirt-1K dataset card and Apache-2.0 license](https://huggingface.co/datasets/xingjunm/Adv-T-shirt)
- [AdvT-shirt-1K authors' repository](https://github.com/Wwangb/AdvT-shirt-1K)
- [BadPatch / AdvT-shirt paper](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Wang_BadPatch_Diffusion-Based_Generation_of_Physical_Adversarial_Patches_ICCVW_2025_paper.html)
- [YOLOv13 paper](https://arxiv.org/abs/2506.17733)

**Limitations:** The exact NCKH task split remains empty. These results concern a separately derived external patch-localization proxy; they do not establish a gain for physical–visual DoS, NMS overload, or unseen patch designs. The source has no official split, the derived split is image-level within source folders, and models were trained with one random seed.
