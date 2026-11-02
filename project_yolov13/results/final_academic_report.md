# YOLOv13 for Physical–Visual DoS at the Edge: Published Ablations and External Zero-Shot Transfer

**Research report — 4 October 2026**  
**Status:** external transfer benchmark completed; project-specific training and causal ablation remain pending due to missing labeled splits.

## Execution status and interpretation

The canonical project train/validation/test directories contain zero images and zero labels; the dataset YAML still has the placeholder class `object`. A fresh audit on 4 October 2026 confirmed this state: `results/dataset_audit_current.json` records 0/0/0 split images, 0 label files and no malformed boxes because no task labels are present. The separate root-level `dataset/` contains 13 raster images and one SVG but no annotations or split metadata. Running `.venv/Scripts/python.exe train_yolov13.py --scale n --name task_dataset_preflight` stopped in dataset validation with “No images found in the configured train split,” before checkpoint download or training run creation. Therefore no project-specific checkpoint, accuracy score, validation/test result, or local causal ablation exists. These values remain unavailable and are not fabricated.

A supplementary external zero-shot benchmark has now been measured on the full available AdvT-shirt-1K mirror: 1,123 images with 4,403 person boxes, evaluated across YOLOv8/10/11/12/13 N/S/L checkpoints. This external corpus is relevant to adversarial-clothing transfer, but it is not the exact NCKH task dataset, has no prescribed split for this protocol, and was not used to fine-tune or select a model. Published MS COCO comparisons and ablations are labeled as literature context; local external measurements are labeled as zero-shot transfer results.

The original project's COCO128 diagnostic candidate was not substituted: it has no fixed train/validation/test split, two images without label files, and possible overlap with pretraining. The external dataset source/license/retrieval audit is in `results/external_dataset_sources.md`.

> **Operational scope deviation.** A one-off YOLOv13/YOLOv12 diagnostic import initialized small Ultralytics settings JSON files under the Windows user AppData configuration directory before the project-local cache setting was applied. Subsequent pipeline runs set `YOLO_CONFIG_DIR` and all experiment data, models, checkpoints, logs, plots, and reports under `project_yolov13/`. A cleanup attempt for the host-level settings artifact was rejected by the local file policy; strict “writes only inside the project” confinement was therefore not fully met for those two settings files.

## Title and abstracts

### English abstract

We evaluate YOLOv13 for physical–visual denial-of-service (DoS) detection while separating task-specific evidence from an external transfer proxy. The project's labeled train/validation/test splits are empty, so fine-tuning, project accuracy measurement and causal architectural ablation cannot be completed. As a supplementary zero-shot experiment, we evaluate COCO-pretrained YOLOv8/10/11/12/13 nano, small and large detectors on the complete available AdvT-shirt-1K mirror (1,123 images, 4,403 person boxes), without fine-tuning. YOLOv13 obtains person mAP@0.5:0.95 of 75.61/83.34/87.57% at N/S/L. This is -0.59/-0.78/-0.40 points relative to YOLOv12; YOLOv13 is not the best model at any scale on this corpus. The results therefore provide no empirical basis for claiming HyperACE-driven improvement in the target task. Published COCO ablations offer architectural motivation but remain separate evidence. The report provides the full benchmark, model hashes, latency/FLOP protocol and data sources while identifying the missing project dataset as the blocker to task-specific scientific conclusions.

### Tóm tắt tiếng Việt

Nghiên cứu đánh giá YOLOv13 cho bài toán từ chối dịch vụ vật lý–thị giác (physical–visual DoS), đồng thời tách biệt bằng chứng trên dữ liệu đề tài với phép đo chuyển miền bên ngoài. Các thư mục train/validation/test có nhãn của đề tài hiện trống nên chưa thể fine-tune, đo độ chính xác trên bài toán mục tiêu hoặc làm ablation nhân quả. Phép thử bổ sung zero-shot dùng các mô hình COCO-pretrained YOLOv8/10/11/12/13 biến thể nano, small và large trên toàn bộ bản dữ liệu AdvT-shirt-1K hiện có (1.123 ảnh, 4.403 hộp người), không huấn luyện thêm. YOLOv13 đạt person mAP@0.5:0.95 lần lượt 75.61/83.34/87.57% cho N/S/L; chênh lệch so với YOLOv12 là -0.59/-0.78/-0.40 điểm. YOLOv13 không dẫn đầu ở bất kỳ scale nào trên tập ngoài này, do đó kết quả không chứng minh cải thiện nhờ HyperACE trong bài toán mục tiêu. Ablation COCO đã công bố chỉ cung cấp cơ sở kiến trúc tham khảo. Báo cáo cung cấp benchmark đầy đủ, hash trọng số, quy trình đo latency/FLOPs và nguồn dữ liệu, đồng thời xác định thiếu dataset đề tài là hạn chế ngăn cản kết luận khoa học đặc thù.

## 1. Introduction and research question

The project asks whether YOLOv13 can improve object detection for a physical–visual DoS/edge-AI application. Earlier convolutional detectors aggregate local features, while pairwise attention is limited in representing global multi-to-multi interactions. YOLOv13 introduces adaptive hypergraph correlation modeling, Full-Pipeline Aggregation-and-Distribution (FullPAD), and depthwise-separable feature blocks. The original work tests these ideas on MS COCO; it does not establish performance on this project's dataset. [Lei et al., 2025](https://arxiv.org/pdf/2506.17733)

The research question is: **under the same project dataset, split, input size, training schedule, augmentation, and evaluation hardware, does locally fine-tuned YOLOv13 improve detection quality enough to justify its latency and resource cost relative to YOLOv8/10/11/12?** This remains unanswered because project data are absent.

## 2. Experimental setup

### 2.1 Hardware and software

| Item | Observed setup |
|---|---|
| Operating system | Windows 11 |
| Python environment | project_yolov13/.venv; Python 3.14.7 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| CUDA | PyTorch CUDA runtime 12.8; CUDA available |
| GPU | NVIDIA GeForce RTX 5060; 8,151 MiB reported memory; driver 610.88 |
| YOLOv13 implementation | iMoonLab source checkout, base revision 73289949533efac82bb5f72ec19b746618656bd2; local ablation-support patch hashes are recorded in results/source_patch_manifest.json |
| Dependency check | pip check: no broken requirements; pipeline Python files compile |
| Attention acceleration | FlashAttention wheel not installed; upstream PyTorch scaled-dot-product-attention fallback is available |

setup_env.bat creates/activates the project venv, points pip, temporary, Ultralytics, Torch, Matplotlib and Hugging Face caches into .cache/, installs CUDA-enabled PyTorch and the upstream source, and records environment/source metadata under results/. HyperACE, FullPAD and DS-C3k2 are implemented in the source checkout, not separately installed pip modules. A project-local, disabled-by-default patch adds ablation switches; standard inference behavior is unchanged. FlashAttention is an optional acceleration; the upstream code falls back to PyTorch attention. Four released checkpoints were loaded and parameter counts recorded in weights/manifest.json.

| Released checkpoint | Parameters loaded |
|---|---:|
| YOLOv13-N | 2.494 M |
| YOLOv13-S | 9.056 M |
| YOLOv13-L | 27.628 M |
| YOLOv13-X | 64.058 M |

These are pretrained inference checkpoints, not fine-tuned project weights.

### 2.2 Dataset availability and splits

| Split | Images found | Labels found | Ratio / dimensions |
|---|---:|---:|---|
| Train | 0 | 0 | Not available |
| Validation | 0 | 0 | Not available |
| Test | 0 | 0 | Not available |
| **Total** | **0** | **0** | Not available |

The dataset YAML points to the intended split directories but lists a placeholder class named object. The project-local audit in `results/dataset_audit.json` records zero images and labels in every split, no dataset-specific checkpoints, and the separate `dataset/` folder's 13 unassigned raster assets, one SVG, and zero annotations. Sampled files include illustrative project material rather than a labeled detection corpus. Other scanned images comprise upstream demo assets, generated plots, and venv package samples; the two discovered archives are unrelated dependency files. No asset has been silently assigned to a split. Original image resolution, sample counts, class names and split proportions cannot be recovered. I also inspected the original project's public COCO128 diagnostic provenance: its script evaluates 128 images from the COCO train2017 subset, the manifest records only 126 corresponding label files, and the experiment explicitly identifies itself as a labeled local diagnostic rather than official COCO evaluation. It supplies no fixed train/validation/test partition, and the source warns of possible overlap with pretrained YOLO training data; it is therefore not substituted for the task dataset. [Diagnostic script](https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/experiments/local_coco128_validation.py), [run manifest](https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/outputs/local_coco128_20261002/manifest.json). The configured input size is 640×640; this is the planned network input, not a measurement of source-image dimensions. The pipeline preserves supplied splits and does not reshuffle them.

### External dataset discovery and applicability

The AdvT-shirt-1K authors describe 1,131 physical-world images with person and adversarial-patch YOLO boxes. The licensed Hugging Face mirror used here contains 1,123 images and 1,123 person-label files with 4,403 valid person boxes; the eight-image discrepancy remains unresolved. The mirror card lists Apache-2.0. The downloaded files were audited and normalized within the project directory. This corpus was evaluated in a zero-shot person-detection protocol only: the full available mirror is a descriptive external evaluation corpus, with no fine-tuning, hyperparameter selection, or source-provided split for this protocol. Patch boxes are retained in the raw snapshot but excluded from person detection. This is a proxy for detection under adversarial clothing, not the exact SpongePatch/physical–visual DoS task. [Hugging Face dataset card/license](https://huggingface.co/datasets/xingjunm/Adv-T-shirt), [authors' repository](https://github.com/Wwangb/AdvT-shirt-1K), [ICCV 2025 BadPatch paper](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Wang_BadPatch_Diffusion-Based_Generation_of_Physical_Adversarial_Patches_ICCVW_2025_paper.html).

APRICOT/APRICOT-Mask remains a relevant alternative for patch localization and unseen-patch generalization: JHU reports 138 development and 873 test images and an Apache-2.0 license for APRICOT-Mask. Retrieval attempts timed out, so it was not evaluated. GAP/NAPGuard reports a larger corpus and split, but the inspected repository does not state a dataset license; it was not downloaded. Sources and retrieval outcomes are recorded in `results/external_dataset_sources.md` and `results/dataset_audit.json`.

### 2.3 Training plan and shared hyperparameters

The local experiment config defines the planned run below. These settings describe the pipeline; they are not training dynamics from a completed experiment.

| Hyperparameter | Configured value |
|---|---|
| Default model scale | Nano; CLI supports N/S/L/X |
| Epochs / early stopping | 600 / patience 50 |
| Input size | 640×640 |
| Batch / workers | 1 / 0 (conservative for the local 8-GiB GPU; held constant across models) |
| Optimizer | SGD |
| Initial LR / final LR factor | 0.01 / 0.01 |
| Scheduler | Ultralytics linear LR decay; cosine schedule disabled; 3-epoch warm-up |
| Momentum / weight decay | 0.937 / 0.0005 |
| Warm-up / close mosaic | 3 epochs / final 10 epochs |
| Seed / determinism | 42 / deterministic mode enabled |
| Device | Auto-select CUDA when available |
| Color augmentation | HSV h=0.015, s=0.7, v=0.4 |
| Geometry augmentation | translate=0.1, scale=0.5, horizontal flip=0.5; rotation, shear, perspective and vertical flip=0 |
| Composite augmentation | mosaic=1.0, mixup=0, copy-paste=0.1; auto-augment disabled, erasing=0 |

The benchmark config enumerates YOLOv8/10/11/12/13 at N/S/L/X (20 model/scale checkpoint entries). Training accepts one scale per invocation. A fair project comparison requires the same split and shared settings for every family and scale, then evaluation on identical held-out splits. No dataset-dependent runs are present.

## 3. Comprehensive benchmark results

### 3.1 Published MS COCO comparison (literature context only)

The table reproduces the YOLOv13 paper's MS COCO validation AP, parameter, FLOP and Tesla T4 latency values for corresponding N/S/L/X baselines. The source reports AP50 and AP50:95, but not Precision or Recall; those cells are marked NR. FPS is derived as 1000 / latency_ms. The authors state that latency uses a Tesla T4 with TensorRT FP16 for all models and that YOLO11/YOLOv12 variants were reproduced using official settings on the same hardware platform. These are literature values, not local re-training results. [Original paper, Table I and Section IV-A](https://arxiv.org/pdf/2506.17733)

| Model | Scale | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | FLOPs (G) | Latency (ms/image) | FPS* |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8 | N | NR | NR | 52.6 | 37.4 | 3.2 | 8.7 | 1.77 | 565.0 |
| YOLOv10 | N | NR | NR | 53.8 | 38.5 | 2.3 | 6.7 | 1.84 | 543.5 |
| YOLO11 | N | NR | NR | 54.2 | 38.6 | 2.6 | 6.5 | 1.53 | 653.6 |
| YOLOv12 | N | NR | NR | 56.0 | 40.1 | 2.6 | 6.5 | 1.83 | 546.4 |
| **YOLOv13** | **N** | NR | NR | **57.8** | **41.6** | **2.5** | **6.4** | 1.97 | 507.6 |
| YOLOv8 | S | NR | NR | 61.8 | 45.0 | 11.2 | 28.6 | 2.33 | 429.2 |
| YOLOv10 | S | NR | NR | 63.0 | 46.3 | 7.2 | 21.6 | 2.53 | 395.3 |
| YOLO11 | S | NR | NR | 62.6 | 45.8 | 9.4 | 21.5 | 2.56 | 390.6 |
| YOLOv12 | S | NR | NR | 64.2 | 47.1 | 9.3 | 21.4 | 2.82 | 354.6 |
| **YOLOv13** | **S** | NR | NR | **65.2** | **48.0** | **9.0** | **20.8** | 2.98 | 335.6 |
| YOLOv8 | L | NR | NR | 69.8 | 53.0 | 43.7 | 165.2 | 8.13 | 123.0 |
| YOLOv10 | L | NR | NR | 70.1 | 53.2 | 24.4 | 120.3 | 7.31 | 136.8 |
| YOLO11 | L | NR | NR | 69.2 | 52.3 | 25.3 | 86.9 | 6.23 | 160.5 |
| YOLOv12 | L | NR | NR | 70.0 | 53.0 | 26.4 | 88.9 | 7.10 | 140.8 |
| **YOLOv13** | **L** | NR | NR | **70.9** | **53.4** | **27.6** | **88.4** | 8.63 | 115.9 |
| YOLOv8 | X | NR | NR | 71.0 | 54.0 | 68.2 | 257.8 | 12.83 | 77.9 |
| YOLOv10 | X | NR | NR | 71.3 | 54.4 | 29.5 | 160.4 | 10.70 | 93.5 |
| YOLO11 | X | NR | NR | 71.0 | 54.2 | 56.9 | 194.9 | 11.35 | 88.1 |
| YOLOv12 | X | NR | NR | 71.1 | 54.4 | 59.1 | 199.0 | 12.46 | 80.3 |
| **YOLOv13** | **X** | NR | NR | **72.0** | **54.8** | **64.0** | **199.2** | 14.67 | 68.2 |

*FPS is calculated from paper latency and was not independently reported. NR means not reported in the source. [Lei et al.](https://arxiv.org/pdf/2506.17733)

#### IEEE-style LaTeX comparison table

```latex
% Requires \usepackage{booktabs,graphicx}
\begin{table*}[t]
\centering
\scriptsize
\caption{Published MS COCO validation results for YOLO baselines and YOLOv13. Precision and recall were not reported. Latency is Tesla T4 TensorRT FP16; FPS is derived as $1000/\text{latency}$. These are literature values, not project-dataset measurements.}
\label{tab:yolov13-coco-reference}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcrrrrrrrr}
\toprule
Model & Scale & Precision (\%) & Recall (\%) & mAP@0.5 (\%) & mAP@0.5:0.95 (\%) & Params (M) & FLOPs (G) & Latency (ms) & FPS \\
\midrule
YOLOv8 & N & NR & NR & 52.6 & 37.4 & 3.2 & 8.7 & 1.77 & 565.0 \\
YOLOv10 & N & NR & NR & 53.8 & 38.5 & 2.3 & 6.7 & 1.84 & 543.5 \\
YOLO11 & N & NR & NR & 54.2 & 38.6 & 2.6 & 6.5 & 1.53 & 653.6 \\
YOLOv12 & N & NR & NR & 56.0 & 40.1 & 2.6 & 6.5 & 1.83 & 546.4 \\
YOLOv13 & N & NR & NR & 57.8 & 41.6 & 2.5 & 6.4 & 1.97 & 507.6 \\
YOLOv8 & S & NR & NR & 61.8 & 45.0 & 11.2 & 28.6 & 2.33 & 429.2 \\
YOLOv10 & S & NR & NR & 63.0 & 46.3 & 7.2 & 21.6 & 2.53 & 395.3 \\
YOLO11 & S & NR & NR & 62.6 & 45.8 & 9.4 & 21.5 & 2.56 & 390.6 \\
YOLOv12 & S & NR & NR & 64.2 & 47.1 & 9.3 & 21.4 & 2.82 & 354.6 \\
YOLOv13 & S & NR & NR & 65.2 & 48.0 & 9.0 & 20.8 & 2.98 & 335.6 \\
YOLOv8 & L & NR & NR & 69.8 & 53.0 & 43.7 & 165.2 & 8.13 & 123.0 \\
YOLOv10 & L & NR & NR & 70.1 & 53.2 & 24.4 & 120.3 & 7.31 & 136.8 \\
YOLO11 & L & NR & NR & 69.2 & 52.3 & 25.3 & 86.9 & 6.23 & 160.5 \\
YOLOv12 & L & NR & NR & 70.0 & 53.0 & 26.4 & 88.9 & 7.10 & 140.8 \\
YOLOv13 & L & NR & NR & 70.9 & 53.4 & 27.6 & 88.4 & 8.63 & 115.9 \\
YOLOv8 & X & NR & NR & 71.0 & 54.0 & 68.2 & 257.8 & 12.83 & 77.9 \\
YOLOv10 & X & NR & NR & 71.3 & 54.4 & 29.5 & 160.4 & 10.70 & 93.5 \\
YOLO11 & X & NR & NR & 71.0 & 54.2 & 56.9 & 194.9 & 11.35 & 88.1 \\
YOLOv12 & X & NR & NR & 71.1 & 54.4 & 59.1 & 199.0 & 12.46 & 80.3 \\
YOLOv13 & X & NR & NR & 72.0 & 54.8 & 64.0 & 199.2 & 14.67 & 68.2 \\
\bottomrule
\end{tabular}%
}
\end{table*}
```
### 3.2 Project-specific results and local inference diagnostic

The project-dataset accuracy table has zero measured rows. P/R/mAP, validation/test scores, and locally fine-tuned baseline measurements must remain unavailable until the dataset is installed and all 20 configured runs are completed. The intended evaluator records split/device metrics, parameters, FLOPs, latency, FPS, checkpoint hash and run metadata in results/benchmark_eval.csv.

A synthetic-input diagnostic was run on released YOLOv13 weights through the Ultralytics prediction path: 640×640 all-zero image, FP32, batch one, five warm-ups and 30 timed calls per model/device, with CUDA synchronization around GPU timings. The table reports median latency and reciprocal FPS. Raw results are in results/local_inference_diagnostic.csv and its JSON companion.

| Pretrained model | Params (M) | FLOPs (G, local profiler) | GPU median (ms) | GPU FPS | CPU median (ms) | CPU FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv13-N | 2.4942 | 6.5464 | 14.367 | 69.60 | 71.342 | 14.02 |
| YOLOv13-S | 9.0555 | 21.1730 | 13.263 | 75.40 | 129.064 | 7.75 |
| YOLOv13-L | 27.6278 | 89.3604 | 25.352 | 39.44 | 433.127 | 2.31 |
| YOLOv13-X | 64.0575 | 200.6534 | 40.621 | 24.62 | 834.074 | 1.20 |

FLOPs were estimated at 640 pixels with the installed Ultralytics profiling utility and may differ from the paper's counting convention. Timings use synthetic input and pretrained weights, not project data or fine-tuned models; they are not directly comparable to TensorRT FP16. N/S/L exceed 30 FPS in this RTX 5060 diagnostic while X does not. None of the CPU runs reaches 30 FPS. These values do not establish accuracy on physical–visual DoS images.

Published YOLOv13 latency on other platforms is separate context. CPU values use ONNX on Intel Xeon Platinum 8352V, not this workstation CPU. [Original paper, Table VII](https://arxiv.org/pdf/2506.17733)

| Model | Tesla T4 GPU, TensorRT FP16 (ms / FPS*) | RTX 4090 GPU, TensorRT FP16 (ms / FPS*) | Xeon 8352V CPU, ONNX (ms / FPS*) |
|---|---:|---:|---:|
| YOLOv13-N | 1.97 / 507.6 | 1.25 / 800.0 | 39.97 / 25.0 |
| YOLOv13-S | 2.98 / 335.6 | 1.27 / 787.4 | 61.12 / 16.4 |
| YOLOv13-L | 8.63 / 115.9 | 2.36 / 423.7 | 185.36 / 5.4 |
| YOLOv13-X | 14.67 / 68.2 | 3.10 / 322.6 | 330.12 / 3.0 |

*FPS is derived from published latency; the paper identifies the CPU runtime as ONNX.

### 3.3 External zero-shot transfer benchmark

AdvT-shirt-1K was used only as a supplementary, external **zero-shot person-detection** corpus. The downloaded Hugging Face mirror contains 1,123 images and 4,403 valid person boxes after auditing; this is eight fewer images than the authors' 1,131-image headline count, a discrepancy retained in the data manifest. The full available corpus was evaluated once. It has no official train/validation/test split for this protocol; no fine-tuning or model selection was performed. These metrics measure transfer to people wearing physical adversarial T-shirts, not the project's exact SpongePatch/physical-visual DoS task. The mirror lists Apache-2.0. Sources: [dataset card and license](https://huggingface.co/datasets/xingjunm/Adv-T-shirt), [authors' repository](https://github.com/Wwangb/AdvT-shirt-1K), [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Wang_BadPatch_Diffusion-Based_Generation_of_Physical_Adversarial_Patches_ICCVW_2025_paper.html).

All checkpoints are COCO-pretrained. Each predicts COCO class 0 (person), with validation on all 1,123 images at 640×640. Precision, recall, mAP@0.5 and mAP@0.5:0.95 come from the same evaluator. Latency is the median of 30 FP32 batch-1 forward calls after five warm-ups on a zero tensor, with CUDA synchronization; it excludes image decoding, preprocessing, host/device transfer, NMS/postprocessing and model loading. GPU and CPU use the same forward-only protocol. FLOPs are framework-reported estimates at 640 and may differ slightly in counting conventions. Results have no repeated-run confidence intervals; treat small timing gaps as descriptive.

YOLOv12 was evaluated with the official v1.0 release source revision `b1035f8` and its matching weights because those checkpoints serialize the earlier `AAttn.qkv` module. On this Windows environment, the optional FlashAttention extension was unavailable; a local compatibility patch uses PyTorch scaled-dot-product attention (SDPA) for the same scaled dot-product operation. Its latency therefore describes this SDPA backend, not the upstream FlashAttention path. The [official YOLOv12 repository](https://github.com/sunsmarterjie/yolov12) explicitly directs users to its implementation; source/weight hashes and revision are recorded in `results/yolov12_v1_source_manifest.json`.

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

**Interpretation.** YOLOv13 does not achieve the highest transfer mAP@0.5:0.95 at any tested scale. Relative to YOLOv12 it is -0.59/-0.78/-0.40 points at N/S/L; the best same-scale non-v13 baseline is YOLOv12-N (76.20%), YOLOv12-S (84.12%), and YOLOv8-L (88.69%). Thus these external data do **not** provide evidence that HyperACE improves person detection in this transfer setting. YOLOv13 has competitive parameter/FLOP counts, but measured forward latency is sensitive to backend and run conditions; the YOLOv12 row uses SDPA and the figures exclude end-to-end preprocessing/postprocessing. No causal architectural attribution can be made from pretrained cross-family comparisons.

#### IEEE-style LaTeX table for the external benchmark

```latex
% Requires \usepackage{booktabs,graphicx}
\begin{table*}[t]
\centering\scriptsize
\caption{Zero-shot transfer evaluation of COCO-pretrained person detectors on the complete available AdvT-shirt-1K mirror (1,123 images, 4,403 person boxes). No fine-tuning was performed. Latencies are median model-forward times at 640\times640, batch 1, FP32; preprocessing and postprocessing are excluded.}
\label{tab:adv-tshirt-transfer}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrrr}
\toprule
Model & Precision (\%) & Recall (\%) & mAP@0.5 (\%) & mAP@0.5:0.95 (\%) & Params (M) & FLOPs (G) & GPU ms & GPU FPS & CPU ms & CPU FPS \\
\midrule
YOLOv8-N & 88.69 & 79.59 & 89.17 & 73.97 & 3.157 & 8.86 & 4.43 & 225.9 & 39.65 & 25.2 \\
YOLOv10-N & 89.64 & 76.74 & 88.28 & 74.51 & 2.776 & 8.74 & 9.34 & 107.0 & 39.78 & 25.1 \\
YOLO11-N & 88.76 & 79.31 & 89.35 & 74.44 & 2.624 & 6.61 & 16.42 & 60.9 & 54.28 & 18.4 \\
YOLOv12-N & 91.27 & 79.35 & 90.04 & 76.20 & 2.603 & 6.65 & 27.04 & 37.0 & 73.17 & 13.7 \\
YOLOv13-N & 89.41 & 79.65 & 89.45 & 75.61 & 2.494 & 6.55 & 32.44 & 30.8 & 85.89 & 11.6 \\
\midrule
YOLOv8-S & 90.77 & 86.04 & 93.52 & 82.16 & 11.167 & 28.81 & 11.46 & 87.3 & 133.35 & 7.5 \\
YOLOv10-S & 89.74 & 86.40 & 93.25 & 82.88 & 8.128 & 25.11 & 22.12 & 45.2 & 135.12 & 7.4 \\
YOLO11-S & 91.21 & 87.45 & 94.10 & 83.15 & 9.459 & 21.71 & 16.35 & 61.2 & 112.02 & 8.9 \\
YOLOv12-S & 91.68 & 88.14 & 94.43 & 84.12 & 9.286 & 21.69 & 25.90 & 38.6 & 142.92 & 7.0 \\
YOLOv13-S & 91.28 & 86.87 & 93.90 & 83.34 & 9.056 & 21.17 & 13.59 & 73.6 & 99.79 & 10.0 \\
\midrule
YOLOv8-L & 91.50 & 92.61 & 96.44 & 88.69 & 43.692 & 165.73 & 19.36 & 51.7 & 664.53 & 1.5 \\
YOLOv10-L & 92.97 & 90.70 & 96.01 & 88.59 & 25.889 & 127.87 & 28.90 & 34.6 & 505.88 & 2.0 \\
YOLO11-L & 91.65 & 91.66 & 95.82 & 87.84 & 25.372 & 87.61 & 36.99 & 27.0 & 385.73 & 2.6 \\
YOLOv12-L & 92.35 & 91.10 & 95.70 & 87.97 & 26.455 & 89.76 & 51.62 & 19.4 & 565.68 & 1.8 \\
YOLOv13-L & 92.75 & 90.98 & 95.72 & 87.57 & 27.628 & 89.36 & 67.94 & 14.7 & 578.75 & 1.7 \\
\bottomrule
\end{tabular}%
}
\end{table*}
```

#### External transfer trade-off plots

![AdvT-shirt-1K person mAP versus GPU forward latency](external_transfer_tradeoff_latency.png)

![AdvT-shirt-1K person mAP versus FLOPs](external_transfer_tradeoff_flops.png)

Machine-readable measurements, checkpoint hashes, run protocol and errors from the initial incompatible YOLOv12 loader are retained in `results/benchmark_adv_tshirt_transfer.csv`, `results/external_transfer_benchmark_manifest.json`, and `results/yolov12_initial_loader_failures.csv`.


### 3.4 Trade-off figures

These plots use the paper's MS COCO table. They are not local accuracy results because project-dataset mAP is unavailable.

![Published MS COCO mAP@0.5:0.95 versus Tesla T4 latency](tradeoff_map_latency_literature_context.png)

![Published MS COCO mAP@0.5:0.95 versus FLOPs](tradeoff_map_flops_literature_context.png)

### 3.5 Quantitative reading

On MS COCO, reported YOLOv13 AP50:95 gains over YOLO11 at N/S/L/X are +3.0/+2.2/+1.1/+0.6 percentage points; gains over YOLOv12 are +1.5/+0.9/+0.4/+0.4. At N/S/L, FLOPs are slightly lower than YOLOv12 (6.4/20.8/88.4 G vs 6.5/21.4/88.9 G), but Tesla T4 latency is higher at each scale (1.97/2.98/8.63 ms vs 1.83/2.82/7.10 ms). At X, YOLOv13 reports +0.4 AP50:95 points vs YOLOv12 with +0.2 G FLOPs and +2.21 ms latency. The COCO paper supports higher reported accuracy, but not a blanket claim that YOLOv13 is faster than each corresponding baseline.
## 4. Ablation study and scientific evidence

### 4.1 Published controlled ablations (MS COCO validation)

No project ablation was run. These are published COCO values; NR means variant-specific latency was not reported. The FullPAD study varies feature-distribution paths; the authors describe the no-distribution row as equivalent to removing HyperACE. The DS experiment compares the DS block family with vanilla blocks and does not isolate DS-C3k2 alone. [Original paper, Tables III–V](https://arxiv.org/pdf/2506.17733)

| Experiment | Variant | mAP@0.5:0.95 (%) | mAP@0.5 (%) | Params (M) | FLOPs (G) | Latency |
|---|---|---:|---:|---:|---:|---:|
| FullPAD / HyperACE | No distribution (paper's HyperACE-removed setting) | 47.1 | 64.1 | NR | NR | NR |
| FullPAD / HyperACE | Backbone-to-neck only | 47.8 | 65.0 | NR | NR | NR |
| FullPAD / HyperACE | In-neck only | 47.6 | 64.8 | NR | NR | NR |
| FullPAD / HyperACE | Neck-to-head only | 47.7 | 64.9 | NR | NR | NR |
| FullPAD / HyperACE | All three FullPAD paths | 48.0 | 65.2 | NR | NR | NR |
| DS blocks, N | DS blocks enabled | 41.6 | 57.8 | 2.5 | 6.4 | NR |
| DS blocks, N | Vanilla blocks | 41.6 | 57.9 | 3.1 | 7.9 | NR |
| DS blocks, S | DS blocks enabled | 48.0 | 65.2 | 9.0 | 20.8 | NR |
| DS blocks, S | Vanilla blocks | 48.0 | 65.3 | 11.7 | 27.1 | NR |
| Hyperedge count | M=2 | 47.8 | 64.7 | 8.6 | 20.4 | NR |
| Hyperedge count | M=4 | 47.9 | 65.0 | 8.8 | 20.5 | NR |
| Hyperedge count | M=8 | 48.0 | 65.2 | 9.0 | 20.8 | NR |
| Hyperedge count | M=16 | 48.1 | 65.2 | 9.6 | 21.5 | NR |

#### IEEE-style LaTeX ablation table

```latex
% Requires \usepackage{booktabs,graphicx}
\begin{table*}[t]
\centering
\scriptsize
\caption{Published YOLOv13 ablations on MS COCO validation. Variant-specific latency was not reported; values are not project-dataset measurements.}
\label{tab:yolov13-ablation-reference}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrr}
\toprule
Experiment & Variant & mAP@0.5:0.95 (\%) & mAP@0.5 (\%) & Params (M) & FLOPs (G) & Latency \\
\midrule
FullPAD/HyperACE & No distribution & 47.1 & 64.1 & NR & NR & NR \\
FullPAD/HyperACE & Backbone-to-neck only & 47.8 & 65.0 & NR & NR & NR \\
FullPAD/HyperACE & In-neck only & 47.6 & 64.8 & NR & NR & NR \\
FullPAD/HyperACE & Neck-to-head only & 47.7 & 64.9 & NR & NR & NR \\
FullPAD/HyperACE & All three FullPAD paths & 48.0 & 65.2 & NR & NR & NR \\
DS blocks, N & DS blocks enabled & 41.6 & 57.8 & 2.5 & 6.4 & NR \\
DS blocks, N & Vanilla blocks & 41.6 & 57.9 & 3.1 & 7.9 & NR \\
DS blocks, S & DS blocks enabled & 48.0 & 65.2 & 9.0 & 20.8 & NR \\
DS blocks, S & Vanilla blocks & 48.0 & 65.3 & 11.7 & 27.1 & NR \\
Hyperedge count & $M=2$ & 47.8 & 64.7 & 8.6 & 20.4 & NR \\
Hyperedge count & $M=4$ & 47.9 & 65.0 & 8.8 & 20.5 & NR \\
Hyperedge count & $M=8$ & 48.0 & 65.2 & 9.0 & 20.8 & NR \\
Hyperedge count & $M=16$ & 48.1 & 65.2 & 9.6 & 21.5 & NR \\
\bottomrule
\end{tabular}%
}
\end{table*}
```

### 4.2 Interpretation and attribution limits

- **HyperACE:** Adaptive hyperedges connect multiple feature locations, extending pairwise interactions to multi-to-multi, cross-location and cross-scale relations. In the published YOLOv13-S ablation, all three distribution paths score +0.9 AP50:95 and +1.1 AP50 points over the no-distribution setting identified by the authors with HyperACE removed. Individual routing paths are 0.2–0.4 AP50:95 points below all routes. This is evidence from the paper's COCO experiment, not this project's DoS dataset.
- **FullPAD:** FullPAD routes correlation-enhanced features between backbone and neck, within the neck, and between neck and head. Single-route variants score below the all-route result, consistent with the hypothesis that broad feature distribution improves representation and gradient flow. The published table does not isolate training variance and gives no latency by variant.
- **DS-C3k2 / DS blocks:** Depthwise-separable blocks aim to retain receptive-field utility while reducing convolution cost. The paper's family-level comparison shows N retaining 41.6 AP50:95 while reducing 1.5 G FLOPs and 0.6 M parameters versus vanilla blocks. S retains 48.0 AP50:95 while the table implies reductions of 6.3 G FLOPs and 2.7 M parameters; AP50 is 0.1 point lower for DS at both scales. This supports an efficiency claim for the DS family, not an isolated causal claim for DS-C3k2.
- **Hyperedge count:** For YOLOv13-S, raising M from 2 to 8 adds +0.2 AP50:95, +0.4 G FLOPs and +0.4 M parameters. Raising M from 8 to 16 adds +0.1 AP50:95, no AP50 gain, +0.7 G FLOPs and +0.6 M parameters, indicating diminishing returns in this sweep.

The paper's Table V implies S-model DS savings of 6.3 G FLOPs and 2.7 M parameters, while the accompanying prose states 4.2 G and 2.2 M. This report uses the table arithmetic and preserves the discrepancy. [Paper, Table V and discussion](https://arxiv.org/pdf/2506.17733)

The project includes train_ablation.py for one-factor-at-a-time runs and verify_ablation_models.py for architecture smoke checks. All four N-scale variants completed finite 320-pixel GPU forward passes; DS-to-dense conversion matched the original DSConv output within 1.8e-7 maximum absolute error. These are software checks, not accuracy or ablation results.

A project causal study must change one factor at a time while holding split, initialization, seed, augmentation, epochs, image size and evaluation device fixed. The local runner supports full YOLOv13, HyperACE disabled, FullPAD distribution disabled, and DS-C3k2 depthwise convolutions replaced by equivalent dense kernels. Record P/R/mAP, parameters, FLOPs and latency for every variant. The individual FullPAD-route and M=2/4/8/16 sweeps above are published COCO evidence; project-specific values remain unmeasured because the dataset is absent.
## 5. Discussion and conclusion

### English discussion

Published COCO results support the hypothesis that HyperACE's adaptive hyperedges can represent group-wise spatial relations beyond ordinary pairwise attention, while FullPAD routes correlation-enhanced features through the detector. The paper reports YOLOv13 AP50:95 gains over YOLO11 of 3.0/2.2/1.1/0.6 points at N/S/L/X, and gains over YOLOv12 of 1.5/0.9/0.4/0.4 points. Published DS-block ablations show lower parameter/FLOP counts with nearly retained AP, but do not isolate DS-C3k2 alone. These are COCO results, not measurements on the present project.

The external zero-shot evidence is more cautious: YOLOv13 reaches 75.61/83.34/87.57% person mAP@0.5:0.95, trailing YOLOv12 by 0.59/0.78/0.40 points. It also trails the best same-scale non-v13 baseline by 0.59/0.78/1.12 points. Thus architectural motivation and published COCO evidence do not guarantee improved transfer to this adversarial-clothing corpus. The comparison uses pretrained checkpoints across different implementations; the YOLOv12 timing uses PyTorch SDPA because FlashAttention is unavailable. Timing is forward-only, single-run, and not end-to-end; small latency differences are not robust conclusions.

The present contribution is a reproducible project-local pipeline, a label-audited external zero-shot benchmark with source and checkpoint provenance, and literature-grounded ablation analysis. It does not yet establish a performance gain for the NCKH physical–visual DoS task. The next scientific step is to restore the true labeled project splits and class map, fine-tune each family under the shared protocol, reserve the test split for final evaluation, and run one-factor-at-a-time ablations with repeated timing on the intended edge hardware. The external benchmark must remain a separate transfer result.

### Thảo luận tiếng Việt

Kết quả COCO đã công bố ủng hộ giả thuyết rằng siêu cạnh thích nghi của HyperACE biểu diễn quan hệ không gian theo nhóm vượt ngoài attention theo cặp thông thường, còn FullPAD phân phối đặc trưng tương quan qua các tầng của detector. Bài báo ghi nhận YOLOv13 tăng AP50:95 so với YOLO11 lần lượt 3,0/2,2/1,1/0,6 điểm ở N/S/L/X, và so với YOLOv12 là 1,5/0,9/0,4/0,4 điểm. Ablation khối DS đã công bố cho thấy giảm tham số/FLOPs trong khi gần như giữ AP, nhưng chưa cô lập riêng DS-C3k2. Đây là kết quả COCO, không phải phép đo trên dữ liệu đề tài.

Bằng chứng zero-shot bên ngoài thận trọng hơn: YOLOv13 đạt person mAP@0.5:0.95 là 75.61/83.34/87.57%, thấp hơn YOLOv12 0.59/0.78/0.40 điểm. Mô hình cũng thấp hơn baseline tốt nhất ở từng scale lần lượt 0.59/0.78/1.12 điểm. Vì vậy, động cơ kiến trúc và kết quả COCO đã công bố không bảo đảm khả năng chuyển miền tốt hơn trên tập áo có patch đối kháng này. Phép đối sánh dùng trọng số pretrained từ các triển khai khác nhau; latency YOLOv12 dùng PyTorch SDPA vì môi trường không có FlashAttention. Đây là phép đo forward một lượt, không phải latency đầu-cuối; chênh lệch nhỏ không nên được xem là kết luận vững chắc.

Đóng góp hiện tại là pipeline có thể tái lập trong thư mục dự án, benchmark zero-shot ngoài đã kiểm tra nhãn và truy xuất nguồn/trọng số, cùng phân tích ablation dựa trên tài liệu gốc. Nghiên cứu chưa chứng minh YOLOv13 cải thiện bài toán physical–visual DoS của đề tài. Bước nghiên cứu tiếp theo là khôi phục đúng split có nhãn và class map, fine-tune các họ mô hình theo giao thức chung, giữ test cho đánh giá cuối, thực hiện ablation từng yếu tố và đo lặp trên phần cứng edge dự kiến. Benchmark ngoài phải được báo cáo tách biệt với kết quả của đề tài.

### Conclusion

YOLOv13 provides a plausible high-order-context architecture, and its paper reports favorable COCO accuracy and DS-block efficiency evidence. On the measured external zero-shot person benchmark, however, YOLOv13 N/S/L does not exceed the best same-scale baseline. No task-specific training or causal ablation was possible because the project dataset is absent. The measured transfer result, data-source audit, full comparison table, latency protocol, published ablation evidence, and implementation limitations are documented; conclusions about the target NCKH task require its labeled splits and controlled retraining.

### Kết luận

YOLOv13 đưa ra hướng kiến trúc hợp lý để mô hình hóa ngữ cảnh bậc cao; bài báo gốc ghi nhận độ chính xác COCO thuận lợi và bằng chứng tiết kiệm tài nguyên của khối DS. Tuy nhiên, trên benchmark person zero-shot bên ngoài đã đo, YOLOv13 N/S/L không vượt baseline tốt nhất cùng scale. Chưa thể huấn luyện hay ablation nhân quả trên bài toán đề tài vì dataset chưa có. Báo cáo lưu kết quả chuyển miền đã đo, nguồn dữ liệu, bảng đối sánh đầy đủ, quy trình latency, ablation đã công bố và các giới hạn triển khai; kết luận cho bài toán NCKH cần split có nhãn và huấn luyện kiểm soát trên dữ liệu thật.

## 6. Reproducibility artifacts and project tree

The authored pipeline, environment, data, checkpoints, results, and reports are confined to `project_yolov13/`. The `.venv/`, `.cache/`, upstream source snapshots, and raw 3,377-file external dataset snapshot are abbreviated below.

```text
project_yolov13/
├── .cache/                           # project-local package/model/runtime caches
├── .venv/                            # isolated Python environment
├── configs/
│   ├── ablation_template.csv
│   ├── benchmark.yaml
│   ├── experiment.yaml
│   ├── paper_ablation.csv
│   ├── paper_device_latency.csv
│   └── paper_results.csv
├── data/
│   ├── images/{train,val,test}/      # task splits currently empty
│   ├── labels/{train,val,test}/      # task labels currently empty
│   ├── external/adv_tshirt_1k/       # external source snapshot (abbreviated)
│   ├── external/adv_tshirt_person_eval/ # zero-shot person benchmark data
│   ├── adv_tshirt_person_eval.yaml
│   └── dataset.yaml                  # placeholder class; not trainable
├── dataset/                          # 13 unsplit raster assets + 1 SVG; no labels
├── docs/
│   └── academic_discussion.md
├── results/
│   ├── ablation_analysis.md
│   ├── benchmark_adv_tshirt_transfer.csv / benchmark_yolov12_transfer.csv
│   ├── comparison_results.md / .tex
│   ├── current_task_dataset_audit.log / dataset_audit_current.json
│   ├── dataset_audit.json
│   ├── external_dataset_sources.md
│   ├── external_transfer_comparison.md / .tex
│   ├── external_transfer_tradeoff_{latency,flops}.png
│   ├── external_transfer_{dataset,benchmark,export}_manifest.json
│   ├── final_academic_report.md / finalize_research_report.log
│   ├── literature_device_latency.md / .tex
│   ├── literature_reference.md / .tex
│   ├── local_inference_diagnostic.csv / .json
│   ├── paper_ablation_reference.md / .tex
│   ├── task_training_preflight.log
│   ├── tradeoff_map_{latency,flops}_literature_context.png
│   └── yolov12_v1_source_manifest.json / yolov12_initial_loader_failures.csv
├── runs/
│   ├── adv_tshirt_transfer/           # zero-shot evaluator outputs
│   └── train/                         # no task training run created
├── weights/                           # YOLOv13-N/S/L/X checkpoints + manifest
├── yolov13_src/                       # iMoonLab YOLOv13 source (abbreviated)
├── yolov12_src/                       # pinned current YOLOv12 source snapshot
├── yolov12_v1_src/                    # release-matched source with documented SDPA fallback
├── audit_task_dataset.py
├── benchmark_adv_tshirt_transfer.py
├── benchmark_eval.py
├── benchmark_inference_diagnostic.py
├── benchmark_yolov12_transfer.py
├── download_external_dataset.py
├── export_external_transfer_artifacts.py
├── export_results.py
├── finalize_research_report.py
├── merge_transfer_benchmark.py
├── pipeline_utils.py
├── prepare_external_benchmark.py
├── retrieve_yolov12_v1.py
├── setup_env.bat
├── train_ablation.py
├── train_baselines.py
├── train_yolov13.py
├── yolov8*.pt / yolov10*.pt / yolo11*.pt / yolo12*.pt # cached baselines
├── verify_ablation_models.py
├── verify_models.py
└── README.md
```

### Pipeline files

- `audit_task_dataset.py`: checks configured split counts, missing/orphan labels, class IDs, normalized bounding boxes and placeholder names.
- `download_external_dataset.py` / `prepare_external_benchmark.py`: retrieve the licensed mirror and produce audited person-only evaluation labels.
- `benchmark_adv_tshirt_transfer.py`: evaluates YOLOv8/10/11/13; `benchmark_yolov12_transfer.py` uses the release-matched official YOLOv12-v1.0 source with SDPA fallback.
- `merge_transfer_benchmark.py` and `export_external_transfer_artifacts.py`: combine valid rows and emit comparison tables/plots.
- `train_yolov13.py`, `train_baselines.py`, and `train_ablation.py`: fine-tuning and one-factor ablation runners; task training halts at data preflight until valid splits are available.
- `finalize_research_report.py`: refreshes this report and the dataset audit from measured outputs.

## References

1. M. Lei et al., “YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception,” arXiv:2506.17733, 2025. [Paper PDF](https://arxiv.org/pdf/2506.17733); [abstract](https://arxiv.org/abs/2506.17733).
2. iMoonLab, [official YOLOv13 implementation and model documentation](https://github.com/iMoonLab/yolov13).
3. Official YOLOv12 source and v1.0 weights, [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12), release tag `v1.0` (source revision `b1035f8` used here).
4. AdvT-shirt-1K, [dataset card/license](https://huggingface.co/datasets/xingjunm/Adv-T-shirt), [authors' repository](https://github.com/Wwangb/AdvT-shirt-1K), and [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Wang_BadPatch_Diffusion-Based_Generation_of_Physical_Adversarial_Patches_ICCVW_2025_paper.html).
5. APRICOT, [paper](https://arxiv.org/abs/1912.08166), [JHU APRICOT-Mask source/license](https://aiem.jhu.edu/datasets/apricot-mask/).
6. Original project repository, [COCO128 local diagnostic script](https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/experiments/local_coco128_validation.py) and [provenance manifest](https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/outputs/local_coco128_20261002/manifest.json).

