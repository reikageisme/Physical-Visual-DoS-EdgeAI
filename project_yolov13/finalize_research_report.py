from __future__ import annotations

import csv
import json

from pipeline_utils import configure_project_environment, project_path, require_local_venv, write_json, utc_timestamp

configure_project_environment()


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    a = text.index(start)
    b = text.index(end, a)
    return text[:a] + replacement.rstrip() + "\n\n" + text[b:]


def main() -> int:
    require_local_venv()
    report_path = project_path("results/final_academic_report.md", must_exist=True)
    report = report_path.read_text(encoding="utf-8")
    with project_path("results/benchmark_adv_tshirt_transfer.csv", must_exist=True).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("status") == "ok"]
    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 successful N/S/L measurements, received {len(rows)}.")
    by_name = {row["model_name"]: row for row in rows}
    score = lambda family, scale: float(by_name[f"{family}-{scale.upper()}"]["map50_95"]) * 100.0
    delta_v12 = {scale: score("YOLOv13", scale) - score("YOLOv12", scale) for scale in "nsl"}
    best_baseline = {
        scale: max((float(row["map50_95"]) * 100.0, row["family"]) for row in rows if row["scale"] == scale and row["family"] != "YOLOv13")
        for scale in "nsl"
    }
    table_md = project_path("results/external_transfer_comparison.md", must_exist=True).read_text(encoding="utf-8")
    table_lines = table_md.splitlines()
    table_start = next(i for i, line in enumerate(table_lines) if line.startswith("| Model |"))
    table_end = next(i for i in range(table_start + 2, len(table_lines)) if not table_lines[i].startswith("|"))
    table_block = "\n".join(table_lines[table_start:table_end])
    tex_table = project_path("results/external_transfer_comparison.tex", must_exist=True).read_text(encoding="utf-8").strip()

    external_section = f"""### 3.3 External zero-shot transfer benchmark

AdvT-shirt-1K was used only as a supplementary, external **zero-shot person-detection** corpus. The downloaded Hugging Face mirror contains 1,123 images and 4,403 valid person boxes after auditing; this is eight fewer images than the authors' 1,131-image headline count, a discrepancy retained in the data manifest. The full available corpus was evaluated once. It has no official train/validation/test split for this protocol; no fine-tuning or model selection was performed. These metrics measure transfer to people wearing physical adversarial T-shirts, not the project's exact SpongePatch/physical-visual DoS task. The mirror lists Apache-2.0. Sources: [dataset card and license](https://huggingface.co/datasets/xingjunm/Adv-T-shirt), [authors' repository](https://github.com/Wwangb/AdvT-shirt-1K), [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Wang_BadPatch_Diffusion-Based_Generation_of_Physical_Adversarial_Patches_ICCVW_2025_paper.html).

All checkpoints are COCO-pretrained. Each predicts COCO class 0 (person), with validation on all 1,123 images at 640×640. Precision, recall, mAP@0.5 and mAP@0.5:0.95 come from the same evaluator. Latency is the median of 30 FP32 batch-1 forward calls after five warm-ups on a zero tensor, with CUDA synchronization; it excludes image decoding, preprocessing, host/device transfer, NMS/postprocessing and model loading. GPU and CPU use the same forward-only protocol. FLOPs are framework-reported estimates at 640 and may differ slightly in counting conventions. Results have no repeated-run confidence intervals; treat small timing gaps as descriptive.

YOLOv12 was evaluated with the official v1.0 release source revision `b1035f8` and its matching weights because those checkpoints serialize the earlier `AAttn.qkv` module. On this Windows environment, the optional FlashAttention extension was unavailable; a local compatibility patch uses PyTorch scaled-dot-product attention (SDPA) for the same scaled dot-product operation. Its latency therefore describes this SDPA backend, not the upstream FlashAttention path. The [official YOLOv12 repository](https://github.com/sunsmarterjie/yolov12) explicitly directs users to its implementation; source/weight hashes and revision are recorded in `results/yolov12_v1_source_manifest.json`.

{table_block}

**Interpretation.** YOLOv13 does not achieve the highest transfer mAP@0.5:0.95 at any tested scale. Relative to YOLOv12 it is {delta_v12['n']:+.2f}/{delta_v12['s']:+.2f}/{delta_v12['l']:+.2f} points at N/S/L; the best same-scale non-v13 baseline is {best_baseline['n'][1]}-N ({best_baseline['n'][0]:.2f}%), {best_baseline['s'][1]}-S ({best_baseline['s'][0]:.2f}%), and {best_baseline['l'][1]}-L ({best_baseline['l'][0]:.2f}%). Thus these external data do **not** provide evidence that HyperACE improves person detection in this transfer setting. YOLOv13 has competitive parameter/FLOP counts, but measured forward latency is sensitive to backend and run conditions; the YOLOv12 row uses SDPA and the figures exclude end-to-end preprocessing/postprocessing. No causal architectural attribution can be made from pretrained cross-family comparisons.

#### IEEE-style LaTeX table for the external benchmark

```latex
{tex_table}
```

#### External transfer trade-off plots

![AdvT-shirt-1K person mAP versus GPU forward latency](external_transfer_tradeoff_latency.png)

![AdvT-shirt-1K person mAP versus FLOPs](external_transfer_tradeoff_flops.png)

Machine-readable measurements, checkpoint hashes, run protocol and errors from the initial incompatible YOLOv12 loader are retained in `results/benchmark_adv_tshirt_transfer.csv`, `results/external_transfer_benchmark_manifest.json`, and `results/yolov12_initial_loader_failures.csv`.
"""

    report = report.replace(
        "# YOLOv13 for Physical–Visual DoS Detection at the Edge: Reproducible Setup and Literature-Grounded Evaluation",
        "# YOLOv13 for Physical–Visual DoS at the Edge: Published Ablations and External Zero-Shot Transfer",
        1,
    )
    report = report.replace("**Preliminary research report — 4 October 2026**  \n**Status:** implementation-ready pipeline and evidence review; project-dataset experiment pending.",
        "**Research report — 4 October 2026**  \n**Status:** external transfer benchmark completed; project-specific training and causal ablation remain pending due to missing labeled splits.", 1)
    report = replace_between(report, "## Execution status and interpretation", "## Title and abstracts", """## Execution status and interpretation

The canonical project train/validation/test directories contain zero images and zero labels; the dataset YAML still has the placeholder class `object`. A fresh audit on 4 October 2026 confirmed this state: `results/dataset_audit_current.json` records 0/0/0 split images, 0 label files and no malformed boxes because no task labels are present. The separate root-level `dataset/` contains 13 raster images and one SVG but no annotations or split metadata. Running `.venv/Scripts/python.exe train_yolov13.py --scale n --name task_dataset_preflight` stopped in dataset validation with “No images found in the configured train split,” before checkpoint download or training run creation. Therefore no project-specific checkpoint, accuracy score, validation/test result, or local causal ablation exists. These values remain unavailable and are not fabricated.

A supplementary external zero-shot benchmark has now been measured on the full available AdvT-shirt-1K mirror: 1,123 images with 4,403 person boxes, evaluated across YOLOv8/10/11/12/13 N/S/L checkpoints. This external corpus is relevant to adversarial-clothing transfer, but it is not the exact NCKH task dataset, has no prescribed split for this protocol, and was not used to fine-tune or select a model. Published MS COCO comparisons and ablations are labeled as literature context; local external measurements are labeled as zero-shot transfer results.

The original project's COCO128 diagnostic candidate was not substituted: it has no fixed train/validation/test split, two images without label files, and possible overlap with pretraining. The external dataset source/license/retrieval audit is in `results/external_dataset_sources.md`.

> **Operational scope deviation.** A one-off YOLOv13/YOLOv12 diagnostic import initialized small Ultralytics settings JSON files under the Windows user AppData configuration directory before the project-local cache setting was applied. Subsequent pipeline runs set `YOLO_CONFIG_DIR` and all experiment data, models, checkpoints, logs, plots, and reports under `project_yolov13/`. A cleanup attempt for the host-level settings artifact was rejected by the local file policy; strict “writes only inside the project” confinement was therefore not fully met for those two settings files.""")

    report = replace_between(report, "### English abstract", "### Tóm tắt tiếng Việt", f"""### English abstract

We evaluate YOLOv13 for physical–visual denial-of-service (DoS) detection while separating task-specific evidence from an external transfer proxy. The project's labeled train/validation/test splits are empty, so fine-tuning, project accuracy measurement and causal architectural ablation cannot be completed. As a supplementary zero-shot experiment, we evaluate COCO-pretrained YOLOv8/10/11/12/13 nano, small and large detectors on the complete available AdvT-shirt-1K mirror (1,123 images, 4,403 person boxes), without fine-tuning. YOLOv13 obtains person mAP@0.5:0.95 of {score('YOLOv13','n'):.2f}/{score('YOLOv13','s'):.2f}/{score('YOLOv13','l'):.2f}% at N/S/L. This is {delta_v12['n']:+.2f}/{delta_v12['s']:+.2f}/{delta_v12['l']:+.2f} points relative to YOLOv12; YOLOv13 is not the best model at any scale on this corpus. The results therefore provide no empirical basis for claiming HyperACE-driven improvement in the target task. Published COCO ablations offer architectural motivation but remain separate evidence. The report provides the full benchmark, model hashes, latency/FLOP protocol and data sources while identifying the missing project dataset as the blocker to task-specific scientific conclusions.""")

    report = replace_between(report, "### Tóm tắt tiếng Việt", "## 1. Introduction and research question", f"""### Tóm tắt tiếng Việt

Nghiên cứu đánh giá YOLOv13 cho bài toán từ chối dịch vụ vật lý–thị giác (physical–visual DoS), đồng thời tách biệt bằng chứng trên dữ liệu đề tài với phép đo chuyển miền bên ngoài. Các thư mục train/validation/test có nhãn của đề tài hiện trống nên chưa thể fine-tune, đo độ chính xác trên bài toán mục tiêu hoặc làm ablation nhân quả. Phép thử bổ sung zero-shot dùng các mô hình COCO-pretrained YOLOv8/10/11/12/13 biến thể nano, small và large trên toàn bộ bản dữ liệu AdvT-shirt-1K hiện có (1.123 ảnh, 4.403 hộp người), không huấn luyện thêm. YOLOv13 đạt person mAP@0.5:0.95 lần lượt {score('YOLOv13','n'):.2f}/{score('YOLOv13','s'):.2f}/{score('YOLOv13','l'):.2f}% cho N/S/L; chênh lệch so với YOLOv12 là {delta_v12['n']:+.2f}/{delta_v12['s']:+.2f}/{delta_v12['l']:+.2f} điểm. YOLOv13 không dẫn đầu ở bất kỳ scale nào trên tập ngoài này, do đó kết quả không chứng minh cải thiện nhờ HyperACE trong bài toán mục tiêu. Ablation COCO đã công bố chỉ cung cấp cơ sở kiến trúc tham khảo. Báo cáo cung cấp benchmark đầy đủ, hash trọng số, quy trình đo latency/FLOPs và nguồn dữ liệu, đồng thời xác định thiếu dataset đề tài là hạn chế ngăn cản kết luận khoa học đặc thù.""")

    # Update the external-source subsection to describe the downloaded and evaluated corpus.
    report = replace_between(report, "### External dataset discovery and applicability", "### 2.3 Training plan and shared hyperparameters", """### External dataset discovery and applicability

The AdvT-shirt-1K authors describe 1,131 physical-world images with person and adversarial-patch YOLO boxes. The licensed Hugging Face mirror used here contains 1,123 images and 1,123 person-label files with 4,403 valid person boxes; the eight-image discrepancy remains unresolved. The mirror card lists Apache-2.0. The downloaded files were audited and normalized within the project directory. This corpus was evaluated in a zero-shot person-detection protocol only: the full available mirror is a descriptive external evaluation corpus, with no fine-tuning, hyperparameter selection, or source-provided split for this protocol. Patch boxes are retained in the raw snapshot but excluded from person detection. This is a proxy for detection under adversarial clothing, not the exact SpongePatch/physical–visual DoS task. [Hugging Face dataset card/license](https://huggingface.co/datasets/xingjunm/Adv-T-shirt), [authors' repository](https://github.com/Wwangb/AdvT-shirt-1K), [ICCV 2025 BadPatch paper](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Wang_BadPatch_Diffusion-Based_Generation_of_Physical_Adversarial_Patches_ICCVW_2025_paper.html).

APRICOT/APRICOT-Mask remains a relevant alternative for patch localization and unseen-patch generalization: JHU reports 138 development and 873 test images and an Apache-2.0 license for APRICOT-Mask. Retrieval attempts timed out, so it was not evaluated. GAP/NAPGuard reports a larger corpus and split, but the inspected repository does not state a dataset license; it was not downloaded. Sources and retrieval outcomes are recorded in `results/external_dataset_sources.md` and `results/dataset_audit.json`.""")

    report = report.replace("### 3.3 Trade-off figures", external_section + "\n\n### 3.4 Trade-off figures", 1)
    report = report.replace("### 3.4 Quantitative reading", "### 3.5 Quantitative reading", 1)

    report = replace_between(report, "### English discussion", "### Conclusion", f"""### English discussion

Published COCO results support the hypothesis that HyperACE's adaptive hyperedges can represent group-wise spatial relations beyond ordinary pairwise attention, while FullPAD routes correlation-enhanced features through the detector. The paper reports YOLOv13 AP50:95 gains over YOLO11 of 3.0/2.2/1.1/0.6 points at N/S/L/X, and gains over YOLOv12 of 1.5/0.9/0.4/0.4 points. Published DS-block ablations show lower parameter/FLOP counts with nearly retained AP, but do not isolate DS-C3k2 alone. These are COCO results, not measurements on the present project.

The external zero-shot evidence is more cautious: YOLOv13 reaches {score('YOLOv13','n'):.2f}/{score('YOLOv13','s'):.2f}/{score('YOLOv13','l'):.2f}% person mAP@0.5:0.95, trailing YOLOv12 by {abs(delta_v12['n']):.2f}/{abs(delta_v12['s']):.2f}/{abs(delta_v12['l']):.2f} points. It also trails the best same-scale non-v13 baseline by {best_baseline['n'][0]-score('YOLOv13','n'):.2f}/{best_baseline['s'][0]-score('YOLOv13','s'):.2f}/{best_baseline['l'][0]-score('YOLOv13','l'):.2f} points. Thus architectural motivation and published COCO evidence do not guarantee improved transfer to this adversarial-clothing corpus. The comparison uses pretrained checkpoints across different implementations; the YOLOv12 timing uses PyTorch SDPA because FlashAttention is unavailable. Timing is forward-only, single-run, and not end-to-end; small latency differences are not robust conclusions.

The present contribution is a reproducible project-local pipeline, a label-audited external zero-shot benchmark with source and checkpoint provenance, and literature-grounded ablation analysis. It does not yet establish a performance gain for the NCKH physical–visual DoS task. The next scientific step is to restore the true labeled project splits and class map, fine-tune each family under the shared protocol, reserve the test split for final evaluation, and run one-factor-at-a-time ablations with repeated timing on the intended edge hardware. The external benchmark must remain a separate transfer result.

### Thảo luận tiếng Việt

Kết quả COCO đã công bố ủng hộ giả thuyết rằng siêu cạnh thích nghi của HyperACE biểu diễn quan hệ không gian theo nhóm vượt ngoài attention theo cặp thông thường, còn FullPAD phân phối đặc trưng tương quan qua các tầng của detector. Bài báo ghi nhận YOLOv13 tăng AP50:95 so với YOLO11 lần lượt 3,0/2,2/1,1/0,6 điểm ở N/S/L/X, và so với YOLOv12 là 1,5/0,9/0,4/0,4 điểm. Ablation khối DS đã công bố cho thấy giảm tham số/FLOPs trong khi gần như giữ AP, nhưng chưa cô lập riêng DS-C3k2. Đây là kết quả COCO, không phải phép đo trên dữ liệu đề tài.

Bằng chứng zero-shot bên ngoài thận trọng hơn: YOLOv13 đạt person mAP@0.5:0.95 là {score('YOLOv13','n'):.2f}/{score('YOLOv13','s'):.2f}/{score('YOLOv13','l'):.2f}%, thấp hơn YOLOv12 {abs(delta_v12['n']):.2f}/{abs(delta_v12['s']):.2f}/{abs(delta_v12['l']):.2f} điểm. Mô hình cũng thấp hơn baseline tốt nhất ở từng scale lần lượt {best_baseline['n'][0]-score('YOLOv13','n'):.2f}/{best_baseline['s'][0]-score('YOLOv13','s'):.2f}/{best_baseline['l'][0]-score('YOLOv13','l'):.2f} điểm. Vì vậy, động cơ kiến trúc và kết quả COCO đã công bố không bảo đảm khả năng chuyển miền tốt hơn trên tập áo có patch đối kháng này. Phép đối sánh dùng trọng số pretrained từ các triển khai khác nhau; latency YOLOv12 dùng PyTorch SDPA vì môi trường không có FlashAttention. Đây là phép đo forward một lượt, không phải latency đầu-cuối; chênh lệch nhỏ không nên được xem là kết luận vững chắc.

Đóng góp hiện tại là pipeline có thể tái lập trong thư mục dự án, benchmark zero-shot ngoài đã kiểm tra nhãn và truy xuất nguồn/trọng số, cùng phân tích ablation dựa trên tài liệu gốc. Nghiên cứu chưa chứng minh YOLOv13 cải thiện bài toán physical–visual DoS của đề tài. Bước nghiên cứu tiếp theo là khôi phục đúng split có nhãn và class map, fine-tune các họ mô hình theo giao thức chung, giữ test cho đánh giá cuối, thực hiện ablation từng yếu tố và đo lặp trên phần cứng edge dự kiến. Benchmark ngoài phải được báo cáo tách biệt với kết quả của đề tài.""")

    report = replace_between(report, "### Conclusion", "## 6. Reproducibility artifacts and project tree", """### Conclusion

YOLOv13 provides a plausible high-order-context architecture, and its paper reports favorable COCO accuracy and DS-block efficiency evidence. On the measured external zero-shot person benchmark, however, YOLOv13 N/S/L does not exceed the best same-scale baseline. No task-specific training or causal ablation was possible because the project dataset is absent. The measured transfer result, data-source audit, full comparison table, latency protocol, published ablation evidence, and implementation limitations are documented; conclusions about the target NCKH task require its labeled splits and controlled retraining.

### Kết luận

YOLOv13 đưa ra hướng kiến trúc hợp lý để mô hình hóa ngữ cảnh bậc cao; bài báo gốc ghi nhận độ chính xác COCO thuận lợi và bằng chứng tiết kiệm tài nguyên của khối DS. Tuy nhiên, trên benchmark person zero-shot bên ngoài đã đo, YOLOv13 N/S/L không vượt baseline tốt nhất cùng scale. Chưa thể huấn luyện hay ablation nhân quả trên bài toán đề tài vì dataset chưa có. Báo cáo lưu kết quả chuyển miền đã đo, nguồn dữ liệu, bảng đối sánh đầy đủ, quy trình latency, ablation đã công bố và các giới hạn triển khai; kết luận cho bài toán NCKH cần split có nhãn và huấn luyện kiểm soát trên dữ liệu thật.""")

    tree_section = """## 6. Reproducibility artifacts and project tree

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

"""
    report = replace_between(report, "## 6. Reproducibility artifacts and project tree", "### Pipeline files", tree_section)
    pipeline_section = """### Pipeline files

- `audit_task_dataset.py`: checks configured split counts, missing/orphan labels, class IDs, normalized bounding boxes and placeholder names.
- `download_external_dataset.py` / `prepare_external_benchmark.py`: retrieve the licensed mirror and produce audited person-only evaluation labels.
- `benchmark_adv_tshirt_transfer.py`: evaluates YOLOv8/10/11/13; `benchmark_yolov12_transfer.py` uses the release-matched official YOLOv12-v1.0 source with SDPA fallback.
- `merge_transfer_benchmark.py` and `export_external_transfer_artifacts.py`: combine valid rows and emit comparison tables/plots.
- `train_yolov13.py`, `train_baselines.py`, and `train_ablation.py`: fine-tuning and one-factor ablation runners; task training halts at data preflight until valid splits are available.
- `finalize_research_report.py`: refreshes this report and the dataset audit from measured outputs.

"""
    report = replace_between(report, "### Pipeline files", "## References", pipeline_section)
    references_section = """## References

1. M. Lei et al., “YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception,” arXiv:2506.17733, 2025. [Paper PDF](https://arxiv.org/pdf/2506.17733); [abstract](https://arxiv.org/abs/2506.17733).
2. iMoonLab, [official YOLOv13 implementation and model documentation](https://github.com/iMoonLab/yolov13).
3. Official YOLOv12 source and v1.0 weights, [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12), release tag `v1.0` (source revision `b1035f8` used here).
4. AdvT-shirt-1K, [dataset card/license](https://huggingface.co/datasets/xingjunm/Adv-T-shirt), [authors' repository](https://github.com/Wwangb/AdvT-shirt-1K), and [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Wang_BadPatch_Diffusion-Based_Generation_of_Physical_Adversarial_Patches_ICCVW_2025_paper.html).
5. APRICOT, [paper](https://arxiv.org/abs/1912.08166), [JHU APRICOT-Mask source/license](https://aiem.jhu.edu/datasets/apricot-mask/).
6. Original project repository, [COCO128 local diagnostic script](https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/experiments/local_coco128_validation.py) and [provenance manifest](https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/outputs/local_coco128_20261002/manifest.json).
"""
    ref_start = report.index("## References")
    report = report[:ref_start] + references_section + "\n"

    report_path.write_text(report, encoding="utf-8")

    audit_path = project_path("results/dataset_audit.json", must_exist=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["external_transfer_benchmark"] = {
        "measured_at_utc": utc_timestamp(),
        "successful_models": len(rows),
        "metric_scope": "zero-shot person detection on full AdvT-shirt-1K mirror; no task-dataset training or fine-tuning",
        "yolov13_map50_95_percent": {scale: round(score("YOLOv13", scale), 4) for scale in "nsl"},
        "yolov13_minus_yolov12_map50_95_points": {scale: round(delta_v12[scale], 4) for scale in "nsl"},
        "csv": "results/benchmark_adv_tshirt_transfer.csv",
        "table_markdown": "results/external_transfer_comparison.md",
        "table_latex": "results/external_transfer_comparison.tex",
    }
    write_json(audit_path, audit)
    print("Updated final bilingual report and dataset audit from 15 measured external-transfer rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
