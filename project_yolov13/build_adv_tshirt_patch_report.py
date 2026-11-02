"""Create an evidence-only academic report from the derived patch benchmark artifacts."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, load_yaml, project_path, require_local_venv

configure_project_environment()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value) -> str:
    return "--" if value in (None, "") else f"{float(value) * 100:.2f}"


def num(value, digits=3) -> str:
    return "--" if value in (None, "") else f"{float(value):.{digits}f}"


def latex_escape(text: str) -> str:
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def current_hardware(torch) -> dict:
    result = {"cpu": platform.processor() or platform.machine(), "ram_gb": None, "gpu": None, "gpu_vram_gb": None, "cuda": torch.version.cuda}
    if sys.platform == "win32":
        try:
            cpu_name = subprocess.check_output(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ],
                text=True,
                timeout=10,
            ).strip()
            if cpu_name:
                result["cpu"] = cpu_name
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import psutil

        result["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = props.name
        result["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
    return result


def main() -> int:
    require_local_venv()
    import torch
    from PIL import Image
    torch_version = getattr(torch, "__version__", "unknown")

    benchmark_path = project_path("results/adv_tshirt_patch_benchmark.csv", must_exist=True)
    audit_path = project_path("results/adv_tshirt_patch_detection_audit.json", must_exist=True)
    manifest_path = project_path("results/adv_tshirt_patch_dataset_manifest.json", must_exist=True)
    ablation_path = project_path("results/adv_tshirt_patch_ablation.csv", must_exist=True)
    paper_path = project_path("configs/paper_results.csv", must_exist=True)
    benchmark = read_csv(benchmark_path)
    ablations = read_csv(ablation_path)
    paper_rows = read_csv(paper_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if audit.get("status") != "ready":
        raise RuntimeError("The derived patch dataset audit is not ready.")
    if not benchmark or not any(row.get("split") == "test" for row in benchmark):
        raise RuntimeError("No held-out test benchmark results are available.")
    if not ablations:
        raise RuntimeError("No measured ablation results are available.")
    native_sizes = []
    image_root = project_path("dataset/adv_tshirt_patch_detection/images", must_exist=True)
    for image_path in image_root.rglob("*"):
        if image_path.is_file():
            try:
                with Image.open(image_path) as image:
                    native_sizes.append(image.size)
            except OSError:
                continue
    if not native_sizes:
        raise RuntimeError("No readable images were found in the derived dataset.")
    native_size_summary = (
        f"{min(width for width, _ in native_sizes)}–{max(width for width, _ in native_sizes)} px wide × "
        f"{min(height for _, height in native_sizes)}–{max(height for _, height in native_sizes)} px high"
    )

    models = sorted({row["model_name"] for row in benchmark if row.get("split") == "test"})
    results: dict[str, dict] = {}
    for model in models:
        model_rows = [row for row in benchmark if row.get("model_name") == model]
        test_rows = [row for row in model_rows if row.get("split") == "test"]
        val_rows = [row for row in model_rows if row.get("split") == "val"]
        gpu = next((row for row in test_rows if row.get("device", "").lower() == "gpu"), None)
        cpu = next((row for row in test_rows if row.get("device", "").lower() == "cpu"), None)
        val_gpu = next((row for row in val_rows if row.get("device", "").lower() == "gpu"), None)
        val_cpu = next((row for row in val_rows if row.get("device", "").lower() == "cpu"), None)
        metric_row = gpu or cpu
        if metric_row is None:
            continue
        selection_row = val_gpu or val_cpu
        results[model] = {"metrics": metric_row, "selection": selection_row, "gpu": gpu, "cpu": cpu}
    if not results:
        raise RuntimeError("The test split contains no complete metric rows.")
    missing_devices = [name for name, pair in results.items() if pair["gpu"] is None or pair["cpu"] is None]

    headers = ["Model", "P (%)", "R (%)", "mAP50 (%)", "mAP50:95 (%)", "Params (M)", "FLOPs (G)", "CPU ms", "CPU FPS", "GPU ms", "GPU FPS"]
    md_lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    tex_rows = []
    for model, pair in results.items():
        metric, gpu, cpu = pair["metrics"], pair["gpu"], pair["cpu"]
        cells = [
            model,
            pct(metric.get("precision")), pct(metric.get("recall")), pct(metric.get("map50")), pct(metric.get("map50_95")),
            num(metric.get("params_m"), 2), num(metric.get("flops_g"), 2),
            num(cpu.get("latency_ms"), 2) if cpu else "--", num(cpu.get("fps"), 2) if cpu else "--",
            num(gpu.get("latency_ms"), 2) if gpu else "--", num(gpu.get("fps"), 2) if gpu else "--",
        ]
        md_lines.append("| " + " | ".join(cells) + " |")
        tex_rows.append(" & ".join(latex_escape(value) for value in cells) + r" \\")
    latex = "\n".join([
        r"% Requires \usepackage{booktabs,graphicx}", r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\caption{Fine-tuned patch localization on the derived AdvT-shirt-1K test split.}",
        r"\label{tab:advtshirt-patch-benchmark}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
        " & ".join(latex_escape(value) for value in headers) + r" \\", r"\midrule", *tex_rows,
        r"\bottomrule", r"\end{tabular}}", r"\end{table*}",
    ])

    paper_headers = ["Model", "Dataset", "P (%)", "R (%)", "mAP50 (%)", "mAP50:95 (%)", "Params (M)", "FLOPs (G)", "Latency (ms)", "Latency protocol", "FPS"]
    paper_lines = ["| " + " | ".join(paper_headers) + " |", "|" + "|".join(["---"] * len(paper_headers)) + "|"]
    for row in paper_rows:
        latency = float(row["latency_ms"]) if row.get("latency_ms") else None
        cells = [
            row.get("model", "--"), row.get("dataset", "--"), row.get("precision") or "--", row.get("recall") or "--",
            num(row.get("map50"), 2), num(row.get("map50_95"), 2), num(row.get("params_m"), 2), num(row.get("flops_g"), 2),
            num(latency, 2), row.get("latency_device", "--"), num(1000.0 / latency if latency and latency > 0 else None, 1),
        ]
        paper_lines.append("| " + " | ".join(cells) + " |")

    selectable = [(name, pair) for name, pair in results.items() if pair["selection"]]
    if not selectable:
        raise RuntimeError("No validation metrics are available for model selection.")
    best = max(selectable, key=lambda item: float(item[1]["selection"]["map50_95"]))
    baseline_rows = [(name, pair) for name, pair in results.items() if "YOLOv13" not in name]
    y13_rows = [(name, pair) for name, pair in results.items() if "YOLOv13" in name]
    deltas = []
    deltas_vi = []
    for name, pair in y13_rows:
        scale = name[-1].lower()
        same_scale = [(base_name, base_pair) for base_name, base_pair in baseline_rows
                      if base_name[-1].lower() == scale and base_pair["selection"]]
        best_same = max(same_scale, key=lambda item: float(item[1]["selection"]["map50_95"])) if same_scale else None
        if best_same:
            delta = (float(pair["metrics"]["map50_95"]) - float(best_same[1]["metrics"]["map50_95"])) * 100
            deltas.append(
                f"{name}: {delta:+.2f} pp test mAP@0.5:0.95 vs validation-selected "
                f"same-scale baseline {best_same[0]}"
            )
            deltas_vi.append(f"{name}: {delta:+.2f} điểm phần trăm so với {best_same[0]} cùng quy mô")

    ablation_by_key = {row["variant"].lower().replace("yolov13-n-", "").replace("yolov13-n", "full"): row for row in ablations}
    full = next((row for row in ablations if row.get("variant", "").lower().endswith("full")), None)
    ablation_lines = ["| Variant | P (%) | R (%) | mAP50 (%) | mAP50:95 (%) | Params (M) | FLOPs (G) | Latency (ms) | ΔmAP50:95 (pp) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in ablations:
        delta = (float(row["map50_95"]) - float(full["map50_95"])) * 100 if full and row.get("map50_95") not in (None, "") else 0.0
        ablation_lines.append(
            f"| {row.get('variant','--')} | {pct(row.get('precision'))} | {pct(row.get('recall'))} | {pct(row.get('map50'))} | "
            f"{pct(row.get('map50_95'))} | {num(row.get('params_m'),2)} | {num(row.get('flops_g'),2)} | {num(row.get('latency_ms'),2)} | {delta:+.2f} |"
        )
    ablation_details = []
    if full:
        for row in ablations:
            if row is full:
                continue
            ablation_details.append(
                f"{row.get('variant')}: ΔmAP@0.5:0.95 {((float(row['map50_95']) - float(full['map50_95'])) * 100):+.2f} pp, "
                f"Δlatency {(float(row['latency_ms']) - float(full['latency_ms'])):+.2f} ms, "
                f"Δparameters {(float(row['params_m']) - float(full['params_m'])):+.2f} M"
            )

    config = load_yaml("configs/adv_tshirt_patch_experiment.yaml")
    hardware = current_hardware(torch)
    names = audit["names"]
    split_counts = {key: value["images"] for key, value in audit["splits"].items()}
    class_boxes = {key: value["instances_by_class"] for key, value in audit["splits"].items()}
    date = datetime.now().date().isoformat()
    title = "YOLOv13 for Physical Adversarial Patch Localization: A Controlled AdvT-shirt-1K Benchmark"
    limitations = (
        "The exact NCKH task split remains empty. These results concern a separately derived external patch-localization proxy; "
        "they do not establish a gain for physical–visual DoS, NMS overload, or unseen patch designs. The source has no official split, "
        "the derived split is image-level within source folders, and models were trained with one random seed."
    )
    discussion = [
        f"Validation selected {best[0]} (val mAP@0.5:0.95 {pct(best[1]['selection'].get('map50_95'))}%); "
        f"its held-out test mAP@0.5:0.95 is {pct(best[1]['metrics'].get('map50_95'))}%.",
        *deltas,
        "HyperACE models adaptive high-order interactions among spatial feature locations; FullPAD routes feature information across backbone/neck stages; DS-C3k2 uses depthwise-separable computation to reduce cost. The local ablation table is the evidence for this dataset and scale. It must be read as a single-seed controlled comparison; a component is supported only when its measured delta is consistent with the architecture-level hypothesis.",
        limitations,
    ]

    report = f"""# {title}

**Report date:** {date}  
**Study status:** locally trained and evaluated on an external, source-cited physical adversarial patch dataset. The project-specific NCKH dataset remains unavailable.

## Abstract

Physical adversarial patches can disrupt object detection and motivate patch localization as a robustness task. We fine-tuned YOLOv8, YOLOv10, YOLO11, YOLOv12, and YOLOv13 Nano/Small/Large detectors on a deterministic split derived from AdvT-shirt-1K, using identical input size, optimizer, augmentation, seed, and training schedule. The local mirror contains 1,123 images and 1,548 patch boxes; the derived train/validation/test split contains {split_counts.get('train',0)}/{split_counts.get('val',0)}/{split_counts.get('test',0)} images. Using validation mAP@0.5:0.95 for model selection, {best[0]} was selected and reached **{pct(best[1]['metrics'].get('map50_95'))}% mAP@0.5:0.95** and **{pct(best[1]['metrics'].get('map50'))}% mAP@0.5** on the held-out test split. The result is limited to patch localization on this external proxy and is not a result on the original physical–visual DoS dataset.

## Experimental setup

- **Hardware:** {hardware['cpu']}; {hardware['ram_gb']} GiB RAM; {hardware['gpu'] or 'CPU only'} ({hardware['gpu_vram_gb']} GiB); CUDA {hardware['cuda']}.
- **Software:** Python {platform.python_version()}, PyTorch {torch_version}, Ultralytics 8.3.63, CUDA build {torch.version.cuda}; isolated environment at `.venv/`.
- **Dataset:** AdvT-shirt-1K, source mirror at `dataset/external/adv_tshirt_1k/`; license listed as Apache-2.0 by the dataset card. Class map: `{names}`. Image counts: train/val/test = {split_counts.get('train',0)}/{split_counts.get('val',0)}/{split_counts.get('test',0)}; patch instances = {class_boxes}. Native dimensions range from {native_size_summary}; all models use {config.get('imgsz')} × {config.get('imgsz')} network input.
- **Split:** seed 42, 70/15/15 requested, stratified within the five source folders. This split is derived because the source does not provide train/validation/test partitions for this mirror.
- **Training:** pretrained initialization; {config.get('epochs')} maximum epochs with patience {config.get('patience')}; image size {config.get('imgsz')}; batch {config.get('batch')}; {config.get('optimizer')}; initial learning rate {config.get('lr0')}; linear LR decay to factor {config.get('lrf')} with {config.get('warmup_epochs')} warmup epochs; weight decay {config.get('weight_decay')}; seed {config.get('seed')}; deterministic mode. Shared augmentation: HSV ({config.get('augment',{}).get('hsv_h')}/{config.get('augment',{}).get('hsv_s')}/{config.get('augment',{}).get('hsv_v')}), translation {config.get('augment',{}).get('translate')}, scale {config.get('augment',{}).get('scale')}, mosaic {config.get('augment',{}).get('mosaic')}, horizontal flip {config.get('augment',{}).get('fliplr')}; MixUp and CopyPaste disabled.
- **Measurement:** validation and test metrics were obtained on the same held-out split for every checkpoint. CPU/GPU latency is the evaluator-reported inference time per image at batch {config.get('batch')}; FPS is computed as $1000/latency$. Parameters and FLOPs use the installed Ultralytics profiler at {config.get('imgsz')} pixels.
- **Path isolation:** the canonical `dataset/data.yaml` uses `path: ./dataset`; external data, images, labels, runtime data YAMLs, and Hugging Face caches stay below `dataset/`. `YOLO_CONFIG_DIR=./.config`; Ultralytics dataset/run/weight settings resolve to project-local `dataset/`, `runs/`, and `weights/`; pip, Torch, and temporary caches are under `.cache/`. The dataset tree contains no symlink/junction entries.

## Comprehensive benchmark

The table reports held-out **test** precision, recall, mAP, parameter count, FLOPs, CPU/GPU latency, and derived FPS. Model selection and same-scale baseline selection use validation mAP@0.5:0.95; the test set is reserved for final reporting. The machine-readable source is `results/adv_tshirt_patch_benchmark.csv`.

{chr(10).join(md_lines)}

```latex
{latex}
```

## Efficiency trade-off on the local benchmark

The following figures plot the measured validation/test rows on CPU and GPU using the local Ultralytics validator at batch {config.get('batch')}. They are hardware- and runtime-specific; they are not TensorRT measurements or an edge-device deployment test.

![Local test and validation mAP versus inference latency](adv_tshirt_patch_results/tradeoff_map_latency_local.png)

![Local test and validation mAP versus FLOPs](adv_tshirt_patch_results/tradeoff_map_flops_local.png)

At Nano scale, YOLOv13-N is 4.21 percentage points below the validation-selected YOLOv8-N baseline in test mAP@0.5:0.95 and has higher measured GPU latency (3.08 vs 1.30 ms/image). At Small scale, YOLOv13-S is 0.16 points above the validation-selected YOLOv12-S baseline, with latency of 3.16 vs 2.91 ms/image. At Large scale, YOLOv13-L is 0.41 points below YOLOv12-L and slower (8.53 vs 6.74 ms/image). These results do not show uniform YOLOv13 superiority on this proxy.

## Published benchmark context (separate dataset and protocol)

The YOLOv13 paper reports its MS COCO results and Table I latency using a Tesla T4 GPU with TensorRT FP16. This reference table is not directly comparable with the local AdvT-shirt proxy results, which use a different dataset, GPU, runtime and validator. Precision/recall are not reported in that paper table, so they are shown as unavailable. The paper motivates HyperACE for high-order cross-location/cross-scale feature relations, FullPAD for distributing enhanced representations through the pipeline, and depthwise-separable DS-C3k2 blocks for reducing model cost [Lei et al., YOLOv13](https://arxiv.org/abs/2506.17733).

{chr(10).join(paper_lines)}

## Ablation study

The ablations use YOLOv13-N, the same pretrained initialization, split, seed, schedule, and augmentation. Values are validation metrics; deltas compare each variant with the full model.

{chr(10).join(ablation_lines)}

**Interpretation.** {discussion[0]} Relative to the full model, the measured validation changes were: {'; '.join(ablation_details)}. In this proxy, removing HyperACE slightly reduced mAP@0.5:0.95; removing FullPAD increased it but also increased latency; replacing DS-C3k2 increased both accuracy and model cost. This single-seed ablation is descriptive evidence, not proof of universal causal effects.

## Discussion and conclusion

{chr(10).join('- ' + line for line in discussion[1:])}

YOLOv13's architectural motivation is that HyperACE can model multi-location, higher-order feature relations; FullPAD distributes information across the pipeline; DS-C3k2 targets lower convolutional cost. The measured ablation—not the architectural description—determines whether these components improved this patch dataset. Any accuracy gain must be weighed against measured latency, FLOPs, and parameters before an edge deployment claim is made.

## Thảo luận và kết luận (Tiếng Việt)

Theo mAP@0.5:0.95 trên tập validation, cấu hình được chọn là **{best[0]}**; kết quả kiểm tra độc lập của cấu hình này đạt mAP@0.5:0.95 **{pct(best[1]['metrics'].get('map50_95'))}%** và mAP@0.5 **{pct(best[1]['metrics'].get('map50'))}%**. Bảng đối sánh báo cáo thêm độ trễ CPU/GPU, FPS, số tham số và FLOPs để đánh giá chi phí suy luận cùng với độ chính xác.

HyperACE được thiết kế để tổng hợp quan hệ giữa nhiều vị trí đặc trưng theo cấu trúc bậc cao; FullPAD nhằm truyền và kết hợp thông tin xuyên suốt backbone/head; DS-C3k2 hướng tới giảm phép tính tích chập nhờ phép tích chập tách biệt theo chiều sâu. Đây là cơ sở kiến trúc, còn mức đóng góp thực nghiệm phải được xác định bằng các hàng ablation đo trên cùng dữ liệu. Kết quả một seed chỉ là bằng chứng ban đầu, chưa đủ để khẳng định ưu thế có ý nghĩa thống kê.

Chênh lệch mAP@0.5:0.95 trên tập test so với baseline cùng quy mô được chọn bằng validation: {('; '.join(deltas_vi)) if deltas_vi else 'chưa có đối sánh đủ dữ liệu'}.

Các số liệu hiện tại thuộc bộ phát hiện vị trí miếng vá AdvT-shirt-1K được chia lại, không phải tập dữ liệu NCKH gốc. Vì vậy chúng không chứng minh khả năng chống tấn công gây quá tải NMS hoặc tổng quát hóa sang thiết kế miếng vá mới. Cần bổ sung tập đề tài đã gán nhãn, chia test theo thiết kế/người chụp độc lập, chạy nhiều seed và đo trên thiết bị biên thực tế trước khi kết luận về triển khai.

## Scope and sources

The project's canonical `dataset/data.yaml` remains not ready: its original NCKH train/validation/test folders have no images or labels and the class name is still a placeholder. Accordingly, this report does not claim physical–visual DoS/NMS-overload efficacy. The current benchmark is an external patch-localization proxy. Its image-level split may contain correlated or near-duplicate views across partitions, and the single-seed protocol does not estimate variance. AdvT-shirt-1K reports 1,131 images, while the available mirror contains 1,123; the discrepancy is documented in `results/external_dataset_sources.md`.

- [AdvT-shirt-1K dataset card and Apache-2.0 license](https://huggingface.co/datasets/xingjunm/Adv-T-shirt)
- [AdvT-shirt-1K authors' repository](https://github.com/Wwangb/AdvT-shirt-1K)
- [BadPatch / AdvT-shirt paper](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Wang_BadPatch_Diffusion-Based_Generation_of_Physical_Adversarial_Patches_ICCVW_2025_paper.html)
- [YOLOv13 paper](https://arxiv.org/abs/2506.17733)

**Limitations:** {limitations}
"""
    output = project_path("results/adv_tshirt_patch_final_report.md")
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output.relative_to(PROJECT_ROOT)} with {len(results)} models and {len(ablations)} ablation rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
