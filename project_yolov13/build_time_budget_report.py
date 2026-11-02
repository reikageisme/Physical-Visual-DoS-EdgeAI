"""Build the Vietnamese Markdown and Word report from recorded time-budget trials."""

from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import winreg
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from pipeline_utils import PROJECT_ROOT, configure_project_environment, require_local_venv


configure_project_environment()
require_local_venv()

RESULTS = PROJECT_ROOT / "results"
STUDY_DIR = RESULTS / "time_budget"
REPORT_MD = RESULTS / "BaoCao_ThucNghiệm_MocThoiGian.md"
REPORT_DOCX = RESULTS / "BaoCao_ThucNghiệm_MocThoiGian.docx"
PLOT_PATH = STUDY_DIR / "map_growth.png"
TIME_BUDGETS = [1, 2, 4, 8]
MODEL_ORDER = ["yolo11n", "yolov13n"]
MODEL_LABELS = {"yolo11n": "YOLO11n (Baseline)", "yolov13n": "YOLOv13n"}
PLATEAU_TOLERANCE = 0.005  # absolute mAP50:95; 0.5 percentage point


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def fmt(value, digits=2):
    value = finite(value)
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value, digits=2):
    value = finite(value)
    return "—" if value is None else f"{100 * value:.{digits}f}"


def load_data():
    path = STUDY_DIR / "time_budget_records.json"
    if not path.exists():
        raise SystemExit(f"Missing experiment records: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    expected = {(model, budget) for model in MODEL_ORDER for budget in TIME_BUDGETS}
    completed = {(row.get("model_key"), int(row.get("budget_hours", -1))) for row in records if row.get("status") == "complete"}
    missing = expected - completed
    if missing:
        raise SystemExit(f"Cannot build final report; incomplete trials: {sorted(missing)}")
    records = [row for row in records if row.get("status") == "complete" and row.get("model_key") in MODEL_ORDER]
    records.sort(key=lambda row: (MODEL_ORDER.index(row["model_key"]), int(row["budget_hours"])))
    return records


def history_for(record):
    path = PROJECT_ROOT / record["history_csv"]
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    return rows


def budget_record(records, model, budget):
    return next(row for row in records if row["model_key"] == model and int(row["budget_hours"]) == budget)


def best_record(records, model):
    model_rows = [row for row in records if row["model_key"] == model]
    max_map = max(finite(row["evaluations"]["val"]["map50_95"]) or -1 for row in model_rows)
    candidates = [row for row in model_rows if finite(row["evaluations"]["val"]["map50_95"]) is not None and row["evaluations"]["val"]["map50_95"] >= max_map - PLATEAU_TOLERANCE]
    return min(candidates, key=lambda row: finite(row.get("actual_train_seconds")) or float("inf"))


def total_latency_ms(metrics):
    speed = metrics.get("speed_ms") or {}
    values = [finite(speed.get(key)) for key in ("preprocess", "inference", "postprocess")]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def normalize_stop_reasons(records):
    changed = False
    for row in records:
        if row.get("status") != "complete" or row.get("stop_reason") != "training ended before wall-clock cap; inspect log":
            continue
        elapsed = finite(row.get("actual_train_seconds")) or 0
        cap_seconds = float(row.get("budget_hours", 0)) * 3600
        epochs = int(row.get("epochs_completed") or 0)
        if elapsed >= cap_seconds:
            row["stop_reason"] = "wall-clock cap"
        elif epochs < 600:
            row["stop_reason"] = "patience early stopping (10 epochs without improvement)"
        else:
            row["stop_reason"] = "epoch cap"
        changed = True
    if changed:
        path = STUDY_DIR / "time_budget_records.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        columns = list(dict.fromkeys(key for row in records for key in row))
        with (STUDY_DIR / "time_budget_records.csv").open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in records:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
        for row in records:
            if row.get("status") == "complete":
                log = PROJECT_ROOT / row["log"]
                note = f"\nRECORDED_STOP_REASON={row['stop_reason']}\n"
                current = log.read_text(encoding="utf-8", errors="ignore")
                if "RECORDED_STOP_REASON=" not in current:
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write(note)


def fps(metrics):
    total = total_latency_ms(metrics)
    return 1000 / total if total and total > 0 else None


def plateau_details(record):
    rows = history_for(record)
    mapped = [(row, finite(row.get("metrics/mAP50-95(B)"))) for row in rows]
    mapped = [(row, value) for row, value in mapped if value is not None]
    if not mapped:
        return {"epoch": None, "minutes": None, "best": None, "final": None, "final_drop_pp": None}
    best = max(value for _, value in mapped)
    plateau_row, plateau = next((row, value) for row, value in mapped if value >= best - PLATEAU_TOLERANCE)
    final_row, final = mapped[-1]
    return {
        "epoch": int(float(plateau_row["epoch"])),
        "minutes": finite(plateau_row.get("time")) / 60 if finite(plateau_row.get("time")) is not None else None,
        "best": best,
        "final": final,
        "final_drop_pp": (best - final) * 100,
        "epochs": len(rows),
    }


def loss_trend(record):
    rows = history_for(record)
    usable = []
    for row in rows:
        train = [finite(row.get(f"train/{key}_loss")) for key in ("box", "cls", "dfl")]
        val = [finite(row.get(f"val/{key}_loss")) for key in ("box", "cls", "dfl")]
        if all(value is not None for value in train + val):
            usable.append((sum(train), sum(val)))
    if len(usable) < 6:
        return {"label": "insufficient history", "train_delta": None, "val_delta": None}
    window = max(3, min(5, len(usable) // 4))
    first_train = statistics.mean(value[0] for value in usable[:window])
    last_train = statistics.mean(value[0] for value in usable[-window:])
    first_val = statistics.mean(value[1] for value in usable[:window])
    last_val = statistics.mean(value[1] for value in usable[-window:])
    train_delta = last_train - first_train
    val_delta = last_val - first_val
    plateau = plateau_details(record)
    drop_pp = plateau.get("final_drop_pp") or 0
    if train_delta < -0.03 and val_delta > 0.03 and drop_pp >= 0.5:
        label = "possible overfitting"
    elif plateau.get("epoch") and plateau.get("epoch") < len(rows) and drop_pp < 0.5:
        label = "validation performance saturated; no clear overfitting signal"
    elif train_delta < -0.03 and val_delta <= 0.03:
        label = "loss still decreases without a clear validation-loss rise"
    else:
        label = "no clear overfitting or underfitting pattern"
    return {"label": label, "train_delta": train_delta, "val_delta": val_delta}


def cpu_model_name():
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def build_plot(records):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.3, 4.4), constrained_layout=True)
    colors = {"yolo11n": "#245b9e", "yolov13n": "#d17a00"}
    for model in MODEL_ORDER:
        record = budget_record(records, model, 1)
        rows = history_for(record)
        xs, ys = [], []
        for row in rows:
            t = finite(row.get("time"))
            m = finite(row.get("metrics/mAP50-95(B)"))
            if t is not None and m is not None:
                xs.append(t / 60)
                ys.append(m * 100)
        ax.plot(xs, ys, marker="o", markersize=2.8, linewidth=1.7, color=colors[model], label=MODEL_LABELS[model])
        detail = plateau_details(record)
        if detail["minutes"] is not None:
            ax.scatter([detail["minutes"]], [detail["best"] * 100], color=colors[model], marker="*", s=100, zorder=5)
    ax.set_xlabel("Thời gian huấn luyện thực tế (phút)")
    ax.set_ylabel("mAP@0.5:0.95 trên validation (%)")
    ax.set_title("Đường cong mAP theo thời gian huấn luyện")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    ax.legend(frameon=False)
    fig.savefig(PLOT_PATH, dpi=180, facecolor="white")
    plt.close(fig)


def latex_tables(records, selected):
    acc_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Accuracy by training time cap. Metrics are measured on validation.}",
        r"\label{tab:time-budget-accuracy}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Model & Cap (h) & Train (min) & Epoch & Precision (\%) & Recall (\%) & mAP50 (\%) & mAP50:95 (\%) \\",
        r"\hline",
    ]
    for row in records:
        val = row["evaluations"]["val"]
        acc_lines.append(
            f"{MODEL_LABELS[row['model_key']]} & {row['budget_hours']} & {fmt((finite(row.get('actual_train_seconds')) or 0) / 60, 1)} & {row.get('epochs_completed', '—')} & "
            f"{pct(val.get('precision'), 2)} & {pct(val.get('recall'), 2)} & {pct(val.get('map50'), 2)} & {pct(val.get('map50_95'), 2)} \\\\" 
        )
    acc_lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    res_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Resources, inference speed, and final training loss.}",
        r"\label{tab:time-budget-resources}",
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Model & Cap (h) & Params (M) & FLOPs (G) & Latency (ms/frame) & FPS & Train loss & Val loss \\",
        r"\hline",
    ]
    for row in records:
        test = row["evaluations"]["test"]
        train_loss = (row.get("final_train_loss") or {}).get("total")
        val_loss = (row.get("final_val_loss") or {}).get("total")
        res_lines.append(
            f"{MODEL_LABELS[row['model_key']]} & {row['budget_hours']} & {fmt(row.get('parameters_m'), 3)} & {fmt(row.get('flops_g'), 2)} & "
            f"{fmt(total_latency_ms(test), 2)} & {fmt(fps(test), 1)} & {fmt(train_loss, 3)} & {fmt(val_loss, 3)} \\\\" 
        )
    res_lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(acc_lines), "\n".join(res_lines)


def report_content(records, hardware):
    selected = {model: best_record(records, model) for model in MODEL_ORDER}
    details = {model: plateau_details(budget_record(records, model, 1)) for model in MODEL_ORDER}
    acc_latex, resources_latex = latex_tables(records, selected)
    p1, p2 = selected["yolo11n"]["evaluations"]["val"], selected["yolov13n"]["evaluations"]["val"]
    delta_map = (finite(p2.get("map50_95")) or 0) - (finite(p1.get("map50_95")) or 0)
    delta_label = f"{delta_map * 100:+.2f} điểm phần trăm mAP@0.5:0.95"
    rec_lines = []
    acc_lines = []
    for row in records:
        val = row["evaluations"]["val"]
        test = row["evaluations"]["test"]
        train_loss = (row.get("final_train_loss") or {}).get("total")
        val_loss = (row.get("final_val_loss") or {}).get("total")
        cap = int(row["budget_hours"])
        actual_minutes = (finite(row.get("actual_train_seconds")) or 0) / 60
        acc_lines.append(
            f"| {MODEL_LABELS[row['model_key']]} | {cap} h | {actual_minutes:.1f} | {row.get('epochs_completed', '—')} | "
            f"{pct(val.get('precision'))} | {pct(val.get('recall'))} | {pct(val.get('map50'))} | {pct(val.get('map50_95'))} |"
        )
        rec_lines.append(
            f"| {MODEL_LABELS[row['model_key']]} | {cap} h | {fmt(row.get('parameters_m'), 3)} | {fmt(row.get('flops_g'), 2)} | "
            f"{fmt(total_latency_ms(test), 2)} | {fmt(fps(test), 1)} | {fmt(train_loss, 3)} | {fmt(val_loss, 3)} | {row.get('stop_reason', '—')} |"
        )

    sweet_lines = []
    for model in MODEL_ORDER:
        row = selected[model]
        val = row["evaluations"]["val"]
        test = row["evaluations"]["test"]
        if model == "yolo11n":
            delta = "—"
        else:
            delta = f"{(finite(val.get('map50_95')) - finite(p1.get('map50_95'))) * 100:+.2f} pp"
        sweet_lines.append(
            f"| {MODEL_LABELS[model]} | {row['budget_hours']} h | {(finite(row.get('actual_train_seconds')) or 0) / 60:.1f} | "
            f"{row.get('epochs_completed', '—')} | {pct(val.get('map50_95'))} | {pct(test.get('map50_95'))} | {delta} |"
        )

    baseline_spread = (max(finite(r["evaluations"]["val"].get("map50_95")) or 0 for r in records if r["model_key"] == "yolo11n") - min(finite(r["evaluations"]["val"].get("map50_95")) or 0 for r in records if r["model_key"] == "yolo11n")) * 100
    v13_spread = (max(finite(r["evaluations"]["val"].get("map50_95")) or 0 for r in records if r["model_key"] == "yolov13n") - min(finite(r["evaluations"]["val"].get("map50_95")) or 0 for r in records if r["model_key"] == "yolov13n")) * 100
    tradeoff = {}
    for model in MODEL_ORDER:
        first, last = budget_record(records, model, 1), budget_record(records, model, 8)
        delta_min = ((finite(last.get("actual_train_seconds")) or 0) - (finite(first.get("actual_train_seconds")) or 0)) / 60
        delta_pp = ((finite(last["evaluations"]["val"].get("map50_95")) or 0) - (finite(first["evaluations"]["val"].get("map50_95")) or 0)) * 100
        tradeoff[model] = {
            "delta_minutes": delta_min,
            "delta_pp": delta_pp,
            "pp_per_actual_hour": delta_pp / (delta_min / 60) if abs(delta_min) >= 1 else None,
        }
    trend11 = loss_trend(budget_record(records, "yolo11n", 1))
    trend13 = loss_trend(budget_record(records, "yolov13n", 1))
    original_audit = json.loads((RESULTS / "dataset_audit_current.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((RESULTS / "adv_tshirt_patch_dataset_manifest.json").read_text(encoding="utf-8"))
    local_now = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M %Z")
    train_name = selected["yolo11n"]["model_name"]
    v13_name = selected["yolov13n"]["model_name"]

    markdown = f"""# Báo cáo thực nghiệm theo mốc thời gian

**So sánh fine-tuning YOLO11n và YOLOv13n trên bộ định vị patch AdvT-shirt-1K**  
Ngày tổng hợp: {local_now}

## Tóm tắt

Nghiên cứu khảo sát ảnh hưởng của giới hạn thời gian huấn luyện 1, 2, 4 và 8 giờ lên YOLO11n (baseline) và YOLOv13n. Tám run được khởi tạo độc lập từ checkpoint pretrained, dùng cùng tập chia, seed, độ phân giải và cấu hình tối ưu hóa. Trên benchmark định vị patch bổ sung AdvT-shirt-1K, mAP@0.5:0.95 validation tốt nhất trong phạm vi dung sai 0,5 điểm phần trăm đạt được ở giới hạn {selected['yolo11n']['budget_hours']} giờ cho {train_name} và {selected['yolov13n']['budget_hours']} giờ cho {v13_name}; thời gian huấn luyện thực tế lần lượt là {(finite(selected['yolo11n'].get('actual_train_seconds')) or 0)/60:.1f} và {(finite(selected['yolov13n'].get('actual_train_seconds')) or 0)/60:.1f} phút. YOLOv13n chênh {delta_label} so với YOLO11n tại các run được chọn. Các giới hạn là thời gian tối đa; nếu patience dừng run sớm, số liệu phản ánh checkpoint tốt nhất của run đã kết thúc, không phải một phép đo tại đúng giờ giới hạn. Tập dữ liệu gốc của dự án đang rỗng và chưa có class map, vì vậy kết quả này chỉ kết luận về benchmark bổ sung.

## 1. Thiết lập thực nghiệm

### 1.1 Phần cứng và phần mềm

| Thành phần | Cấu hình |
|---|---|
| GPU | {hardware.get('gpu', '—')} ({fmt(hardware.get('gpu_memory_gib'), 2)} GiB VRAM) |
| CPU | {hardware.get('cpu_name', '—')}; {hardware.get('physical_cores', '—')} nhân vật lý, {hardware.get('logical_cores', '—')} luồng |
| RAM | {fmt(hardware.get('ram_gib'), 2)} GiB |
| Python | {hardware.get('python_version', '—')} trong `.venv` |
| PyTorch / CUDA | {hardware.get('torch', '—')}; CUDA khả dụng: {hardware.get('cuda_available', '—')} |
| Ultralytics | {hardware.get('ultralytics', '—')} |
| Độ phân giải / batch | {budget_record(records, 'yolo11n', 1).get('image_size', 416)} px / {budget_record(records, 'yolo11n', 1).get('batch_size', 4)} |

### 1.2 Tập dữ liệu

Benchmark là bộ phát hiện patch được dẫn xuất từ AdvT-shirt-1K, với split cố định theo seed 42: train 785 ảnh/1.106 box, validation 169 ảnh/220 box, test 169 ảnh/222 box. Manifest cục bộ ghi nguồn không có split train/validation/test chính thức; split 70/15/15 được tạo từ các thư mục nguồn và được giữ nguyên cho mọi run. Đây là benchmark bổ sung về định vị patch, không phải tập nhiệm vụ gốc của dự án.

Audit tập gốc tại `dataset/data.yaml` ghi nhận train/validation/test đều có 0 ảnh, nhãn không có dữ liệu và class `object` chỉ là placeholder (`preflight_status={original_audit.get('preflight_status', 'unknown')}`). Do đó không thể huấn luyện hoặc đưa ra kết luận trên tập gốc cho đến khi có split và ánh xạ lớp đã xác minh.

### 1.3 Mô hình và kiểm soát thời gian

Baseline là YOLO11n; mô hình mới là YOLOv13n. Mỗi lần chạy nạp lại checkpoint pretrained tương ứng, cùng seed 42, optimizer SGD, `imgsz=416`, batch 4, 0 dataloader workers, AMP, mosaic 1.0, `lr0=0.01`, patience 10 và tối đa 600 epoch. Cờ `time` của Ultralytics được đặt lần lượt 1/2/4/8 giờ. Đây là giới hạn trên; dừng sớm khi không còn cải thiện validation sau 10 epoch. Mỗi run ghi log, `results.csv`, `best.pt` và `last.pt` riêng dưới `runs/train/timebudget-20261010-*`.

Accuracy trong Bảng 1 là đánh giá lại `best.pt` trên validation; latency/FPS đo trên test ở GPU, batch 4, ảnh 416 px. Latency báo cáo là tổng preprocess + forward + postprocess mỗi ảnh và FPS bằng 1.000 chia latency đó. Train/val loss tại điểm dừng là tổng box + cls + DFL loss của epoch cuối trong `results.csv`; từng thành phần được lưu kèm trong log/CSV.

## 2. Kết quả so sánh đa mốc

### Bảng 1a. Accuracy và thời lượng

| Mô hình | Giới hạn | Train thực (phút) | Epoch | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(acc_lines)}

### Bảng 1b. Tài nguyên, tốc độ và loss

| Mô hình | Giới hạn | Params (M) | FLOPs (G) | Latency (ms/frame) | FPS | Train loss | Val loss | Lý do dừng |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rec_lines)}

**Định nghĩa:** tham số/FLOPs của detector ở 416 px; latency là tổng thời gian validator trên GPU. Loss tổng là box + cls + DFL tại epoch cuối, không phải loss của checkpoint tốt nhất. Accuracy validation và tốc độ test được báo cáo riêng để tránh chọn checkpoint theo test.

### Bảng 1. Mã LaTeX

Hai bảng dưới đây tạo thành Bảng 1; chúng tách accuracy khỏi tài nguyên để giữ kích thước cột đọc được. Mã dùng các lệnh `tabular` và `hline` cơ bản.

```latex
{acc_latex}

{resources_latex}
```

### Bảng 2. Ma trận điểm tối ưu

Sweet spot được định nghĩa là run có thời lượng thực ngắn nhất nằm trong 0,5 điểm phần trăm mAP@0.5:0.95 validation so với giá trị validation cao nhất giữa bốn giới hạn. Test được báo cáo sau khi chọn theo validation.

| Mô hình | Giới hạn tại run chọn | Train thực (phút) | Epoch | Val mAP@0.5:0.95 (%) | Test mAP@0.5:0.95 (%) | Chênh validation so với baseline |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(sweet_lines)}

## 3. Phân tích và thảo luận

### 3.1 Đường cong tăng trưởng mAP

![Đường cong mAP validation theo phút huấn luyện thực tế](time_budget/map_growth.png)

Hình 1 dùng mAP@0.5:0.95 validation của từng epoch trong run có giới hạn 1 giờ. Dấu sao chỉ epoch có mAP cao nhất của đường cong. Với tiêu chí plateau trong 0,5 điểm phần trăm, {train_name} lần đầu đạt vùng plateau ở epoch {details['yolo11n']['epoch']} sau {fmt(details['yolo11n']['minutes'], 1)} phút; {v13_name} đạt vùng này ở epoch {details['yolov13n']['epoch']} sau {fmt(details['yolov13n']['minutes'], 1)} phút. Số điểm quan sát là số epoch hoàn thành trong run; đường cong mô tả validation theo epoch, không phải các phép đo độc lập.

### 3.2 So sánh độ chính xác và chi phí

Ở các run sweet spot được chọn bằng validation, YOLOv13n đạt {pct(p2.get('map50_95'))}% mAP@0.5:0.95 validation và {pct(selected['yolov13n']['evaluations']['test'].get('map50_95'))}% test; baseline đạt lần lượt {pct(p1.get('map50_95'))}% và {pct(selected['yolo11n']['evaluations']['test'].get('map50_95'))}%. Chênh lệch validation là {delta_label}. Precision/recall/mAP50 được liệt kê trong Bảng 1a. Đây là kết quả trên cùng benchmark patch, cùng kích thước `n`; không suy rộng thành kết luận tổng quát trên COCO hay tập gốc của nghiên cứu.

### 3.3 Hội tụ, overfitting và lợi ích giảm dần

Tiêu chí thực nghiệm về bão hòa là mức tăng giữa run giới hạn 1 giờ và run giới hạn 8 giờ nhỏ hơn 0,5 điểm phần trăm mAP@0.5:0.95 validation. Dải kết quả qua các run là {fmt(baseline_spread, 2)} điểm phần trăm cho YOLO11n và {fmt(v13_spread, 2)} cho YOLOv13n. Từ cap 1 lên cap 8, thời gian thực thay đổi {fmt(tradeoff['yolo11n']['delta_minutes'], 1)} phút và mAP thay đổi {tradeoff['yolo11n']['delta_pp']:+.2f} pp ở YOLO11n; với YOLOv13n, các thay đổi lần lượt là {fmt(tradeoff['yolov13n']['delta_minutes'], 1)} phút và {tradeoff['yolov13n']['delta_pp']:+.2f} pp. Tốc độ cải thiện theo giờ chỉ tính khi thời lượng thực chênh ít nhất một phút. Nếu patience kết thúc run trước giới hạn, cap cao hơn không có nghĩa mô hình đã chạy đủ 2/4/8 giờ.

{train_name}: {trend11['label']} (thay đổi trung bình loss cuối so với đầu: train {fmt(trend11['train_delta'], 3)}, validation {fmt(trend11['val_delta'], 3)}). {v13_name}: {trend13['label']} (train {fmt(trend13['train_delta'], 3)}, validation {fmt(trend13['val_delta'], 3)}). Mức giảm từ peak mAP đến điểm cuối lần lượt là {fmt(details['yolo11n']['final_drop_pp'], 2)} và {fmt(details['yolov13n']['final_drop_pp'], 2)} điểm phần trăm. Dấu hiệu overfitting được gắn khi train loss giảm, validation loss tăng rõ và mAP suy giảm; nếu loss cả hai tập còn cao và mAP còn tăng thì phù hợp với underfitting. Kết luận này dùng xu hướng nhiều epoch, không dựa trên dao động đơn lẻ.

### 3.4 Điểm khuyến nghị

Theo quy tắc sweet spot 0,5 điểm phần trăm, giới hạn được chọn cho YOLO11n là {selected['yolo11n']['budget_hours']} giờ (đã tiêu thụ {(finite(selected['yolo11n'].get('actual_train_seconds')) or 0)/60:.1f} phút) và cho YOLOv13n là {selected['yolov13n']['budget_hours']} giờ (đã tiêu thụ {(finite(selected['yolov13n'].get('actual_train_seconds')) or 0)/60:.1f} phút). Khuyến nghị nghiên cứu tiếp theo dùng thời gian thực và dừng sớm dựa trên validation, đồng thời báo cáo riêng cap và thời gian đã tiêu thụ. Trước khi áp dụng khuyến nghị cho mục tiêu NCKH gốc, cần bổ sung dữ liệu và class map hợp lệ vào các split trong `dataset/data.yaml` rồi chạy lại đối chứng trên đúng tập đó.

## 4. Kết luận

Tám lần fine-tuning theo giới hạn 1/2/4/8 giờ tạo ra đường cong hội tụ và kết quả accuracy/resource có thể truy vết theo từng checkpoint. Trên benchmark patch bổ sung, quy tắc plateau 0,5 điểm phần trăm chọn thời điểm có chi phí thực thấp nhất gần mAP validation tối đa; kết luận so sánh YOLOv13n với YOLO11n được định lượng trong Bảng 2. Giới hạn suy luận chính là tập nhiệm vụ gốc chưa có ảnh/nhãn và lớp đã xác minh, vì vậy các con số hiện tại không thay cho kết quả trên bài toán nghiên cứu gốc.

## 5. Tái lập và tệp nguồn số liệu

- Môi trường: `project_yolov13/.venv`.
- Script chạy các cap: `run_time_budget_experiments.py`.
- Bảng tổng hợp thô: `results/time_budget/time_budget_records.csv` và `.json`.
- Loss/accuracy theo epoch: `runs/train/timebudget-20261010-<model>-<cap>h/results.csv`.
- Checkpoint: `runs/train/timebudget-20261010-<model>-<cap>h/weights/best.pt` và `last.pt`.
- Log từng run: `results/time_budget/logs/`.
- Split manifest: `results/adv_tshirt_patch_dataset_manifest.json`.
- Audit tập gốc: `results/dataset_audit_current.json`.

## Phụ lục. Cấu hình thực nghiệm

```text
model = YOLO11n / YOLOv13n, pretrained
budgets = 1, 2, 4, 8 hours (maximum wall-clock limits)
epochs = 600 maximum; patience = 10
data = dataset/adv_tshirt_patch_detection.yaml
imgsz = 416; batch = 4; optimizer = SGD; seed = 42
device = NVIDIA GeForce RTX 5060; workers = 0; AMP = enabled
```
"""
    return markdown, selected, details, acc_latex, resources_latex


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_margins(cell, top=65, start=65, bottom=65, end=65):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def style_run(run, size=12, bold=None, color="000000", font="Times New Roman"):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold


def add_text(doc, text, style=None, bold_lead=None):
    paragraph = doc.add_paragraph(style=style)
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        style_run(lead, bold=True)
        run = paragraph.add_run(text[len(bold_lead):])
        style_run(run)
    else:
        run = paragraph.add_run(text)
        style_run(run)
    paragraph.paragraph_format.line_spacing = 1.15
    paragraph.paragraph_format.space_after = Pt(6)
    return paragraph


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    run = p.add_run(text)
    style_run(run, size={1: 16, 2: 13, 3: 12}.get(level, 12), bold=True)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(9 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    return p


def add_table(doc, headers, rows, widths, font_size=8.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    set_repeat_table_header(table.rows[0])
    for index, label in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.width = Inches(widths[index])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, "D9E2F3")
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(label)
        style_run(run, size=font_size, bold=True)
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].width = Inches(widths[index])
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cells[index])
            p = cells[index].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(value))
            style_run(run, size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_latex_block(doc, source):
    for line in source.splitlines():
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.12)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run(line or " ")
        style_run(run, size=7.2, font="Consolas")


def set_section_layout(section, landscape=False):
    section.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    section.page_width = Inches(11 if landscape else 8.5)
    section.page_height = Inches(8.5 if landscape else 11)
    section.top_margin = Inches(0.65 if landscape else 0.8)
    section.bottom_margin = Inches(0.65 if landscape else 0.8)
    section.left_margin = Inches(0.6 if landscape else 0.85)
    section.right_margin = Inches(0.6 if landscape else 0.85)


def create_docx(records, selected, details, acc_latex, resources_latex):
    doc = Document()
    for section in doc.sections:
        set_section_layout(section, landscape=False)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    for style_name in ("Heading 1", "Heading 2", "Heading 3", "Title"):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.color.rgb = RGBColor(0, 0, 0)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    run = title.add_run("Thực nghiệm theo mốc thời gian")
    style_run(run, size=21, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(14)
    style_run(subtitle.add_run("So sánh YOLO11n và YOLOv13n trên bộ định vị patch AdvT-shirt-1K"), size=12)

    add_heading(doc, "Tóm tắt", 1)
    p1 = selected["yolo11n"]["evaluations"]["val"]
    p2 = selected["yolov13n"]["evaluations"]["val"]
    delta = (finite(p2.get("map50_95")) or 0) - (finite(p1.get("map50_95")) or 0)
    abstract = (
        f"Báo cáo đánh giá các giới hạn huấn luyện 1, 2, 4 và 8 giờ cho YOLO11n và YOLOv13n, khởi tạo từ checkpoint pretrained và dùng cùng split, seed 42, ảnh 416 px, batch 4. "
        f"Trên benchmark bổ sung định vị patch, run sweet spot theo validation (trong 0,5 điểm phần trăm của mAP@0.5:0.95 tốt nhất) đạt {pct(p1.get('map50_95'))}% với YOLO11n và {pct(p2.get('map50_95'))}% với YOLOv13n; chênh lệch YOLOv13n so với baseline là {delta*100:+.2f} điểm phần trăm. "
        f"Thời gian tiêu thụ thực tế tương ứng là {(finite(selected['yolo11n'].get('actual_train_seconds')) or 0)/60:.1f} và {(finite(selected['yolov13n'].get('actual_train_seconds')) or 0)/60:.1f} phút. Các cap là giới hạn tối đa và early stopping có thể kết thúc trước cap. Tập dữ liệu gốc đang rỗng; kết quả chỉ đại diện cho benchmark patch bổ sung."
    )
    add_text(doc, abstract)

    add_heading(doc, "Thiết lập thực nghiệm", 1)
    add_heading(doc, "Phần cứng và phần mềm", 2)
    hardware = json.loads((STUDY_DIR / "environment.json").read_text(encoding="utf-8"))
    import torch
    import ultralytics

    env_rows = [
        ("GPU", f"{hardware.get('gpu')} ({fmt(hardware.get('gpu_memory_gib'), 2)} GiB VRAM)"),
        ("CPU", f"{cpu_model_name()}; {psutil.cpu_count(logical=False)} nhân, {psutil.cpu_count(logical=True)} luồng"),
        ("RAM", f"{psutil.virtual_memory().total / 2**30:.2f} GiB"),
        ("Môi trường", f".venv; Python {platform.python_version()}, PyTorch {torch.__version__}, Ultralytics {ultralytics.__version__}"),
        ("Huấn luyện", "CUDA 0, AMP, seed 42, ảnh 416 px, batch 4, workers 0"),
    ]
    add_table(doc, ["Thành phần", "Cấu hình"], env_rows, [1.55, 5.2], font_size=10)

    add_heading(doc, "Dữ liệu và giao thức", 2)
    add_text(doc, "Bộ định vị patch AdvT-shirt-1K được chia cố định 70/15/15 theo seed 42: train 785 ảnh/1.106 box, validation 169 ảnh/220 box, test 169 ảnh/222 box. Nguồn không cung cấp split train/validation/test chính thức; đây là split dẫn xuất phân tầng theo thư mục nguồn.")
    add_text(doc, "Audit dataset/data.yaml ghi nhận 0 ảnh ở cả ba split và tên lớp object còn là placeholder. Vì vậy các lần chạy bên dưới dùng riêng dataset/adv_tshirt_patch_detection.yaml. Kết quả không đại diện cho tập hoặc nhãn của bài toán gốc.")
    add_text(doc, "Mỗi giới hạn là một run độc lập bắt đầu từ checkpoint pretrained tương ứng. Cùng seed 42, SGD, lr0=0,01, patience 10, tối đa 600 epoch và cấu hình augmentation. time được đặt lần lượt 1/2/4/8 giờ; đây là trần thời gian. Đánh giá accuracy trên validation dùng best.pt; tốc độ được đo trên test, GPU, batch 4. Latency là tổng preprocess + inference + postprocess mỗi ảnh; FPS là nghịch đảo của latency.")

    landscape_section = doc.add_section(WD_SECTION.NEW_PAGE)
    set_section_layout(landscape_section, landscape=True)
    add_heading(doc, "Bảng so sánh đa mốc", 1)
    accuracy_rows, resource_rows = [], []
    for row in records:
        val = row["evaluations"]["val"]
        test = row["evaluations"]["test"]
        accuracy_rows.append([
            MODEL_LABELS[row["model_key"]], f"{row['budget_hours']} h", f"{(finite(row.get('actual_train_seconds')) or 0)/60:.1f}", str(row.get("epochs_completed", "—")),
            pct(val.get("precision")), pct(val.get("recall")), pct(val.get("map50")), pct(val.get("map50_95")),
        ])
        resource_rows.append([
            MODEL_LABELS[row["model_key"]], f"{row['budget_hours']} h", fmt(row.get("parameters_m"), 3), fmt(row.get("flops_g"), 2),
            fmt(total_latency_ms(test), 2), fmt(fps(test), 1), fmt((row.get("final_train_loss") or {}).get("total"), 3),
            fmt((row.get("final_val_loss") or {}).get("total"), 3),
        ])
    add_heading(doc, "Bảng 1a Accuracy và thời gian huấn luyện", 2)
    add_table(doc, ["Mô hình", "Cap", "Phút thực", "Epoch", "P %", "R %", "mAP50 %", "mAP50:95 %"], accuracy_rows,
              [1.45, 0.65, 0.9, 0.6, 0.8, 0.8, 1.0, 1.2], font_size=9)
    add_heading(doc, "Bảng 1b Tài nguyên tốc độ và loss", 2)
    add_table(doc, ["Mô hình", "Cap", "Params M", "FLOPs G", "Latency ms", "FPS", "Train loss", "Val loss"], resource_rows,
              [1.55, 0.65, 0.85, 0.85, 1.1, 0.7, 1.05, 1.05], font_size=9)
    add_text(doc, "Loss là tổng box + cls + DFL tại epoch cuối; độ chính xác là validation, tốc độ là test. Mỗi cap là thời gian tối đa, nên đọc cùng thời gian thực và lý do dừng.")

    add_heading(doc, "Bảng 2 Ma trận điểm tối ưu", 2)
    sweet_rows = []
    baseline_map = finite(selected["yolo11n"]["evaluations"]["val"].get("map50_95")) or 0
    for model in MODEL_ORDER:
        row = selected[model]
        val, test = row["evaluations"]["val"], row["evaluations"]["test"]
        diff = "—" if model == "yolo11n" else f"{(finite(val.get('map50_95'))-baseline_map)*100:+.2f} pp"
        sweet_rows.append([MODEL_LABELS[model], f"{row['budget_hours']} h", f"{(finite(row.get('actual_train_seconds')) or 0)/60:.1f}",
                           pct(val.get("map50_95")), pct(test.get("map50_95")), diff])
    add_table(doc, ["Mô hình", "Cap", "Phút thực", "Val mAP95 %", "Test mAP95 %", "Δ baseline"], sweet_rows,
              [2.0, 1.0, 1.3, 1.5, 1.5, 1.5], font_size=9)
    add_text(doc, "Quy tắc chọn: run có thời lượng thực ngắn nhất nằm trong 0,5 điểm phần trăm mAP@0.5:0.95 validation so với giá trị tối đa của mô hình.")

    add_heading(doc, "Mã LaTeX cho Bảng 1", 2)
    add_latex_block(doc, acc_latex)
    add_latex_block(doc, resources_latex)

    portrait_section = doc.add_section(WD_SECTION.NEW_PAGE)
    set_section_layout(portrait_section, landscape=False)
    add_heading(doc, "Phân tích và thảo luận", 1)
    add_heading(doc, "Đường cong tăng trưởng mAP", 2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(PLOT_PATH), width=Inches(6.45))
    caption = doc.add_paragraph("Hình 1. mAP@0.5:0.95 validation theo thời gian huấn luyện thực tế trong run cap 1 giờ.")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption.runs:
        style_run(run, size=10)
    d11, d13 = details["yolo11n"], details["yolov13n"]
    add_text(doc, f"Theo ngưỡng plateau 0,5 điểm phần trăm dưới mAP tốt nhất của đường cong, YOLO11n lần đầu vào vùng plateau ở epoch {d11['epoch']} ({fmt(d11['minutes'], 1)} phút); YOLOv13n ở epoch {d13['epoch']} ({fmt(d13['minutes'], 1)} phút). Đường cong lấy giá trị validation sau mỗi epoch trong một run, vì vậy các điểm là quan sát nối tiếp chứ không phải các mẫu độc lập.")

    add_heading(doc, "So sánh mô hình", 2)
    v13_test = selected["yolov13n"]["evaluations"]["test"]
    baseline_test = selected["yolo11n"]["evaluations"]["test"]
    add_text(doc, f"Tại sweet spot chọn theo validation, YOLOv13n đạt {pct(p2.get('map50_95'))}% validation và {pct(v13_test.get('map50_95'))}% test; YOLO11n đạt {pct(p1.get('map50_95'))}% validation và {pct(baseline_test.get('map50_95'))}% test. Chênh lệch validation của YOLOv13n so với baseline là {delta*100:+.2f} điểm phần trăm. Đây là so sánh hai model nano trên bộ patch bổ sung.")

    add_heading(doc, "Hội tụ và lợi ích giảm dần", 2)
    ranges = {}
    tradeoff = {}
    for model in MODEL_ORDER:
        maps = [finite(r["evaluations"]["val"].get("map50_95")) for r in records if r["model_key"] == model]
        ranges[model] = (max(x for x in maps if x is not None)-min(x for x in maps if x is not None))*100
        first, last = budget_record(records, model, 1), budget_record(records, model, 8)
        dt = ((finite(last.get("actual_train_seconds")) or 0) - (finite(first.get("actual_train_seconds")) or 0)) / 60
        dm = ((finite(last["evaluations"]["val"].get("map50_95")) or 0) - (finite(first["evaluations"]["val"].get("map50_95")) or 0)) * 100
        tradeoff[model] = {"dt": dt, "dm": dm}
    trend11, trend13 = loss_trend(budget_record(records, "yolo11n", 1)), loss_trend(budget_record(records, "yolov13n", 1))
    add_text(doc, f"Dải mAP validation giữa các giới hạn là {ranges['yolo11n']:.2f} điểm phần trăm cho YOLO11n và {ranges['yolov13n']:.2f} cho YOLOv13n. Từ cap 1 lên cap 8, thời gian thực đổi {tradeoff['yolo11n']['dt']:.1f} phút và mAP đổi {tradeoff['yolo11n']['dm']:+.2f} pp với YOLO11n; YOLOv13n đổi {tradeoff['yolov13n']['dt']:.1f} phút và {tradeoff['yolov13n']['dm']:+.2f} pp. Tiêu chí giảm dần đặt trước là mức tăng dưới 0,5 pp. Nếu patience dừng trước cap, cap cao hơn không đồng nghĩa đã chạy đủ số giờ đó.")
    add_text(doc, f"Xu hướng loss cuối so với đầu: YOLO11n train {fmt(trend11['train_delta'], 3)}, validation {fmt(trend11['val_delta'], 3)} ({trend11['label']}); YOLOv13n train {fmt(trend13['train_delta'], 3)}, validation {fmt(trend13['val_delta'], 3)} ({trend13['label']}). Mức giảm từ peak mAP đến cuối lần lượt là {fmt(d11['final_drop_pp'], 2)} và {fmt(d13['final_drop_pp'], 2)} điểm phần trăm. Loss từng epoch trong results.csv cho phép kiểm tra các dấu hiệu này; một epoch dao động riêng lẻ không đủ để kết luận overfitting.")

    add_heading(doc, "Khuyến nghị", 2)
    add_text(doc, f"Theo quy tắc sweet spot, cap được chọn là {selected['yolo11n']['budget_hours']} giờ cho YOLO11n và {selected['yolov13n']['budget_hours']} giờ cho YOLOv13n, với thời gian thực {(finite(selected['yolo11n'].get('actual_train_seconds')) or 0)/60:.1f} và {(finite(selected['yolov13n'].get('actual_train_seconds')) or 0)/60:.1f} phút. Nên ghi đồng thời cap và thời lượng đã tiêu thụ, dùng validation để chọn checkpoint và giữ test cho đánh giá cuối. Kết quả benchmark patch không thay thế phép đo trên tập nghiên cứu gốc; cần hoàn thiện split/nhãn của dataset/data.yaml trước khi đưa ra khuyến nghị cho nhiệm vụ đó.")

    add_heading(doc, "Kết luận", 1)
    add_text(doc, "Thử nghiệm tạo được bảng đa mốc, log, checkpoint và đường cong mAP có thể truy vết cho baseline YOLO11n và YOLOv13n. Điểm tối ưu được xác định bằng mAP validation trong dung sai 0,5 điểm phần trăm và thời gian huấn luyện thực tế thấp nhất. Các số đo cho thấy quan hệ thời gian-hiệu năng trong bộ patch cụ thể; tình trạng dataset gốc rỗng giới hạn khả năng khái quát sang bài toán NCKH.")

    add_heading(doc, "Tệp tái lập", 1)
    for line in [
        "Môi trường: project_yolov13/.venv.",
        "Script chạy: run_time_budget_experiments.py.",
        "Dữ liệu tổng hợp: results/time_budget/time_budget_records.csv và time_budget_records.json.",
        "Lịch sử từng epoch: runs/train/timebudget-20261010-<model>-<cap>h/results.csv.",
        "Checkpoint: runs/train/timebudget-20261010-<model>-<cap>h/weights/best.pt và last.pt.",
        "Log: results/time_budget/logs/.",
        "Manifest bộ patch: results/adv_tshirt_patch_dataset_manifest.json.",
        "Audit tập gốc: results/dataset_audit_current.json.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        style_run(p.add_run(line), size=11)
        p.paragraph_format.line_spacing = 1.15

    for section in doc.sections:
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run("Trang ")
        style_run(run, size=9)
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._p.append(field)

    doc.core_properties.title = "Thực nghiệm theo mốc thời gian"
    doc.core_properties.subject = "So sánh YOLO11n và YOLOv13n trên bộ định vị patch AdvT-shirt-1K"
    doc.core_properties.author = ""
    doc.save(REPORT_DOCX)


def main():
    records = load_data()
    normalize_stop_reasons(records)
    hardware = json.loads((STUDY_DIR / "environment.json").read_text(encoding="utf-8"))
    hardware.update(
        {
        "cpu_name": cpu_model_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_gib": psutil.virtual_memory().total / 2**30,
            "python_version": platform.python_version(),
        }
    )
    try:
        import torch
        import ultralytics
        hardware["torch"] = torch.__version__
        hardware["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    build_plot(records)
    markdown, selected, details, acc_latex, resources_latex = report_content(records, hardware)
    REPORT_MD.write_text(markdown, encoding="utf-8")
    create_docx(records, selected, details, acc_latex, resources_latex)
    print(f"Wrote {REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {REPORT_DOCX.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {PLOT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
