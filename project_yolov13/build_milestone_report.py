"""Create the Vietnamese milestone study report from completed records."""

from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import winreg
from datetime import datetime

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
MILESTONES = [5, 10, 15, 20, 25, 30]
MODEL_ORDER = ["yolo11n", "yolov13n"]
MODEL_LABELS = {"yolo11n": "YOLO11n (Baseline)", "yolov13n": "YOLOv13n"}
PLATEAU_TOLERANCE = 0.005


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


def load_records():
    path = STUDY_DIR / "milestone_records.json"
    if not path.exists():
        raise SystemExit(f"Chưa có dữ liệu thực nghiệm: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    complete = [row for row in rows if row.get("status") == "complete"]
    expected = {(model, minute) for model in MODEL_ORDER for minute in MILESTONES}
    found = {(row.get("model_key"), int(row.get("nominal_budget_minutes", -1))) for row in complete}
    missing = expected - found
    if missing:
        raise SystemExit(f"Chưa thể lập báo cáo; thiếu mốc hoàn tất: {sorted(missing)}")
    return sorted(complete, key=lambda row: (MODEL_ORDER.index(row["model_key"]), row["nominal_budget_minutes"]))


def by_model_budget(records, model, minutes):
    return next(row for row in records if row["model_key"] == model and int(row["nominal_budget_minutes"]) == minutes)


def history(record):
    path = PROJECT_ROOT / record["history_csv"]
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def histories_by_model(records):
    return {model: history(next(row for row in records if row["model_key"] == model)) for model in MODEL_ORDER}


def checkpoint_plateau(rows):
    mapped = [(row, finite(row.get("metrics/mAP50-95(B)"))) for row in rows]
    mapped = [(row, score) for row, score in mapped if score is not None]
    best_row, best_score = max(mapped, key=lambda pair: pair[1])
    threshold = best_score - PLATEAU_TOLERANCE
    first_row, first_score = next(pair for pair in mapped if pair[1] >= threshold)
    return {
        "best_row": best_row,
        "best_score": best_score,
        "first_near_best_row": first_row,
        "first_near_best_score": first_score,
        "first_near_best_minutes": (finite(first_row.get("time")) or 0) / 60,
    }


def sweet_spot(records, histories, model):
    plateau = checkpoint_plateau(histories[model])
    limit = plateau["best_score"] - PLATEAU_TOLERANCE
    candidates = [
        row for row in records
        if row["model_key"] == model and finite(row.get("selected_val", {}).get("map50_95")) is not None
        and finite(row["selected_val"]["map50_95"]) >= limit
    ]
    chosen = min(candidates, key=lambda row: int(row["nominal_budget_minutes"])) if candidates else None
    return {**plateau, "milestone_record": chosen}


def loss_trend(rows):
    pairs = []
    for row in rows:
        train = [finite(row.get(f"train/{key}_loss")) for key in ("box", "cls", "dfl")]
        val = [finite(row.get(f"val/{key}_loss")) for key in ("box", "cls", "dfl")]
        if all(x is not None for x in train + val):
            pairs.append((sum(train), sum(val)))
    if len(pairs) < 6:
        return {"label": "Thiếu lịch sử để kết luận", "train_change": None, "val_change": None}
    width = min(5, max(3, len(pairs) // 4))
    train_change = statistics.mean(v[0] for v in pairs[-width:]) - statistics.mean(v[0] for v in pairs[:width])
    val_change = statistics.mean(v[1] for v in pairs[-width:]) - statistics.mean(v[1] for v in pairs[:width])
    scores = [finite(row.get("metrics/mAP50-95(B)")) for row in rows]
    best_idx = max(range(len(scores)), key=lambda i: scores[i] if scores[i] is not None else -1)
    drop_pp = ((scores[best_idx] or 0) - (scores[-1] or 0)) * 100
    if train_change < -0.04 and val_change > 0.03 and drop_pp >= 0.5:
        label = "Có dấu hiệu overfitting nhẹ: train loss giảm, val loss tăng và mAP cuối giảm sau đỉnh."
    elif train_change < -0.04 and val_change <= 0.03:
        label = "Loss train và validation cùng giảm hoặc ổn định; chưa thấy tín hiệu overfitting rõ."
    elif best_idx < len(rows) - 1 and drop_pp < 0.5:
        label = "mAP đã gần bão hòa; dao động cuối nhỏ, chưa đủ bằng chứng kết luận overfitting."
    else:
        label = "Chưa thấy dấu hiệu overfitting rõ; cần thận trọng vì validation nhỏ."
    return {"label": label, "train_change": train_change, "val_change": val_change, "mAP_drop_pp": drop_pp}


def cpu_name():
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def hardware_details():
    import torch
    import ultralytics

    env = json.loads((STUDY_DIR / "milestone_environment.json").read_text(encoding="utf-8"))
    return {
        **env,
        "cpu_name": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gib": psutil.virtual_memory().total / 2**30,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }


def build_plot(histories, sweet):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.3, 4.5), constrained_layout=True)
    colors = {"yolo11n": "#245b9e", "yolov13n": "#d17a00"}
    for model in MODEL_ORDER:
        rows = histories[model]
        xs, ys = [], []
        for row in rows:
            elapsed = finite(row.get("time"))
            score = finite(row.get("metrics/mAP50-95(B)"))
            if elapsed is not None and score is not None:
                xs.append(elapsed / 60)
                ys.append(score * 100)
        ax.plot(xs, ys, marker="o", markersize=2.7, linewidth=1.7, color=colors[model], label=MODEL_LABELS[model])
        details = sweet[model]
        best = details["best_row"]
        best_t = (finite(best.get("time")) or 0) / 60
        ax.scatter([best_t], [details["best_score"] * 100], color=colors[model], marker="*", s=125, zorder=4)
    for minute in MILESTONES:
        ax.axvline(minute, color="#8a8a8a", linewidth=0.5, alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_xlabel("Thời gian huấn luyện tích lũy (phút)")
    ax.set_ylabel("mAP@0.5:0.95 validation (%)")
    ax.set_title("Đường cong hội tụ theo thời gian huấn luyện")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
    ax.legend(frameon=False)
    fig.savefig(PLOT_PATH, dpi=190, facecolor="white")
    plt.close(fig)


def latex_tables(records, sweet):
    acc = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Kết quả trên test theo thời điểm huấn luyện; checkpoint được chọn theo validation.}",
        r"\label{tab:milestone-test}", r"\begin{tabular}{llrrrrrrrr}", r"\hline",
        r"Mô hình & Cap (phút) & Thực (phút) & Epoch & P (%) & R (%) & mAP50 (%) & mAP50:95 (%) & Latency (ms) & FPS " + r"\\",
        r"\hline",
    ]
    for row in records:
        test = row["test"]
        acc.append(
            f"{MODEL_LABELS[row['model_key']]} & {row['nominal_budget_minutes']} & {fmt((finite(row.get('selected_elapsed_seconds')) or 0)/60,1)} & {row['selected_epoch']} & "
            f"{pct(test.get('precision'))} & {pct(test.get('recall'))} & {pct(test.get('map50'))} & {pct(test.get('map50_95'))} & "
            f"{fmt(test.get('latency_ms_per_frame'),2)} & {fmt(test.get('fps'),1)} " + r"\\")
    acc += [r"\hline", r"\end{tabular}", r"\end{table}"]

    res = [
        r"\begin{table}[htbp]", r"\centering", r"\caption{Chi phí tài nguyên và loss ở cuối ngân sách.}",
        r"\label{tab:milestone-resources-loss}", r"\begin{tabular}{llrrrrrr}", r"\hline",
        r"Mô hình & Cap (phút) & Params (M) & FLOPs (G) & Epoch cuối & Epoch chọn & Train loss & Val loss " + r"\\",
        r"\hline",
    ]
    for row in records:
        res.append(
            f"{MODEL_LABELS[row['model_key']]} & {row['nominal_budget_minutes']} & {fmt(row.get('parameters_m'),3)} & {fmt(row.get('flops_g'),2)} & "
            f"{row.get('epoch_endpoint','—')} & {row.get('selected_epoch','—')} & {fmt((row.get('endpoint_train_loss') or {}).get('total'),3)} & "
            f"{fmt((row.get('endpoint_val_loss') or {}).get('total'),3)} " + r"\\")
    res += [r"\hline", r"\end{tabular}", r"\end{table}"]

    opt = [
        r"\begin{table}[htbp]", r"\centering", r"\caption{Sweet spot theo mAP validation trong dung sai 0,5 điểm phần trăm.}",
        r"\label{tab:sweet-spot}", r"\begin{tabular}{lrrrrr}", r"\hline",
        r"Mô hình & Cap (phút) & Thời gian ckpt (phút) & Epoch & Val mAP50:95 (%) & Test mAP50:95 (%) " + r"\\", r"\hline",
    ]
    for model in MODEL_ORDER:
        item = sweet[model]["milestone_record"]
        opt.append(
            f"{MODEL_LABELS[model]} & {item['nominal_budget_minutes']} & {fmt((finite(item.get('selected_elapsed_seconds')) or 0)/60,1)} & {item['selected_epoch']} & "
            f"{pct(item['selected_val'].get('map50_95'))} & {pct(item['test'].get('map50_95'))} " + r"\\")
    opt += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(acc), "\n".join(res), "\n".join(opt)


def milestone_curve_rows(records, model):
    rows = [row for row in records if row["model_key"] == model]
    previous = None
    output = []
    for row in rows:
        score = finite(row.get("selected_val", {}).get("map50_95"))
        gain_pp = None if previous is None or score is None else (score - previous[1]) * 100
        delta_minutes = None if previous is None else (row["nominal_budget_minutes"] - previous[0])
        rate = gain_pp / delta_minutes if gain_pp is not None and delta_minutes else None
        output.append((row, gain_pp, rate))
        if score is not None:
            previous = (row["nominal_budget_minutes"], score)
    return output


def report_markdown(records, histories, sweet, hardware, audit, manifest, latex):
    acc_latex, res_latex, opt_latex = latex
    now = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M %Z")
    s11, s13 = sweet["yolo11n"], sweet["yolov13n"]
    selected11 = s11["milestone_record"]
    selected13 = s13["milestone_record"]
    test_delta = (finite(selected13["test"].get("map50_95")) or 0) - (finite(selected11["test"].get("map50_95")) or 0)
    metric_lines = []
    resource_lines = []
    loss_lines = []
    for row in records:
        t = row["test"]
        actual = (finite(row.get("selected_elapsed_seconds")) or 0) / 60
        metric_lines.append(
            f"| {MODEL_LABELS[row['model_key']]} | {row['nominal_budget_minutes']} | {actual:.1f} | {row['selected_epoch']} | "
            f"{pct(t.get('precision'))} | {pct(t.get('recall'))} | {pct(t.get('map50'))} | {pct(t.get('map50_95'))} | "
            f"{fmt(t.get('latency_ms_per_frame'))} | {fmt(t.get('fps'),1)} |"
        )
        resource_lines.append(
            f"| {MODEL_LABELS[row['model_key']]} | {row['nominal_budget_minutes']} | {fmt(row.get('parameters_m'),3)} | {fmt(row.get('flops_g'),2)} | "
            f"{row.get('epoch_endpoint','—')} | {row.get('selected_epoch','—')} |"
        )
        loss_lines.append(
            f"| {MODEL_LABELS[row['model_key']]} | {row['nominal_budget_minutes']} | {fmt((finite(row.get('endpoint_elapsed_seconds')) or 0)/60,1)} | "
            f"{fmt((row.get('endpoint_train_loss') or {}).get('total'),3)} | {fmt((row.get('endpoint_val_loss') or {}).get('total'),3)} | "
            f"{fmt((row.get('endpoint_train_loss') or {}).get('box'),3)} / {fmt((row.get('endpoint_train_loss') or {}).get('cls'),3)} / {fmt((row.get('endpoint_train_loss') or {}).get('dfl'),3)} | "
            f"{fmt((row.get('endpoint_val_loss') or {}).get('box'),3)} / {fmt((row.get('endpoint_val_loss') or {}).get('cls'),3)} / {fmt((row.get('endpoint_val_loss') or {}).get('dfl'),3)} |"
        )

    sweet_lines = []
    for model in MODEL_ORDER:
        detail = sweet[model]
        row = detail["milestone_record"]
        sweet_lines.append(
            f"| {MODEL_LABELS[model]} | {row['nominal_budget_minutes']} | {fmt((finite(row.get('selected_elapsed_seconds')) or 0)/60,1)} | {row['selected_epoch']} | "
            f"{pct(row['selected_val'].get('map50_95'))} | {pct(row['test'].get('map50_95'))} | {fmt(detail['best_score']*100,2)} |"
        )

    curve_lines = []
    for model in MODEL_ORDER:
        for row, gain, rate in milestone_curve_rows(records, model):
            curve_lines.append(
                f"| {MODEL_LABELS[model]} | {row['nominal_budget_minutes']} | {fmt(row['selected_val'].get('map50_95')*100,2)} | "
                f"{fmt(gain,2)} | {fmt(rate,3)} | {row['selected_epoch']} |"
            )

    loss_notes = {model: loss_trend(histories[model]) for model in MODEL_ORDER}
    final_train_summaries = {}
    for model in MODEL_ORDER:
        first = histories[model][0]
        last = histories[model][-1]
        final_train_summaries[model] = {
            "epochs": len(histories[model]),
            "duration": (finite(last.get("time")) or 0) / 60,
            "plateau": sweet[model]["first_near_best_minutes"],
            "best_epoch": sweet[model]["best_row"].get("epoch"),
            "best_val": sweet[model]["best_score"],
            "train_first": sum(finite(first.get(f"train/{k}_loss")) or 0 for k in ("box", "cls", "dfl")),
            "train_last": sum(finite(last.get(f"train/{k}_loss")) or 0 for k in ("box", "cls", "dfl")),
        }

    markdown = f"""# Báo cáo khoa học: Thực nghiệm theo mốc thời gian huấn luyện

**So sánh YOLO11n (baseline) và YOLOv13n trên benchmark định vị patch AdvT-shirt-1K**  
Ngày tổng hợp: {now}

## Tóm tắt

Nghiên cứu khảo sát mối quan hệ giữa thời gian fine-tuning và độ chính xác của YOLO11n và YOLOv13n trên benchmark định vị patch dẫn xuất từ AdvT-shirt-1K. Sau pilot 10 phút mỗi mô hình, các mốc được điều chỉnh từ lưới ban đầu theo giờ sang 5, 10, 15, 20, 25 và 30 phút. Mỗi mô hình được huấn luyện một lần với cùng lịch tối đa 50 epoch, seed 42 và patience 10; checkpoint tốt nhất theo mAP@0.5:0.95 validation đạt được trước mỗi mốc được đánh giá trên test. Theo tiêu chí sớm nhất nằm trong 0,5 điểm phần trăm so với đỉnh validation, sweet spot là **{selected11['nominal_budget_minutes']} phút** cho YOLO11n và **{selected13['nominal_budget_minutes']} phút** cho YOLOv13n. Tại các mốc này, test mAP@0.5:0.95 lần lượt là **{pct(selected11['test'].get('map50_95'))}%** và **{pct(selected13['test'].get('map50_95'))}%** (YOLOv13n − YOLO11n: {test_delta*100:+.2f} điểm phần trăm). Các kết quả chỉ áp dụng cho benchmark patch bổ sung; tập dữ liệu gốc của dự án chưa có ảnh/nhãn để huấn luyện.

## 1. Thiết lập thực nghiệm

### 1.1 Phần cứng và môi trường

| Thành phần | Cấu hình |
|---|---|
| GPU | {hardware.get('gpu','—')} ({fmt(hardware.get('gpu_memory_gib'),2)} GiB VRAM) |
| CPU | {hardware.get('cpu_name','—')}; {hardware.get('physical_cores','—')} nhân vật lý, {hardware.get('logical_cores','—')} luồng |
| RAM | {fmt(hardware.get('ram_gib'),2)} GiB |
| Môi trường | `.venv`; Python {hardware.get('python_version','—')}, PyTorch {hardware.get('torch','—')}, Ultralytics {hardware.get('ultralytics','—')} |
| Thiết lập | CUDA 0, AMP, 416×416, batch 4, 0 workers, seed 42 |

### 1.2 Dữ liệu và phạm vi kết luận

Benchmark là split cố định của AdvT-shirt-1K cho phát hiện/định vị patch: train 785 ảnh/1.106 box, validation 169 ảnh/220 box, test 169 ảnh/222 box. Manifest nguồn ghi nhận split này được dẫn xuất cục bộ; đây không phải split benchmark chính thức. Kết quả vì vậy phản ánh bài toán định vị patch trên split này.

Audit `dataset/data.yaml` ghi nhận train/validation/test đều có 0 ảnh và tên lớp `object` là placeholder (trạng thái: `{audit.get('preflight_status','unknown')}`). Không thể suy rộng kết quả sang nhiệm vụ gốc trước khi cung cấp tập dữ liệu hợp lệ và ánh xạ lớp đã xác minh.

### 1.3 Mô hình và giao thức thời gian

YOLO11n được dùng làm baseline cỡ nano, YOLOv13n là mô hình cần so sánh. Cả hai được khởi tạo từ pretrained weights và dùng cùng SGD (`lr0=0,01`, momentum 0,937, weight decay 0,0005), ảnh 416, batch 4, AMP, augmentation, seed 42, tối đa 50 epoch và patience 10. Một trajectory cố định cho mỗi mô hình được lưu theo epoch; Ultralytics `time` không bật trong lượt chính để các mốc dùng chung lịch learning rate. Mốc 5/10/15/20/25/30 phút được ánh xạ tới epoch cuối cùng có thời gian tích lũy không vượt mốc. Ở mỗi mốc, chọn checkpoint có validation mAP@0.5:0.95 cao nhất trong các epoch đã hoàn tất; nếu run đã dừng sớm, các mốc dài hơn dùng checkpoint cuối sẵn có và được ghi nhận theo thời gian thực tế.

Pilot cap 10 phút đạt 17 epoch/600,9 giây với YOLO11n (mAP validation tốt nhất 80,88%) và 13 epoch/607,6 giây với YOLOv13n (80,86%). Pilot phục vụ định vị thô; Ultralytics tự ước lượng số epoch theo cap thời gian. Các số pilot không gộp vào Bảng 1. Trong lượt thử ban đầu, YOLO11n cap 1 giờ dừng do patience sau 71 epoch/35,2 phút; cap 2 giờ bị ngắt ở epoch 11, còn cap 4/8 giờ và YOLOv13n chưa chạy. Các run này dùng cap thời gian với lịch tối đa 600 epoch nên được loại khỏi so sánh fixed-50 để tránh trộn lịch learning rate.

### 1.4 Chỉ số

Precision, recall, mAP@0.5 và mAP@0.5:0.95 trong Bảng 1 đo trên test ở batch 4 sau khi checkpoint được chọn bằng validation. Latency được đo riêng trên checkpoint sweet spot của mỗi kiến trúc, batch 1, ba lần; báo cáo trung vị tổng preprocess + inference + postprocess mỗi ảnh. Cùng latency đại diện cho các mốc của cùng kiến trúc; FPS = 1.000/latency. Parameters và FLOPs báo cáo theo kiến trúc tại 416 px. Loss tại mỗi mốc là tổng box + cls + DFL ở epoch cuối đã hoàn thành trước cap, không phải loss của checkpoint validation tốt nhất.

## 2. Kết quả

### Bảng 1. Độ chính xác test và tốc độ theo thời gian

| Mô hình | Mốc cap (phút) | Thời gian checkpoint (phút) | Epoch chọn | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Latency (ms/frame) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

### Bảng 1a. Mã LaTeX: kết quả test

```latex
{acc_latex}
```

### Bảng 1b. Tài nguyên mô hình

| Mô hình | Mốc cap (phút) | Parameters (M) | FLOPs (G) | Epoch endpoint | Epoch checkpoint |
|---|---:|---:|---:|---:|---:|
{chr(10).join(resource_lines)}

### Bảng 1c. Loss tại cuối mốc thời gian

| Mô hình | Cap (phút) | Thời gian endpoint (phút) | Train loss tổng | Val loss tổng | Train box/cls/dfl | Val box/cls/dfl |
|---|---:|---:|---:|---:|---|---|
{chr(10).join(loss_lines)}

### Mã LaTeX cho bảng tài nguyên và loss

```latex
{res_latex}
```

### Bảng 2. Ma trận điểm tối ưu

Sweet spot là mốc cap nhỏ nhất mà checkpoint tốt nhất đến mốc đó nằm trong 0,5 điểm phần trăm mAP@0.5:0.95 validation so với đỉnh validation của toàn trajectory. Tiêu chí này chọn theo validation; test chỉ dùng để báo cáo hiệu năng của checkpoint đã chọn.

| Mô hình | Cap sweet spot (phút) | Thời gian checkpoint (phút) | Epoch | Val mAP@0.5:0.95 (%) | Test mAP@0.5:0.95 (%) | Đỉnh val toàn run (%) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(sweet_lines)}

#### Mã LaTeX cho Bảng 2

```latex
{opt_latex}
```

## 3. Phân tích và thảo luận

### 3.1 Đường cong mAP theo thời gian

![Đường cong mAP validation theo thời gian](map_growth.png)

Hình thể hiện mAP@0.5:0.95 validation sau mỗi epoch. Dấu sao là đỉnh validation của từng trajectory; các vạch dọc chỉ các mốc khảo sát. Các đường có thể dao động do tập validation nhỏ (169 ảnh), vì vậy lựa chọn checkpoint dùng validation tốt nhất tích lũy đến mốc thay vì coi một epoch cuối đơn lẻ là đại diện.

| Mô hình | Mốc (phút) | Best-so-far val mAP@0.5:0.95 (%) | Δ so với mốc trước (điểm %) | Mức tăng biên (điểm %/phút) | Epoch checkpoint |
|---|---:|---:|---:|---:|---:|
{chr(10).join(curve_lines)}

### 3.2 Sweet spot và hiệu quả thời gian

YOLO11n đạt đỉnh validation {pct(s11['best_score'])}% ở epoch {s11['best_row']['epoch']} (khoảng {s11['first_near_best_minutes']:.1f} phút đến checkpoint đầu tiên nằm trong dung sai 0,5 điểm phần trăm). YOLOv13n đạt đỉnh {pct(s13['best_score'])}% ở epoch {s13['best_row']['epoch']} (khoảng {s13['first_near_best_minutes']:.1f} phút đến vùng tương đương). Theo các cap khảo sát, điều này tương ứng sweet spot {selected11['nominal_budget_minutes']} phút và {selected13['nominal_budget_minutes']} phút. YOLOv13n cần ít thời gian hơn để đạt vùng đỉnh theo tiêu chí đã định; chênh lệch test mAP@0.5:0.95 tại sweet spot là {test_delta*100:+.2f} điểm phần trăm so với baseline. Tổng training dừng ở 15,25 phút với YOLOv13n và 30,40 phút với YOLO11n; checkpoint sweet spot xuất hiện lần lượt ở 8,6 và 25,4 phút. Latency batch 1 của YOLOv13n là 29,85 ms/frame (33,5 FPS), so với 15,27 ms/frame (65,5 FPS) của baseline. Do đó YOLOv13n đạt vùng validation đỉnh sớm hơn và có ít tham số/FLOPs, nhưng tại sweet spot có test mAP@0.5:0.95 thấp hơn và latency cao hơn trên GPU này.

Sau đỉnh validation, YOLO11n tiếp tục khoảng 5,0 phút và YOLOv13n khoảng 6,7 phút trước khi patience dừng run, nhưng không đạt mAP cao hơn. Các mốc dài hơn chỉ lặp checkpoint cuối. Đây là dấu hiệu diminishing returns trong cấu hình và split đã thử, không phải khẳng định phổ quát cho mọi dữ liệu.

### 3.3 Hội tụ, loss, overfitting và underfitting

{{loss11}}

{{loss13}}

Train loss giảm từ {final_train_summaries['yolo11n']['train_first']:.3f} xuống {final_train_summaries['yolo11n']['train_last']:.3f} với YOLO11n và từ {final_train_summaries['yolov13n']['train_first']:.3f} xuống {final_train_summaries['yolov13n']['train_last']:.3f} với YOLOv13n. Hai run hoàn tất lần lượt {final_train_summaries['yolo11n']['epochs']} và {final_train_summaries['yolov13n']['epochs']} epoch; thời gian training tích lũy là {final_train_summaries['yolo11n']['duration']:.1f} và {final_train_summaries['yolov13n']['duration']:.1f} phút. Loss giảm cùng với mAP cao ở vùng đỉnh nên không thấy dấu hiệu underfitting rõ trong benchmark này. Cũng chưa có bằng chứng overfitting mạnh vì val loss giảm; tuy vậy validation chỉ có 169 ảnh và mAP dao động, nên nhận định cần xem là mô tả chứ không phải kiểm định thống kê.

### 3.4 Trần hiệu năng và khuyến nghị

Mốc khảo sát dày 5 phút phù hợp hơn lưới 1/2/4/8 giờ vì các trajectory đã đạt vùng đỉnh validation trong vài đến vài chục phút. Với cấu hình hiện tại, dùng 10 phút làm screening ban đầu cho YOLOv13n; dùng đến 30 phút khi cần kiểm tra đầy đủ vùng plateau của YOLO11n. Các nghiên cứu tiếp theo nên lặp ít nhất 3 seed, mở rộng sang split/dataset đúng của nhiệm vụ gốc, và đo latency riêng ở batch 1 trên thiết bị edge đích. Khi dữ liệu hoặc augmentation thay đổi, cần hiệu chỉnh lại các mốc.

## 4. Kết luận

Thử nghiệm pilot đã thay lưới cap dài theo giờ bằng các mốc 5–30 phút, dựa trên thời gian hội tụ quan sát được. Giao thức chính dùng chung một lịch 50 epoch cho mỗi mô hình và checkpoint từng epoch; các mốc được đánh giá theo best-so-far validation, sau đó báo cáo test. Sweet spot theo dung sai 0,5 điểm phần trăm là {selected11['nominal_budget_minutes']} phút với YOLO11n và {selected13['nominal_budget_minutes']} phút với YOLOv13n. Kết luận về mô hình chỉ có phạm vi của benchmark patch dẫn xuất; cần tập dữ liệu gốc hợp lệ và nhiều seed trước khi đưa ra khuyến nghị triển khai rộng.

## 5. Tệp và khả năng tái lập

- Kết quả theo mốc: `results/time_budget/milestone_records.json` và `.csv`.
- Cấu hình môi trường: `results/time_budget/milestone_environment.json`.
- Log: `results/time_budget/logs/milestones-20261010-*-fixed50.log`.
- CSV theo epoch/checkpoint: `runs/train/milestones-20261010-*-fixed50/`.
- Pilot: `results/time_budget/pilot/pilot_records.json` và thư mục log/checkpoint tương ứng.
- Manifest split: `results/adv_tshirt_patch_dataset_manifest.json`.
- Audit dữ liệu gốc: `results/dataset_audit_current.json`.

## Phụ lục. Cấu hình

```text
models = YOLO11n pretrained; YOLOv13n pretrained
milestones = 5, 10, 15, 20, 25, 30 minutes
one fixed-schedule run/model; epochs=50; patience=10; save_period=1
imgsz=416; batch=4; seed=42; optimizer=SGD; AMP=True; workers=0
dataset=dataset/adv_tshirt_patch_detection.yaml
checkpoint selection = highest validation mAP50:95 observed by each milestone
test evaluation = after validation-based selection; one test split, 169 images
```
"""
    markdown = markdown.replace("{loss11}", f"**YOLO11n:** {loss_notes['yolo11n']['label']} (Δ train loss theo các cửa sổ đầu/cuối: {fmt(loss_notes['yolo11n']['train_change'],3)}; Δ val loss: {fmt(loss_notes['yolo11n']['val_change'],3)}).")
    markdown = markdown.replace("{loss13}", f"**YOLOv13n:** {loss_notes['yolov13n']['label']} (Δ train loss theo các cửa sổ đầu/cuối: {fmt(loss_notes['yolov13n']['train_change'],3)}; Δ val loss: {fmt(loss_notes['yolov13n']['val_change'],3)}).")
    return markdown


def set_section(section, landscape=False):
    section.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    section.page_width = Inches(11 if landscape else 8.5)
    section.page_height = Inches(8.5 if landscape else 11)
    section.top_margin = Inches(0.62 if landscape else 0.75)
    section.bottom_margin = Inches(0.62 if landscape else 0.75)
    section.left_margin = Inches(0.55 if landscape else 0.8)
    section.right_margin = Inches(0.55 if landscape else 0.8)


def style_run(run, size=12, bold=None, font="Times New Roman", color="000000"):
    run.font.name = font
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold


def add_paragraph(doc, text="", bold=False, size=12, align=None, style=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    style_run(r, size=size, bold=bold)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(4)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    r = p.add_run(text)
    style_run(r, size={1: 16, 2: 13, 3: 12}.get(level, 12), bold=True)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(7 if level == 1 else 4)
    p.paragraph_format.space_after = Pt(3)
    return p


def shade_cell(cell, color="D9E2F3"):
    props = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    props.append(shd)


def margin_cell(cell, top=55, left=55, bottom=55, right=55):
    props = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for edge, value in (("top", top), ("start", left), ("bottom", bottom), ("end", right)):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        mar.append(node)
    props.append(mar)


def add_table(doc, headers, data, widths, font_size=8.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    header = table.rows[0]
    tr_pr = header._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)
    for i, text in enumerate(headers):
        cell = header.cells[i]
        cell.width = Inches(widths[i])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        shade_cell(cell)
        margin_cell(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        style_run(p.add_run(str(text)), size=font_size, bold=True)
    for values in data:
        cells = table.add_row().cells
        for i, text in enumerate(values):
            cells[i].width = Inches(widths[i])
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            margin_cell(cells[i])
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            style_run(p.add_run(str(text)), size=font_size)
    return table


def create_docx(records, histories, sweet, hardware, audit, manifest, latex):
    doc = Document()
    set_section(doc.sections[0], landscape=False)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(4)
    for name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        doc.styles[name].font.name = "Times New Roman"
        doc.styles[name]._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        doc.styles[name].font.color.rgb = RGBColor(0, 0, 0)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    style_run(title.add_run("THỰC NGHIỆM THEO MỐC THỜI GIAN HUẤN LUYỆN"), size=20, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(14)
    style_run(subtitle.add_run("So sánh YOLO11n và YOLOv13n trên benchmark định vị patch AdvT-shirt-1K"), size=12)

    add_heading(doc, "Tóm tắt", 1)
    s11 = sweet["yolo11n"]["milestone_record"]
    s13 = sweet["yolov13n"]["milestone_record"]
    delta = (finite(s13["test"].get("map50_95")) or 0) - (finite(s11["test"].get("map50_95")) or 0)
    abstract = (
        f"Nghiên cứu đánh giá mối quan hệ giữa thời gian fine-tuning và hiệu năng của YOLO11n (baseline) và YOLOv13n trên benchmark định vị patch dẫn xuất từ AdvT-shirt-1K. Pilot 10 phút mỗi mô hình dẫn tới các mốc cuối 5, 10, 15, 20, 25 và 30 phút. Mỗi mô hình có một trajectory cố định tối đa 50 epoch, seed 42, patience 10 và checkpoint từng epoch; tại mỗi mốc chọn best validation mAP@0.5:0.95 rồi đánh giá trên test. Sweet spot trong dung sai 0,5 điểm phần trăm là {s11['nominal_budget_minutes']} phút cho YOLO11n và {s13['nominal_budget_minutes']} phút cho YOLOv13n. Test mAP@0.5:0.95 tương ứng {pct(s11['test'].get('map50_95'))}% và {pct(s13['test'].get('map50_95'))}%, chênh lệch {delta*100:+.2f} điểm phần trăm. Kết quả chỉ áp dụng cho benchmark bổ sung; tập dữ liệu gốc chưa có ảnh và nhãn hợp lệ."
    )
    add_paragraph(doc, abstract)

    add_heading(doc, "1. Thiết lập thực nghiệm", 1)
    add_heading(doc, "1.1 Phần cứng và phần mềm", 2)
    env_rows = [
        ("GPU", f"{hardware.get('gpu')} ({fmt(hardware.get('gpu_memory_gib'),2)} GiB VRAM)"),
        ("CPU", f"{hardware.get('cpu_name')}; {hardware.get('physical_cores')} nhân vật lý, {hardware.get('logical_cores')} luồng"),
        ("RAM", f"{fmt(hardware.get('ram_gib'),2)} GiB"),
        ("Môi trường", f".venv; Python {hardware.get('python_version')}, PyTorch {hardware.get('torch')}, Ultralytics {hardware.get('ultralytics')}"),
        ("Cấu hình", "CUDA 0, AMP, 416×416, batch 4, workers 0, seed 42"),
    ]
    add_table(doc, ["Thành phần", "Thông số"], env_rows, [1.6, 5.25], font_size=10)

    add_heading(doc, "1.2 Dữ liệu và phạm vi", 2)
    add_paragraph(doc, "Benchmark định vị patch dùng split dẫn xuất AdvT-shirt-1K: train 785 ảnh/1.106 box, validation 169/220 box, test 169/222 box. Split được giữ cố định nhưng không phải split chính thức.")
    add_paragraph(doc, f"Audit dataset/data.yaml: 0 ảnh trong các split, class object là placeholder (trạng thái {audit.get('preflight_status','unknown')}). Kết quả chỉ đại diện benchmark patch bổ sung, không phải nhiệm vụ gốc.")

    add_heading(doc, "1.3 Mô hình, mốc thời gian và lựa chọn checkpoint", 2)
    add_paragraph(doc, "YOLO11n là baseline, YOLOv13n là mô hình mới; cả hai dùng pretrained weights, SGD, ảnh 416, batch 4, AMP, seed 42, tối đa 50 epoch và patience 10. Mỗi mô hình có một trajectory với lịch learning rate cố định và checkpoint từng epoch. Tại mốc 5–30 phút chọn best-so-far validation; sau early stopping, mốc dài hơn dùng checkpoint cuối và ghi thời gian thực.")
    add_paragraph(doc, "Pilot 10 phút: YOLO11n 17 epoch/600,9 giây; YOLOv13n 13 epoch/607,6 giây. Pilot chỉ dùng hiệu chỉnh lưới; cap 1/2/4/8 giờ ban đầu đã dừng và bị loại khỏi kết quả chính.")

    add_heading(doc, "1.4 Chỉ số", 2)
    add_paragraph(doc, "Accuracy đo trên test ở batch 4, sau chọn checkpoint bằng validation. Latency đo batch 1 ba lần tại sweet spot; bảng dùng trung vị preprocess + inference + postprocess. FPS = 1.000/latency. Parameters/FLOPs đo ở 416 px; loss cuối cap là box + cls + DFL của epoch endpoint.")

    landscape = doc.add_section(WD_SECTION.NEW_PAGE)
    set_section(landscape, landscape=True)
    add_heading(doc, "2. Kết quả so sánh đa mốc", 1)
    main_rows = []
    res_rows = []
    loss_rows = []
    opt_rows = []
    for row in records:
        test = row["test"]
        main_rows.append((
            MODEL_LABELS[row["model_key"]], str(row["nominal_budget_minutes"]),
            fmt((finite(row.get("selected_elapsed_seconds")) or 0)/60,1), str(row["selected_epoch"]),
            pct(test.get("precision")), pct(test.get("recall")), pct(test.get("map50")), pct(test.get("map50_95")),
            fmt(test.get("latency_ms_per_frame")), fmt(test.get("fps"),1),
        ))
        res_rows.append((MODEL_LABELS[row["model_key"]], str(row["nominal_budget_minutes"]), fmt(row.get("parameters_m"),3), fmt(row.get("flops_g"),2), str(row.get("epoch_endpoint")), str(row.get("selected_epoch"))))
        loss_rows.append((
            MODEL_LABELS[row["model_key"]], str(row["nominal_budget_minutes"]), fmt((finite(row.get("endpoint_elapsed_seconds")) or 0)/60,1),
            fmt((row.get("endpoint_train_loss") or {}).get("total"),3), fmt((row.get("endpoint_val_loss") or {}).get("total"),3),
        ))
    for model in MODEL_ORDER:
        row = sweet[model]["milestone_record"]
        opt_rows.append((MODEL_LABELS[model], str(row["nominal_budget_minutes"]), fmt((finite(row.get("selected_elapsed_seconds")) or 0)/60,1), str(row["selected_epoch"]), pct(row["selected_val"].get("map50_95")), pct(row["test"].get("map50_95"))))
    add_heading(doc, "Bảng 1. Độ chính xác test và tốc độ", 2)
    add_table(doc, ["Mô hình", "Cap\n(phút)", "Thực\n(phút)", "Epoch", "P\n(%)", "R\n(%)", "mAP50\n(%)", "mAP50:95\n(%)", "Latency\n(ms)", "FPS"], main_rows, [1.38,0.58,0.62,0.45,0.58,0.58,0.68,0.78,0.72,0.46], font_size=8.0)
    add_paragraph(doc, "Mỗi hàng là checkpoint validation tốt nhất đạt được không muộn hơn cap; thời gian thực ghi theo epoch của checkpoint đó. Accuracy đo trên test.", size=9)
    add_heading(doc, "Bảng 1b. Độ phức tạp mô hình", 2)
    add_table(doc, ["Mô hình", "Cap (phút)", "Params (M)", "FLOPs (G)", "Epoch endpoint", "Epoch chọn"], res_rows, [2.0,1.15,1.35,1.35,1.55,1.55], font_size=9)
    add_heading(doc, "Bảng 1c. Loss ở cuối cap", 2)
    add_table(doc, ["Mô hình", "Cap (phút)", "Endpoint (phút)", "Train loss", "Val loss"], loss_rows, [2.0,1.2,1.55,1.5,1.5], font_size=9)
    add_heading(doc, "Bảng 2. Ma trận điểm tối ưu", 2)
    add_table(doc, ["Mô hình", "Cap (phút)", "Ckpt (phút)", "Epoch", "Val mAP95 (%)", "Test mAP95 (%)"], opt_rows, [2.0,1.2,1.4,0.7,1.55,1.55], font_size=9)
    doc.add_section(WD_SECTION.NEW_PAGE)
    set_section(doc.sections[-1], landscape=False)

    add_heading(doc, "3. Phân tích và thảo luận", 1)
    add_heading(doc, "3.1 Đường cong mAP theo thời gian", 2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(PLOT_PATH), width=Inches(6.5))
    add_paragraph(doc, "Đường thể hiện mAP@0.5:0.95 validation theo từng epoch; dấu sao đánh dấu đỉnh. Các mốc cap là 5, 10, 15, 20, 25 và 30 phút. Vì validation có 169 ảnh, dao động theo epoch được xử lý bằng tiêu chí best-so-far đến mỗi mốc.")

    add_heading(doc, "3.2 Sweet spot và lợi ích biên", 2)
    s11, s13 = sweet["yolo11n"], sweet["yolov13n"]
    train_min_11 = next(r for r in records if r["model_key"] == "yolo11n")["training_actual_seconds"] / 60
    train_min_13 = next(r for r in records if r["model_key"] == "yolov13n")["training_actual_seconds"] / 60
    latency_11 = s11["milestone_record"]["test"].get("latency_ms_per_frame")
    latency_13 = s13["milestone_record"]["test"].get("latency_ms_per_frame")
    add_paragraph(doc, f"YOLO11n đạt đỉnh validation {pct(s11['best_score'])}% ở epoch {s11['best_row']['epoch']}; YOLOv13n đạt {pct(s13['best_score'])}% ở epoch {s13['best_row']['epoch']}. Cap nhỏ nhất trong dung sai 0,5 điểm phần trăm lần lượt là {s11['milestone_record']['nominal_budget_minutes']} và {s13['milestone_record']['nominal_budget_minutes']} phút. YOLOv13n đạt vùng đỉnh sớm hơn; test mAP@0.5:0.95 tại sweet spot chênh {(finite(s13['milestone_record']['test'].get('map50_95'))-finite(s11['milestone_record']['test'].get('map50_95')))*100:+.2f} điểm phần trăm so với baseline. Training dừng ở {train_min_13:.2f} phút với YOLOv13n và {train_min_11:.2f} phút với YOLO11n; latency batch 1 lần lượt là {fmt(latency_13)} và {fmt(latency_11)} ms/frame. YOLOv13n đạt vùng đỉnh sớm hơn nhưng chậm hơn khi suy luận trên GPU này.")
    for model in MODEL_ORDER:
        trend = loss_trend(histories[model])
        add_paragraph(doc, f"{MODEL_LABELS[model]}: {trend['label']} Δ train loss={fmt(trend.get('train_change'),3)}, Δ val loss={fmt(trend.get('val_change'),3)}.")
    add_paragraph(doc, "Sau đỉnh validation, YOLO11n tiếp tục khoảng 5,0 phút và YOLOv13n khoảng 6,7 phút trước khi patience dừng run, nhưng không đạt mAP cao hơn. Đây là diminishing returns trong cấu hình này; cap vượt thời điểm early stopping lặp checkpoint cuối và thời gian thực được ghi riêng.")

    add_paragraph(doc, "Loss train và validation đều giảm đến cuối; không thấy underfitting rõ ở split này và chưa có bằng chứng overfitting mạnh. Vì validation chỉ có 169 ảnh, các dấu hiệu này được xem mô tả và cần kiểm chứng bằng nhiều seed.")

    add_heading(doc, "3.3 Khuyến nghị", 2)
    recommendation = add_paragraph(doc, "Với cấu hình thử nghiệm này, dùng 10 phút làm vòng screening cho YOLOv13n và cho phép baseline YOLO11n tới khoảng 30 phút để chạm vùng đỉnh. Lưới 5 phút phân giải đủ giai đoạn tăng nhanh và plateau; không có căn cứ để tiếp tục đặt cap theo giờ cho benchmark nhỏ này. Cần lặp nhiều seed, dùng dữ liệu đúng bài toán gốc, và đo riêng latency batch 1 trên phần cứng edge đích trước khi kết luận triển khai.")
    recommendation.paragraph_format.keep_together = True

    add_heading(doc, "4. Kết luận", 1)
    add_paragraph(doc, f"Pilot thực nghiệm cho phép thay lưới giờ bằng lưới 5–30 phút. Với fixed schedule và checkpoint theo epoch, sweet spot validation theo ngưỡng đã định là {s11['milestone_record']['nominal_budget_minutes']} phút cho YOLO11n và {s13['milestone_record']['nominal_budget_minutes']} phút cho YOLOv13n. Báo cáo kèm test metrics, tài nguyên, latency, loss và log có thể tái lập. Các kết quả chỉ thuộc benchmark định vị patch dẫn xuất; tập gốc rỗng nên cần dữ liệu hợp lệ trước khi ngoại suy.")
    add_heading(doc, "Tệp kết quả", 2)
    add_paragraph(doc, "Milestone records: results/time_budget/milestone_records.json và .csv; log: results/time_budget/logs/; checkpoint/lịch sử: runs/train/milestones-20261010-*-fixed50/; pilot: results/time_budget/pilot/.", size=10)
    add_paragraph(doc, "Lượt cap giờ thăm dò: YOLO11n cap 1h dừng patience ở 71 epoch/35,2 phút; cap 2h dừng thủ công ở epoch 11; cap 4/8h và YOLOv13n chưa chạy. Loại khỏi bảng chính vì lịch 600 epoch khác lịch fixed-50.", size=9)

    doc.core_properties.title = "Thực nghiệm theo mốc thời gian huấn luyện"
    doc.core_properties.subject = "So sánh YOLO11n và YOLOv13n"
    doc.core_properties.author = ""
    REPORT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(REPORT_DOCX)


def main():
    records = load_records()
    histories = histories_by_model(records)
    sweet = {model: sweet_spot(records, histories, model) for model in MODEL_ORDER}
    hardware = hardware_details()
    audit = json.loads((RESULTS / "dataset_audit_current.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / "adv_tshirt_patch_dataset_manifest.json").read_text(encoding="utf-8"))
    build_plot(histories, sweet)
    latex = latex_tables(records, sweet)
    markdown = report_markdown(records, histories, sweet, hardware, audit, manifest, latex)
    REPORT_MD.write_text(markdown, encoding="utf-8")
    create_docx(records, histories, sweet, hardware, audit, manifest, latex)
    print(f"Created {REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"Created {REPORT_DOCX.relative_to(PROJECT_ROOT)}")
    print(f"Created {PLOT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
