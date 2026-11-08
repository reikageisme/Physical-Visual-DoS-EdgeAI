# ANALYSIS R3 — Proxy N so với hậu xử lý và toàn trình trên biến thể tham chiếu Overload/Phantom (THĂM DÒ, chỉ đọc)

Ngày: 21/10/2026. Nhãn bắt buộc khi trích: **thăm dò, dữ liệu lịch sử OLD128/reference, không ghim core, mô hình và mối đe dọa
khác SG-GA**. Trước khi viết file này đã xem: trung vị seed 0 (Overload: N 42 → 6075, postprocess 6,44 → 22,40 ms,
total 412,80 → 399,11 ms; Phantom: N 42 → 4121, postprocess 7,80 → 25,82 ms, total 668,88 → 687,77 ms). Không mù dữ liệu.
Script: `sponge-sim/studies/r3_reference/run.py`. Đầu ra: `results/r3/<run_id>/summary.json`.

## Nguồn
`outputs/extended_q1_20261002/reference/{overload,phantom_sponges}_seed{0..9}.json:records` (32 ảnh × {clean, patched} ×
3 mẫu). Giao thức (`reference/manifest.json:protocol`): YOLOv5s, 640, UAP cộng toàn ảnh, chuẩn L2 ≤ 70, τ đánh giá 0,25,
IoU 0,45, max_det 300, CPU, biến thể từ repo Underload (`original_paper_experiment_reproduction: false`).

## Đại lượng (theo ảnh i, seed s; trung vị 3 mẫu cho thời gian)
r_N = N_patched / max(N_clean, 1); r_pp = postprocess_patched / postprocess_clean; r_tot = total_patched / total_clean;
và Δ_pp = postprocess_patched − postprocess_clean (ms), Δ_tot = total_patched − total_clean (ms).
Báo trung bình toàn ma trận [ảnh × seed] và CI hai tầng (20.000 lần, seed 20261021); tỷ phần postprocess/total sạch và có
nhiễu (trung vị). Không p-value.

## Diễn giải được phép
- Nếu r_N lớn nhiều bậc mà CI của r_tot chứa 1: "tăng N nhiều bậc không đi kèm tăng toàn trình phát hiện được trong cấu hình
  CPU này" + MDE/độ rộng CI; KHÔNG nói "không có tác động".
- Không so trực tiếp với SG-GA như cùng mối đe dọa (khác mô hình, kích thước, quyền truy cập, ràng buộc nhiễu).
