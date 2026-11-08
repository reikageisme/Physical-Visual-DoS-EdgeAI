# PREREG E2b — Proxy N_τ trên ảnh thật + patch thật, đo theo giao thức E5 (XÁC NHẬN)

Trạng thái: **NHÁP, chờ duyệt** (chưa tag). Tag dự kiến: `prereg-e2b-20261021`. Chạy SAU E1.
Script: `studies/e2_proxy/e2b_run.py`, `e2b_analyze.py RUN_DIR AA_SUMMARY` (AA_SUMMARY = run A/A P1_cpu2 dùng cho E1).

## Câu hỏi
Khi đo đúng giao thức (ghim core, nhiều lần lặp, tách công đoạn, 4 backend), thay đổi N_τ do patch SG-GA thật gây ra có
dự đoán được thay đổi độ trễ NMS và toàn trình không? Và độ trễ NMS chiếm bao nhiêu phần toàn trình (headroom H2)?
E2a (thăm dò, lịch sử) cho ρ trong ảnh ≈ 0,12 nhưng bị nhiễu đo chi phối; E2b là phép xác nhận.

## Phạm vi
CPU laptop ghim core (P1_cpu2 = logical 2), Windows Balanced, YOLOv8n 320×320, patch chèn số. Cohort X1-500; 100 ảnh của E5
(`studies/e5_protocol/run.py:IMAGE_IDS`, seed chọn 20261021) — tập con của 500, nên không độc lập với X1-500.

## Thiết kế
- Điều kiện: clean + sg_ga q2500 seed 0–9 (`outputs/x1_500img_20261018/optimization/sg_ga_seed{s}/patch_q2500.npy`),
  vị trí patch = `location` trong record X1-500 tương ứng. Forward mới tái sinh decoded; **fail-closed**: nếu active_candidates
  (τ = 0,01, cap None) khác record X1-500 → dừng, không chạy tiếp (smoke 2 ảnh × 3 điều kiện: khớp 6/6).
- Đo vi mô: 4 backend × cap {None, 100}; 2 warm-up + 10 lần/ô; backend xen kẽ vòng tròn, thứ tự Latin (seed 2202);
  thứ tự điều kiện trong ảnh hoán vị (seed 2201). Đo riêng `tau_ns` và `nms_ns` (prepare() ngoài vùng đo).
- Toàn trình: load → insert → forward → decode → τ → NMS (torch_pure, tv_nms), 1 warm-up + 5 lần, đo trực tiếp.
- Tham chiếu mỗi 10 ảnh + cuối phiên (mô tả, không loại). Verify ghim: cận trên + cận dưới 0,5; FAIL → không chạy.
- Ước lượng: ~20–35 phút (smoke: ~1 s/đơn vị ảnh×điều kiện, 1.100 đơn vị).

## Endpoint (đơn vị: cặp ảnh × seed; bootstrap hai tầng ảnh × seed, seed 20261021)
- **Chính 1 (proxy trong ảnh)**: với mỗi backend (cap None): ρ_Spearman(ΔN, Δt), ΔN = N_patch − N_clean, Δt = trung vị NMS
  patch − trung vị NMS clean cùng ảnh. CI hai tầng (2.000 lần). Báo cạnh ρ giữa ảnh (clean) để kiểm Simpson (đảo dấu hay chỉ khác
  độ lớn). Holm trong họ 4 backend chỉ áp dụng nếu có kiểm định; ở đây báo CI, không p-value.
- **Chính 2 (tác động lên toàn trình)**: tỷ số toàn trình patch/clean theo (ảnh, seed), trung bình, CI hai tầng 20.000 lần,
  với torch_pure và tv_nms. So với MDE `total` của A/A P1_cpu2. Nếu CI nằm trong ±MDE: báo "không phát hiện tác động vượt
  MDE = x%"; KHÔNG nói "không có tác động" hay "tương đương".
- **Phụ**: H2 headroom = trung vị NMS clean / toàn trình clean; phần thay đổi NMS do patch / toàn trình clean (CI hai tầng);
  tỷ số NMS patch/clean theo backend × cap; ΔN (mean, CI); mô hình dự đoán log t ~ log(N+1) chỉ-N (gộp backend) so với
  N + backend, GroupKFold 5 gập theo ảnh (MAE log). Không suy nhân quả từ tương quan.

## Tiêu chí loại
Không loại theo giá trị. Lệch N → dừng toàn bộ run (không loại riêng ảnh). Verify FAIL → không chạy.

## Đầu ra
`sponge-sim/results/e2b/<run_id>/`: manifest.json, raw_P1_cpu2.jsonl, summary.json, run.log.
