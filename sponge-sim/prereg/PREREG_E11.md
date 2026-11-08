# PREREG_E11 — Đo lại bảng patch với bước chèn patch ngoài vùng bấm giờ

Ngày soạn: 03/11/2026. Tag dự kiến: `prereg-e11-20261103` (tag TRƯỚC khi đo đầy đủ). Mã: `studies/e11_patch_remeasure/run.py`, `analyze.py`.

## Lý do
Phép đo toàn trình của E6/E6-A1/E6-A2 (`studies/e6_remeasure/run.py:198-205`) gọi `insert` (chèn patch) trong vùng bấm giờ
cho nhánh có patch nhưng không cho nhánh sạch. Chẩn đoán post hoc (`results/diag_insert/insert_cost_20261103.json`): bước chèn tốn
khoảng 0,35 ms ở 320 px, 1,66 ms ở 640 px (patch 128) và 0,53 ms ở 640 px (patch 64). Hiệu chỉnh bằng phép trừ (từ phiên khác)
không được dùng làm kết quả chính. E11 thay thế toàn bộ cột "toàn trình" của bảng patch.

**Đã thấy dữ liệu:** các tỷ số E6 chưa hiệu chỉnh và đã hiệu chỉnh đã được xem (DEVIATIONS 03/11). Các giả thuyết dưới đây vì vậy
là xác nhận lại trên phép đo đã sửa, không phải giả thuyết mù.

## Sửa mã (đã kiểm bằng `tests/test_e11_symmetry.py`)
Ảnh đầu vào (đã chèn hoặc sạch) dựng một lần NGOÀI vùng đo (`build_input`); mọi nhánh đo cùng hàm `pipeline_fn`
(forward → decode → lọc τ=0,01 → NMS). Chọn đưa ra ngoài thay vì chèn giả ở nhánh sạch vì bước chèn là thao tác dựng kịch bản,
không thuộc pipeline của bộ phát hiện.

## Thiết kế (cân)
- 100 ảnh: `IMAGE_IDS[:100]` của E5/E6 (cùng tập). Mọi nhánh trên CÙNG 100 ảnh.
- Seed patch 0–4 cho mọi nhánh. Bằng chứng chọn trước: `PREREG_E6.md` dòng 20 trong tag `prereg-e6-20261101`
  (commit 9130774, 01/11/2026 18:11 +0700) ghi "Seed patch 0–4 (5 trong 10 seed của E7; chọn cố định…)"; phép đo E6 đầu tiên
  bắt đầu 16:41Z cùng ngày (`results/e6/E6_P1cpu2_20261101/manifest.json: created_utc`), sau tag. `PREREG_E7.md` dòng 51 ghi
  thứ tự sinh patch seed-major và seed 0–4 là tập E6 cần. Không đổi tập seed.
- Nhánh: `clean_a`, `clean_b` (320 px), `clean640_a`, `clean640_b` (640 px), C1, C2, C3, C4, B1, B2, B3, B4 (×5 seed).
  Bỏ R_sg/R_tanh (patch cũ, chỉ tham khảo, không có trong bảng của bài).
- Patch đóng băng: `results/e7_gpu/final/runs/e7/<điều kiện>/seed<k>/patch_q62500*.npy`; SHA-256 phải khớp `patches_index.json`,
  lệch hoặc thiếu thì dừng toàn bộ.
- Đo vi mô giống E6: τ ∈ {0,01; 0,25}, 4 backend, 2 warm-up + 10 lần, khối Latin (seed 4402). Toàn trình: torch_pure và tv_nms,
  1 warm-up + 5 lần, thứ tự backend Latin (seed 4403). Thứ tự đơn vị trong ảnh hoán vị (seed 4401). Tham chiếu drift mỗi 10 ảnh và cuối.
- Profile P1_cpu2 (logical CPU 2), kiểm ghim dưới tải (effective ≥ 0,5, không đạt thì dừng); torch 1 luồng.
- Fail-closed: N(τ=0,01) của clean_a/clean_b phải bằng bản ghi X1-500 đã khóa.

## Endpoint (đơn vị: cặp ảnh × seed; bootstrap hai tầng 20.000 lần, seed 20261021)
Mẫu số sạch = trung bình clean_a và clean_b cùng canvas, cùng ảnh, cùng phiên.
- E11-P1 (chính): tỷ số toàn trình R của từng nhánh, backend tv_nms và torch_pure, KTC 95%.
- E11-P2: sàn nhiễu trong phiên MDE_tot = mde_paired(SD log(clean_b/clean_a) theo ảnh, n=100) cho từng canvas × backend;
  t_total sạch (trung vị) cùng phiên; ngưỡng Δt* = MDE_tot × t_total.
- E11-P3: Δt_NMS (vi mô, τ=0,01) và dự đoán Amdahl R̂ = 1 + Δt_NMS / t_total.
- Phụ: tỷ số N (τ 0,01; 0,25), tỷ số NMS vi mô 4 backend.

## Quy tắc quyết định
- Một nhánh "làm chậm toàn trình phát hiện được" khi cận dưới KTC 95% của R > 1 + MDE_tot (trong phiên).
- H1: với tv_nms, không nhánh GA nào (C1–C4, B1, B2) đạt quy tắc trên.
- H2: B4 với torch_pure đạt quy tắc trên (Δt_NMS dự kiến vượt Δt*); B4 với tv_nms không đạt.
- H3: R̂ nằm trong KTC 95% của R ở ít nhất 14/16 ô (8 nhánh × 2 backend).
- Báo song song MDE của A/A riêng (E5: 0,55% ở 320 px; A/A 640 lần 2: 0,69%) để đối chiếu; quy tắc chính dùng MDE trong phiên.

## Loại điểm / độ nhạy
Không loại theo giá trị. Phân tích độ nhạy định trước: bỏ các khối có điểm tham chiếu đầu khối bị cờ drift (> 3%)
(`analyze.py --exclude-flagged`). Nếu > 3/11 điểm tham chiếu bị cờ: báo cả hai, coi kết quả chính là kém tin cậy, đề xuất đo lại.

## Thời gian ước tính
3.200 đơn vị ở 320 px (~1 s) + 1.200 đơn vị ở 640 px (~2 s) ≈ 1,5–2,5 giờ. Trong lúc đo không chạy gì khác (quy tắc 9).

## Hạn chế
Một laptop, một lõi logic, Windows Balanced, tần số không khoá; một model (YOLOv8n); 100 ảnh; 5 seed patch; tensor đã chuẩn bị.
