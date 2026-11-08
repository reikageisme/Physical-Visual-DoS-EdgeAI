# PREREG E6: đo lại patch thật bằng giao thức ghim core, và kiểm "N có dự đoán được độ trễ không"

Trạng thái: **NHÁP, chờ duyệt** (chưa tag). Tag dự kiến: `prereg-e6-<ngày>`.
Mã: `sponge-sim/studies/e6_remeasure/run.py` (đo), `analyze.py` (phân tích). Chạy SAU E1 (cần đường cong E1) và SAU E7 (cần patch).
Yêu cầu thầy: "chạy lại giao thức ghim core trên các patch thật" và "đo ít nhất ... thiết bị ARM" (`docs/REVIEW_THAY_NAM.md`).
Dữ liệu lịch sử (OLD128/X1-500) KHÔNG dùng cho kết luận chính; chỉ nhánh tham khảo R_sg/R_tanh dùng patch cũ để đối chiếu.

## Câu hỏi
Khi đo đúng giao thức (ghim core, A/A, nhiều lần lặp, tách công đoạn): (i) patch SG-GA thật làm N, độ trễ NMS và độ trễ toàn trình thay đổi bao nhiêu;
(ii) **chỉ biết N**, đường cong tiêm tổng hợp (E1) dự đoán độ trễ NMS thật tốt đến đâu, và khi nào sai.

## Phạm vi
Laptop: i5-1335U, profile `P1_cpu2` (logical 2), Windows Balanced, YOLOv8n float32, tensor chuẩn bị sẵn (resize trực tiếp). Điện thoại ARM: cùng script
sau khi PREREG_E5_phone đạt cổng, chỉ nhánh chính và số lần lặp giảm (xem cuối). Một model, một laptop, một điện thoại.

## Tập dữ liệu và nhánh
- Ảnh đánh giá: 100 ảnh `val2017` cố định của E5 (`studies/e5_protocol/run.py:IMAGE_IDS`), tách khỏi tập tối ưu (100 ảnh `train2017`, E7).
- Nhánh chính (100 ảnh × 5 seed): `clean`, `C1` (320 px, 64 px, ngưỡng tối ưu 0,01), `C2` (640 px, 128 px), `C3` (320 px, ngưỡng tối ưu 0,25).
- Nhánh phụ (50 ảnh đầu × 5 seed): `C4` (640 px, 64 px), `B1` GA tâm, `B2` GA vị trí ngẫu nhiên, `B3` tìm kiếm ngẫu nhiên, `B4` gradient tanh, `R_sg`/`R_tanh` (patch cũ 4 ảnh, tham khảo).
- Seed patch 0–4 (5 trong 10 seed của E7; chọn cố định, không chọn theo kết quả). Patch lấy từ `patch_q62500*.npy` của E7; SHA256 phải khớp `patches_index.json` (lệch thì dừng).
- Với `clean` và `R_sg` ở 320 px, N (ngưỡng 0,01) phải bằng bản ghi lịch sử đã khóa; lệch thì dừng (fail-closed). Các nhánh E7 mới không có bản ghi lịch sử để đối chiếu.

## Đo (mỗi đơn vị = ảnh × nhánh × seed)
Forward thật → decode → N tại τ ∈ {0,01; 0,25}; đo riêng lọc-τ và NMS cho 4 backend (xen kẽ vòng tròn, Latin, `prepare()` ngoài vùng đo; 2 warm-up + 10 lần);
đo toàn trình (chèn patch → forward → decode → τ → NMS) cho `torch_pure` và `tv_nms` (1 warm-up + 5 lần). Thứ tự đơn vị trong ảnh hoán vị (seed 3301). Tham chiếu xung nhịp mỗi 10 ảnh.
Vị trí chèn: saliency Laplacian theo ảnh (cùng hàm bản gốc, canvas theo nhánh); tâm/ngẫu nhiên theo nhánh B1/B2.
Verify ghim: cận trên + cận dưới 0,5; FAIL thì không chạy.

## Endpoint (đơn vị suy luận: cặp ảnh × seed; bootstrap hai tầng ảnh × seed, 20.000 lần, seed 20261021)
- **Chính 1 (tác động của patch lên N)**: tỷ số N_patched / max(N_clean, 1) theo ảnh rồi trung bình, tại τ = 0,01 và 0,25, cho C1, C2, C3; CI hai tầng. Đơn vị đếm, không phải độ trễ.
- **Chính 2 (tác động lên toàn trình)**: tỷ số độ trễ toàn trình patched/clean cho C1, C2, C3 và cả hai backend, CI hai tầng; so với MDE `total` của A/A P1_cpu2 (run A/A hợp lệ trước E1).
  Nếu CI nằm trọn trong ±MDE: báo "không phát hiện tác động vượt MDE = x%". Không dùng "tương đương", "không phụ thuộc", "không có tác động".
- **Chính 3 (N có dự đoán độ trễ NMS không? — câu hỏi trung tâm)**: với mỗi backend, ở τ = 0,01, cap None, mọi đơn vị có N > 0 (clean và mọi nhánh patch):
  dự đoán chỉ-từ-N bằng đường cong E1 `P1_cpu2` (nội suy log–log theo N, lưới A = 2100 cho canvas 320, A = 8400 cho canvas 640, nc = 80, cho từng hình học disjoint / dense_cluster / mixed).
  Báo: (a) sai số tuyệt đối trên thang log |ln(đo / dự đoán_mixed)|, trung vị và phân vị 90; (b) tỉ lệ điểm nằm trong **vùng bao** [min, max] của ba hình học; (c) theo từng nhánh.
  Không có ngưỡng "đạt/không" định trước cho (a)–(b): đây là mô tả khoảng cách giữa dự đoán chỉ-từ-N và đo thật. Kết luận "N dự đoán tốt" hay "không" chỉ được viết kèm các con số này và MDE.
- **Phụ**: tương quan hạng trong ảnh ρ(ΔN, ΔNMS) và giữa ảnh (clean) cho từng backend và τ, CI hai tầng (2.000 lần); tỷ số NMS patched/clean; tỷ phần NMS/toàn trình sạch;
  phần thay đổi NMS do patch / toàn trình sạch (headroom); mọi nhánh phụ ở mức mô tả.
- Không p-value, không Holm (không có kiểm định giả thuyết nào trong họ này). Diễn giải thang CI so với MDE.

## Kích thước và thời gian (ước lượng, chưa đo)
Nhánh chính: 100 ảnh × 16 đơn vị = 1.600; nhánh phụ: 50 ảnh × 35 đơn vị = 1.750. Mỗi đơn vị ≈ 1–4 giây (640 px chậm hơn). Ước lượng laptop **≈ 2–3 giờ**; có thể chạy `--arms C1,C2,C3`
trước (≈ 1–1,5 giờ) rồi nhánh phụ. Điện thoại: chỉ nhánh chính, 20 ảnh × 5 seed, 5 lần lặp vi mô ≈ 30–60 phút/profile (sẽ thay bằng số smoke).

## Điều kiện vận hành
Cổng như E5: máy rảnh thật do bạn xác nhận; không dùng máy khi đo; Balanced; cắm sạc. Điện thoại: theo `PREREG_E5_phone.md` (mạng tắt, nghỉ giữa đợt, xen kẽ profile). Mọi sai lệch ghi vào `DEVIATIONS.md`.

## Tiêu chí loại và hạn chế
Không loại điểm đo theo giá trị. Lệch N hoặc SHA256 → dừng toàn bộ run. Hạn chế: 5 seed patch, 100 ảnh, một model, một laptop, tensor chuẩn bị sẵn (không camera, không print-and-capture),
đường cong E1 đo trên laptop nên dự đoán chỉ áp cho laptop; điện thoại cần đường cong E1-phone riêng.

## Đầu ra
`sponge-sim/results/e6/<run_id>/`: `manifest.json` (SHA256 từng patch, mã, trọng số), `raw_<profile>.jsonl`, `summary.json`, `run.log`.
