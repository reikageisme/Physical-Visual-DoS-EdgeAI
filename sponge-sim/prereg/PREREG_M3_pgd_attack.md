# PREREG_M3 — Tấn công thật: PGD hộp trắng đẩy số đối tượng giữ lại K (model phân đoạn)

Soạn 06/11/2026. CHƯA tag. Tag khi quyết định chạy: `prereg-m3-<ngày>` (commit + tag TRƯỚC khi chạy sinh tấn công).
Cần GPU (PGD nhiều vòng). Bỏ GA (đã cho thấy không hơn tìm kiếm ngẫu nhiên ở bài cũ).

## Lý do
M0/M1 cho thấy chi phí dựng mask tăng theo K (đối tượng giữ lại sau NMS), K* ≈ 50 (full-res). Nhưng đó là K TIÊM.
M3 kiểm điều còn thiếu: **một tấn công thật có đẩy K lên gần max_det=300 không**, để biến chi phí mask thành tấn công sẵn sàng thật.

## Giả thuyết
- H1: tồn tại nhiễu đối kháng (ngân sách L∞) làm K (sau NMS) tăng từ K_sạch (vài–vài chục) lên gần 300 trên YOLOv8n-seg.
- H2 (chuyển giao): nhiễu tối ưu trên YOLOv8n-seg có đẩy K trên YOLO26-seg (không NMS) không. Đăng ký trước cả hai kết cục:
  chuyển được (chi phí ở mask, không ở NMS) HOẶC không (thiết kế một-một của model không-NMS chặn K).

## Endpoint
- **M3-P1 (chính):** K_best (trung vị theo ảnh của K thật sau NMS tại nhiễu tốt nhất) so với K_sạch. Thành công nếu K_best ≥ K* của M1
  (tức đủ để làm toàn trình chậm vượt sàn nhiễu) ở ≥ nửa số ảnh.
- **M3-P2:** quỹ đạo K theo vòng PGD (mô tả hội tụ). **M3-P3:** chuyển giao YOLO26-seg (áp cùng nhiễu, đếm K).
- Số K là K THẬT đếm bằng `non_max_suppression`, KHÔNG phải proxy. Proxy chỉ dùng làm hàm mục tiêu khả vi.

## Phương pháp
- **PGD hộp trắng, ngân sách L∞** trên ảnh (mặc định toàn ảnh; tuỳ chọn patch = chỉ nhiễu trong một hộp). eps, alpha, steps đăng ký trước.
- **Hàm mục tiêu khả vi (proxy cho K):** preds = model(x) → [1, 4+nc+nm, A]; c_i = max điểm lớp của anchor i (đã sigmoid trong forward).
  N_soft = Σ_i sigmoid((c_i − τ)/T) = số ứng viên mềm vượt ngưỡng. PGD **tăng** N_soft. Lưới anchor rải đều không gian nên nhiều anchor
  cùng vượt ngưỡng → nhiều hộp sống sót rời nhau sau NMS → K tăng. (Tuỳ chọn: thêm số hạng khuyến khích hộp nhỏ để giảm chồng lấn.)
  N_soft là PROXY; mọi kết luận K dùng số đếm thật. Nếu proxy không đẩy được K thật, ghi kết quả âm và thử proxy có số hạng phân tán IoU.
- **EOT:** v1 không EOT (nhiễu số, không print-camera) — ghi rõ phạm vi. EOT là mở rộng sau.
- **Đánh giá:** mỗi `eval_every` vòng, đếm K thật; giữ nhiễu có K thật lớn nhất. Lưu nhiễu tốt nhất để đưa vào M1/M2 đo thời gian thật.

## Thiết kế
- Model: YOLOv8n-seg (tối ưu); YOLO26-seg (chỉ đánh giá chuyển giao, không tối ưu lại). Ảnh: ≥ 20 (dùng IMAGE_IDS như M1).
- Thiết bị: GPU (device=cuda). Smoke 1 ảnh/ít vòng chạy được trên CPU chỉ để kiểm cơ chế, KHÔNG kết luận.
- Tất định: seed ghi lại; nhiễu khởi tạo 0.

## Quy tắc quyết định
- K_best ≥ K* (M1) ở ≥ nửa số ảnh → tấn công thật đạt (bề mặt lỗ hổng hiện thực hoá). Đưa nhiễu vào M1/M2 đo toàn trình thật.
- Không đạt → kết quả âm trung thực: proxy/PGD chưa đẩy đủ K; thử proxy phân tán, hoặc kết luận K khó đẩy trên cấu hình này
  (vẫn là phát hiện có giá trị, song song kết quả GA cũ).

## Tiên quyết (khi có GPU)
- [ ] torch+cuda+ultralytics; `SPONGE_REPO_ROOT`; dataset + `yolov8n-seg.pt` (+ yolo26-seg nếu có cho chuyển giao).
- [ ] Smoke 1 ảnh vài vòng (scratchpad) để ước thời gian/VRAM trước khi chạy đủ.

## Lệnh (SAU khi tag; trên GPU)
```
python -X utf8 studies/m3_attack/run.py --run-dir results/m3/M3_<ngày> --device cuda
python -X utf8 studies/m3_attack/analyze.py --run-dir results/m3/M3_<ngày>
```
