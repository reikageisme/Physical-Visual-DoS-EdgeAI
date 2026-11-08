# PREREG E10 — Đo lại nhiễu loạn Phantom Sponges / Overload dưới giao thức ghim lõi; kiểm tiến cứu mô hình Amdahl

Trạng thái: **ĐÃ DUYỆT 02/11/2026** theo chỉ đạo của người dùng ("đại phẫu thuật, máy rảnh, làm tới khi hoàn thiện đầy đủ số liệu") sau review chi tiết của thầy
(`docs/REVIEW_THAY_NAM.md`, mục 2 và 4). Tag `prereg-e10-20261102` đặt TRƯỚC khi đo đầy đủ.
Mã: `studies/e10_reference/run.py`, `analyze.py`; nạp mô hình: `src/sponge_sim/yolov5_ref.py` (chép từ `experiments/extended_reference.py:35-82`, có header SHA256).
Smoke 2 ảnh (02/11, scratchpad `e10_smoke`, 138 s) đã chạy để kiểm mã; số smoke không dùng.

## Câu hỏi
1. Với tập ứng viên THẬT ở vùng 10^3–10^4 (do nhiễu loạn toàn ảnh hộp trắng tạo ra), N và độ trễ toàn trình thay đổi bao nhiêu dưới giao thức ghim lõi?
2. (Chính) Mô hình kiểu Amdahl, chỉ dùng đường cong tiêm tổng hợp E1 và thời gian toàn trình của ảnh sạch, có dự đoán đúng tỷ số toàn trình của các tấn công thật không?
3. Bộ dự đoán thời gian NMS chỉ-từ-N (và N + K) sai bao nhiêu trên các tập ứng viên lớn này?

## Thiết kế
- Mô hình: YOLOv5s của kho tham chiếu Underload (trọng số SHA256 `8b3b748c...`, như lúc sinh nhiễu loạn), float32, CPU, letterbox gốc 640 px.
- Nhiễu loạn: 10 tệp đã lưu (`outputs/extended_q1_20261002/reference`): Overload seed 0–4, Phantom Sponges seed 0–4 (seed 0–4 chọn trước, không theo kết quả);
  SHA256 từng tệp phải khớp bản ghi lúc sinh (lệch thì dừng). Áp dụng như mã gốc: ảnh = clamp(sạch + δ, 0, 1). Đây là các biến thể tham chiếu đã điều chỉnh
  (khác mô hình, kích thước, quyền truy cập so với SG-GA), không phải tái hiện bài gốc.
- Ảnh: 50 ảnh đầu của danh sách giữ ra E5 (`IMAGE_IDS[:50]`). Nhánh mỗi ảnh: clean_a, clean_b (hai bản sạch, A/A trong phiên), 10 nhánh nhiễu loạn; thứ tự hoán vị (seed 10102).
- Mỗi đơn vị: forward không đo -> decode (điểm = obj × max lớp, như mã gốc) -> N tại τ 0,01 và 0,25.
  NMS riêng ở τ 0,25 (ngưỡng đánh giá của mã gốc), không cap, 4 backend (Latin seed 10103), 1 khởi động + 3 lần.
  Toàn trình (forward + decode + τ 0,25 + NMS, một lần gọi) cho tv_nms, tv_batched, ultra_native, 1 khởi động + 3 lần
  (PyTorch thuần chỉ đo NMS riêng vì vòng lặp mất 70–100 ms/lần ở N khoảng 6000).
- Profile `P1_cpu2`, verify ghim đầu và cuối phiên; tham chiếu xung nhịp mỗi 5 ảnh; trong lúc đo trợ lý không chạy việc gì khác (quy tắc 9). Ước tính ~50 phút.

## Endpoint (đơn vị: (ảnh, seed); bootstrap hai tầng ảnh × seed, 20.000 lần, seed 20261021)
- **E10-Chính (Amdahl tiến cứu)**: với mỗi phương pháp × backend toàn trình,
  R_pred = 1 + [f(N_patched) − f(N_clean)] / T_clean, f = nội suy log-log đường cong NMS E1 (nc 80, không cap, hình học mixed; lưới A = 10000 cho N ≥ 100,
  A = 2100 cho N < 100), N = số ứng viên vào NMS ở τ 0,25, T_clean = trung vị toàn trình của clean_a và clean_b cùng ảnh, cùng backend.
  Đại lượng: trung vị |ln(R_đo / R_pred)| và KTC 95%. Quy tắc định trước: mô hình **đạt** cho một ô nếu cận trên KTC < 0,05 (sai dưới khoảng 5%).
  Không tham số nào được khớp trên dữ liệu E10. Lựa chọn hình học mixed là định trước (không chọn sau khi xem).
- **E10-1**: tỷ số N (τ 0,01 và 0,25) nhiễu loạn / sạch, trung vị N, KTC.
- **E10-2**: tỷ số toàn trình nhiễu loạn / sạch theo phương pháp × backend, KTC; so với MDE trong phiên (từ clean_b so với clean_a). Quy tắc đọc như các thí nghiệm trước.
- **E10-3 (mô tả)**: sai số bộ dự đoán NMS (nội suy chỉ-từ-N, M_N, M_NK khớp trên E1 như E9) trên các tập ứng viên của phiên này.
- Không p-value. Kết quả âm giữ nguyên. Sai lệch ghi DEVIATIONS.md.
