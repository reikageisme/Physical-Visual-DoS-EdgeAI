# PREREG E6-A2 (bổ sung): đo các baseline B1–B4 trên 50 ảnh còn lại để bảng patch cân

Trạng thái: **ĐÃ DUYỆT 02/11/2026** theo chỉ đạo của người dùng ("đại phẫu thuật ... làm tới khi hoàn thiện đầy đủ số liệu") sau review chi tiết của thầy
(mục 5: "Bảng 12 không cân: điều kiện chính 100 ảnh × 5 seed, baseline 50 ảnh × 5 seed"). Tag `prereg-e6a2-20261102` đặt trước khi đo.
Không đổi endpoint hay quy tắc của PREREG_E6.

## Thiết kế
- Cùng giao thức, cùng mã E6 (`studies/e6_remeasure/run.py`), thêm hai cờ chỉ ảnh hưởng tập ảnh: `--secondary-offset 50 --only-secondary-images`.
- Nhánh: clean (320 px) + B1 GA tâm, B2 GA vị trí ngẫu nhiên, B3 tìm kiếm ngẫu nhiên, B4 gradient tanh; seed patch 0–4 (như E6: chọn cố định trước khi xem kết quả, là 5 seed đầu trong 10 seed tối ưu).
- Ảnh: 50 ảnh thứ 51–100 của danh sách giữ ra E5 (E6 đã đo 1–50 cho các baseline).
- Lệnh: `python -X utf8 studies/e6_remeasure/run.py --run-dir results/e6/E6A2_P1cpu2_20261102 --profile P1_cpu2 --e7-dir results/e7_gpu/final/runs/e7 --arms B1,B2,B3,B4 --secondary-offset 50 --only-secondary-images`
- Ước tính ~25 phút. Trong lúc đo không chạy việc gì khác (quy tắc 9).

## Phân tích
- Gộp: với mỗi ảnh, tỷ số patched/clean tính trong đúng phiên đo ảnh đó (ảnh 1–50 từ E6, ảnh 51–100 từ E6-A2); bootstrap hai tầng ảnh × seed như PREREG_E6.
  Bảng patch của bài dùng 100 ảnh × 5 seed cho mọi nhánh. Báo thêm kết quả riêng từng nửa (mô tả).
