# PREREG E6-A1 (bổ sung): nhánh `clean640` cho E6

Trạng thái: **ĐÃ DUYỆT 02/11/2026, phương án (a)**; tag `prereg-e6a1-20261102` đặt TRƯỚC khi đo.  Soạn 02/11/2026.
Bổ sung cho `PREREG_E6.md` (tag `prereg-e6-20261101`); không thay đổi bất kỳ endpoint hay quy tắc nào của PREREG_E6.

## Lý do (khuyết điểm thiết kế đã biết)
PREREG_E6 chỉ có nhánh `clean` ở 320 px. Các nhánh C2 (640 px, patch 128) và C4 (640 px, patch 64) đo ở canvas 640 nên tỷ số toàn trình (3,5–3,6)
là chi phí của độ phân giải, không phải tác động của patch (`results/e6/E6_P1cpu2_20261102/summary.json:arms.C2.total_tv_nms`). Khuyết điểm này được phát hiện
SAU khi đo, khi đọc kết quả; ghi vào `DEVIATIONS.md`. Bổ sung này không dựa vào việc tối ưu theo kết quả nào: nó thêm đối chứng còn thiếu.

## Nhánh thêm
- `clean640`: cùng 100 ảnh `val2017` của E6, canvas 640, không patch, không seed (dùng chung cho mọi cột seed như `clean`).
  Mã: thêm khóa `clean640` vào `ARMS` trong `studies/e6_remeasure/run.py` (canvas 640, không patch, không đối chiếu bản ghi lịch sử); các nhánh khác không đổi.
- Đo đúng như PREREG_E6 (N tại τ ∈ {0,01; 0,25}; NMS 4 backend, 2 warm-up + 10 lần; toàn trình `torch_pure`, `tv_nms`, 1 warm-up + 5 lần; xen kẽ, hoán vị seed 3301; tham chiếu xung nhịp mỗi 10 ảnh).
- Để so cặp với C2/C4: chạy lại C2 (100 ảnh × 5 seed) và C4 (50 ảnh × 5 seed) CÙNG LƯỢT với `clean640` trong một run mới (không dùng số C2/C4 của run cũ làm đối chứng, vì khác thời điểm → có thể drift).
  Run cũ giữ nguyên làm bản ghi (không xóa, không ghi đè).

## Endpoint thêm (cùng cách tính với Chính 1–2 của PREREG_E6; bootstrap hai tầng, 20.000 lần, seed 20261021)
- **A1-1**: tỷ số N_patched / max(N_clean640, 1) tại τ = 0,01 và 0,25 cho C2, C4.
- **A1-2**: tỷ số độ trễ toàn trình patched / clean640 cho C2, C4, hai backend; so với MDE `total` của A/A P1_cpu2 hợp lệ gần nhất.
  Quy tắc đọc như PREREG_E6: CI nằm trọn trong ±MDE ⇒ "không phát hiện tác động vượt MDE = x%"; không dùng "tương đương".
- **A1-3 (mô tả)**: sai số dự đoán chỉ-từ-N cho `clean640` và C2/C4 (lưới A = 8400); tương quan hạng giữa ảnh trên `clean640`.
- Không p-value. **MDE 640 px: chọn phương án (a)**: chạy A/A P1_cpu2 ở canvas 640 (`studies/e5_protocol/run.py --profiles P1_cpu2 --canvas 640`, cùng cấu hình E5: 100 ảnh × 20 lần, ABBA/BAAB, 20.000 bootstrap, seed 20261021), ngay TRƯỚC lượt đo A1. MDE dùng cho A1-2 là `total` của A/A 640 này; ngưỡng drift 3%.
  A/A 640 không hợp lệ (ghim FAIL) thì dừng và báo; không dùng MDE 320 thay thế để kết luận "không phát hiện" ở 640 px.
- Duyệt: người dùng duyệt phương án (a) và yêu cầu chạy ngày 02/11/2026 ("làm cả 3"; máy đã được tuyên bố sẵn sàng).

## Điều kiện chạy
- Verify ghim `P1_cpu2` (cận trên + cận dưới 0,5) PASS, nếu không thì dừng. Drift tham chiếu > 3% ghi vào báo cáo, giữ trong phân tích chính, kèm phân tích bỏ khối bị cờ.
- Máy rảnh (không dùng tay, cắm sạc, Balanced không đổi), ước tính ~20 phút cho đo A1 (không tính A/A 640).
- Lệnh: `python -X utf8 studies/e6_remeasure/run.py --run-dir results/e6/E6A1_P1cpu2_<ngày> --profile P1_cpu2 --e7-dir results/e7_gpu/final/runs/e7 --arms clean640,C2,C4 --skip-clean320`;
phân tích: `analyze.py <run> <A/A 640 summary.json> <E1 summary.json> P1_cpu2 --a1` (C2, C4 so với `clean640`; mã đã thêm cờ `--a1`, mặc định không đổi). Thay đổi mã chỉ gồm: nhánh `clean640`, cờ `--skip-clean320`, `--canvas` của E5, cờ `--a1`; các đường mã cũ giữ nguyên.

## Báo cáo
Số đưa vào bài tại Bảng 3 (hàng 640 px) và đoạn Results; bỏ ô "không diễn giải được". Nếu A1-2 ra "không phát hiện", ghi kèm MDE và CI, không viết "không có tác động".
