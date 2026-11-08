# PREREG_E11b — Đo lại bảng patch lần 2 (bước chèn ngoài vùng bấm giờ), CHỈ CHẠY KHI MÁY RẢNH THẬT

Soạn 03/11/2026. Chưa tag. Tag khi quyết định chạy: `prereg-e11b-<ngày>` (commit + tag TRƯỚC khi đo).

## Lý do
E11 lần 1 (`results/e11/E11_P1cpu2_20261103`, tag `prereg-e11-20261103`) bị loại theo quy tắc PREREG_E11: 11/11 điểm tham chiếu bị cờ drift,
sàn nhiễu toàn trình trong phiên 7,5% ở 320 px. Bài hiện báo toàn trình của patch dưới dạng DỰ ĐOÁN của mô hình ngân sách.
E11b thay dự đoán bằng số đo nếu đạt tiêu chí chất lượng dưới đây.

## Giống hệt PREREG_E11
Mã `studies/e11_patch_remeasure/run.py` và `analyze.py` (không sửa), cùng 100 ảnh, seed patch 0–4 (bằng chứng chọn trước:
`PREREG_E6.md:20` trong tag `prereg-e6-20261101`), cùng nhánh, cùng số lần lặp, cùng endpoint (E11-P1..P3), cùng quy tắc quyết định H1–H3,
cùng phân tích độ nhạy (`--exclude-flagged`).

## Tiêu chí chất lượng (định trước) để số đo E11b được dùng trong bài
1. Kiểm ghim đạt ở đầu phiên (effective ≥ 0,5).
2. Không quá 3/11 điểm tham chiếu bị cờ (> 3%).
3. Sàn nhiễu toàn trình trong phiên ở 320 px (tv_nms) ≤ 2 × MDE của A/A 320 px (tức ≤ 1,10%).
Không đạt bất kỳ tiêu chí nào → E11b bị loại như E11 lần 1; bài giữ cột dự đoán. Đạt → ghi tên run vào `E11_ACCEPTED`
(`paper_vi/tables/make_tables.py`), thay cột dự đoán bằng cột đo (và giữ cột dự đoán cạnh đó để so), ghi DEVIATIONS.

## Checklist chuẩn bị máy (người dùng tự làm; harness không đổi cấu hình)
- [ ] Cắm sạc, pin ≥ 80%; chế độ nguồn Windows giữ nguyên như các lần đo trước (Balanced); ghi `powercfg /getactivescheme` vào PROGRESS.
- [ ] Tắt/thoát: trình duyệt, IDE (kể cả indexer), OneDrive/Google Drive sync, Docker, Ollama/LM Studio, Tailscale/WARP nếu không cần, game launcher, Sunshine.
- [ ] Tạm hoãn Windows Update và quét theo lịch của antivirus trong 3 giờ.
- [ ] Tắt sleep và tắt màn hình tự động trong 3 giờ; không dùng máy, không chạm bàn phím/chuột trong lúc đo.
- [ ] Đặt máy nơi thoáng, để nghỉ 5–10 phút trước khi chạy.
- [ ] Trợ lý (Claude Code) không chạy việc gì khác trong lúc đo (quy tắc 9); tốt nhất đóng cả ứng dụng Claude nếu tự chạy bằng terminal.

## Lệnh (chạy từ thư mục `sponge-sim`, sau khi tag)
```
python -X utf8 studies/e11_patch_remeasure/run.py --run-dir results/e11/E11b_P1cpu2_<ngày> --e7-dir results/e7_gpu/final/runs/e7
python -X utf8 studies/e11_patch_remeasure/analyze.py results/e11/E11b_P1cpu2_<ngày>
python -X utf8 studies/e11_patch_remeasure/analyze.py results/e11/E11b_P1cpu2_<ngày> --exclude-flagged
```
Thời gian: lần 1 mất 2 giờ 12 phút (11:35–13:47).
