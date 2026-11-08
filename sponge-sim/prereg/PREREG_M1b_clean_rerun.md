# PREREG_M1b — Chạy lại M1 sạch (chống throttle), CHỈ CHẠY KHI MÁY NGUỘI VÀ RẢNH

Soạn 04/11/2026. CHƯA tag. Tag khi quyết định chạy: `prereg-m1b-<ngày>` (commit + tag TRƯỚC khi đo).

## Lý do
M1 lần 1 (`results/m1/M1_P1cpu2_20261104`, tag `prereg-m1-20261104`) TRƯỢT cổng chất lượng: throttle nhiệt từ block 18
(~36 phút), 15/26 điểm tham chiếu bị cờ, max 42,5% (xem DEVIATIONS). Kết luận K* vẫn vững qua ba cách xem (toàn 50 ảnh;
36 ảnh trước throttle; 22 ảnh block không cờ — đều cho retinaFalse K*=300, retinaTrue K*=50, tỷ số gần trùng), nhưng theo
tiêu chí đã đăng ký, cần một phiên SẠCH để làm bảng chính. M1b đổi DUY NHẤT: thêm nghỉ hạ nhiệt giữa block và giảm số ảnh.

## Giống hệt M1 (không đổi cái gì ảnh hưởng số đo)
Mã `studies/m1_mask_positive/{run,analyze}.py` (cùng commit), cùng endpoint (M1-P1 K*, M1-P2 ngân sách, M1-P3 liều–đáp ứng),
cùng tiêm K, cùng NMS thật, cùng backend, cùng phân tích, cùng quy tắc quyết định. Nghỉ hạ nhiệt nằm NGOÀI vùng đo
(`time.sleep` giữa block, trước khi đọc reference) nên KHÔNG đổi giá trị đo; chỉ đưa lõi về trạng thái nguội.

## Đổi so với M1 (ghi rõ, có trong manifest qua cfg)
- `--cooldown-s 25`: nghỉ 25 giây giữa mỗi block (2 ảnh). Reference kế tiếp đọc trạng thái đã nguội.
- `--n-images 30`: giảm 50 → 30 ảnh (rút ngắn phiên; CI rộng hơn chút nhưng M1 cho thấy 22 ảnh đã đủ tách K*).
- Ước lượng: đo ~30/50 của 52 phút ≈ 31 phút + 15 block × 25 s ≈ 6 phút nghỉ ≈ ~37–45 phút tổng.

## Tiêu chí chất lượng (định trước) để M1b được dùng làm bảng chính
1. Ghim pass đầu và cuối phiên (effective ≥ 0,5).
2. **≤ 2 điểm tham chiếu bị cờ (>3%)** trên toàn phiên, và KHÔNG có chuỗi throttle (không có ≥2 block liên tiếp > 10%).
3. Sàn nhiễu total trong phiên ≤ 2 × MDE của A/A.
Không đạt → loại như M1; thử lại khi máy nguội hơn (tăng cooldown, hoặc chia hai phiên 15 ảnh). Đạt → M1b là bảng chính;
M1 + kiểm độ nhạy (`summary_excl.json`) làm bằng chứng phụ cho tính vững. Ghi mọi lần vào DEVIATIONS.

## Checklist chuẩn bị máy (người dùng; chặt hơn vì M1 đã throttle)
- [ ] **Để máy NGHỈ NGUỘI ≥ 10–15 phút trước khi chạy** (M1 vừa chạy 52 phút nóng).
- [ ] Cắm sạc; power plan Balanced (`381b4222-...`), ghi `powercfg /getactivescheme`.
- [ ] Đặt máy nơi thoáng, kê cao cho thoát nhiệt; nếu có, bật quạt/đế tản.
- [ ] Tắt trình duyệt, IDE/indexer, sync cloud, Docker, antivirus quét; hoãn Windows Update; tắt sleep/màn hình tắt.
- [ ] Trợ lý không chạy việc gì khác trong lúc đo (quy tắc 9).

## Lệnh (chạy từ `sponge-sim`, SAU khi tag; sau khi đo chạy cả hai phân tích)
```
python -X utf8 studies/m1_mask_positive/run.py --run-dir results/m1/M1b_P1cpu2_<ngày> --n-images 30 --cooldown-s 25
python -X utf8 studies/m1_mask_positive/analyze.py --run-dir results/m1/M1b_P1cpu2_<ngày>
python -X utf8 studies/m1_mask_positive/analyze.py --run-dir results/m1/M1b_P1cpu2_<ngày> --exclude-flagged
```
