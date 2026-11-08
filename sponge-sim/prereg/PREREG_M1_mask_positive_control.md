# PREREG_M1 — Đối chứng dương đầy đủ: độ trễ toàn trình theo K đối tượng giữ lại (model phân đoạn)

Soạn 04/11/2026. CHƯA tag. Tag khi quyết định chạy: `prereg-m1-<ngày>` (commit + tag TRƯỚC khi đo).
Nối tiếp M0 (cổng go/no-go, phán quyết GO: tag `prereg-m0-20261104`, run `results/m0/M0_P1cpu2_20261104`).

## Lý do
M0 cho thấy công đoạn dựng mask từng đối tượng là công đoạn ĐỦ LỚN (khác NMS 2–4%): tại K=300, mask chiếm ~8% toàn trình
với mask proto-res và ~78% với full-res, trên 1 lõi CPU. M0 đo total = forward + mask (bỏ decode/NMS), K tiêm vào riêng
công đoạn mask. M1 là đối chứng dương ĐẦY ĐỦ: tiêm K vào TOÀN pipeline thật (forward + decode + lọc điểm + NMS + dựng mask),
đo độ trễ toàn trình theo K, tìm ngưỡng K* và khớp mô hình ngân sách có số hạng mask. Đây là bản song song của E9 (đối chứng
dương cho NMS) nhưng cho công đoạn mask — và là kết quả ĐƯA VÀO BÀI (khác M0 chỉ là cổng).

## Giả thuyết
- H0: tỷ số độ trễ toàn trình patched/clean KHÔNG vượt 1 + MDE ở mọi K ≤ 300 (không có K* trong tầm max_det).
- H1: tồn tại K* ≤ 300 mà từ đó tỷ số toàn trình có cận dưới KTC > 1 + MDE; K* phụ thuộc model, retina_masks, thiết bị.
- H2 (mô hình ngân sách): R ≈ 1 + Δt_mask(K)/t_total dự đoán được tỷ số toàn trình; kiểm theo quy tắc định trước từng ô.

## Endpoint (đăng ký trước)
- **M1-P1 (chính):** K* = K nhỏ nhất (trong {10,50,100,300}) có cận dưới KTC95 của tỷ số toàn trình (patched K / baseline)
  > 1 + MDE_total của phiên, cho từng (model, retina). Đơn vị suy luận = ẢNH; CI bootstrap theo ảnh trên log tỷ số (20.000 lần).
- **M1-P2 (ngân sách):** khớp M_mask: ln t_total = a + b·ln(forward) + c·ln(max(K,1)); và kiểm R_pred = 1 + Δt_mask/t_total so R_meas.
  Quy tắc đạt/không đạt định trước: |ln(R_pred/R_meas)| ≤ MDE ở mỗi ô. Báo cả ô đạt và ô không đạt (không chỉ chọn ô đạt).
- **M1-P3 (liều–đáp ứng, mô tả):** đường tỷ số toàn trình theo K ∈ {0,10,50,100,300}, từng (model, retina).
- Phụ (mô tả): tỷ phần mask/total; phân rã công đoạn (forward, decode, tau, NMS, mask).

## Thiết kế
- **Model:** YOLOv8n-seg và YOLO11n-seg (trục model). YOLO26-seg (không NMS) là endpoint CHUYỂN GIAO riêng (PREREG_M1b),
  vì cơ chế ghép một-một của model không-NMS có thể làm K khó đẩy lên — đăng ký trước cả hai kết cục đều chấp nhận.
- **retina_masks:** False (proto-res, mặc định) và True (full-res).
- **K:** {0, 10, 50, 100, 300}. K bị chặn ở `max_det = 300` (Ultralytics) — ghi rõ đây là trần của bề mặt tấn công.
- **Độ phân giải:** 640×640. **Thiết bị:** P1_cpu2 (1 lõi P ghim). GPU/ARM là M2 (trục thiết bị, PREREG riêng, cần phần cứng).
- **Số ảnh:** 50 (chặt hơn M0 20 ảnh để CI hẹp; ước lượng thời gian sau smoke). **Lặp:** 5/ô, warmup 1. baseline K=0 ×3 copy.
- **Ngẫu hóa:** thứ tự ô (model×K×retina) hoán vị theo khối Latin; benchmark tham chiếu đầu/giữa/cuối, cờ > 3%.
- **A/A + MDE:** dùng A/A chuẩn của phiên (clean K=0 lặp) cho MDE_total; mọi tỷ số đọc so với 1 + MDE.

## Tiêm K vào pipeline thật (cơ chế — khác M0)
Chạy đủ công đoạn thật rồi THAY tập giữ lại bằng K tiêm, trước khi dựng mask (song song E9: forward thật, đầu ra bị thay):
`forward(ảnh)` → decode (box+score+class+32 hệ số mask) → lọc tau → NMS thật (đầu ra bỏ) → tiêm K detection tổng hợp
(hộp + 32 hệ số, tất định theo seed, dựng NGOÀI vùng đo) → `process_mask`/`process_mask_native`. total bấm trong MỘT lần gọi.
Khác M0: total GỒM decode + lọc + NMS thật (không còn là cận trên). Cần bổ sung decode seg (mang theo 32 hệ số mask) vào
`provenance`/helper của harness bằng COPY kèm header nguồn + test hồi quy (quy tắc 2); KHÔNG sửa code cũ.

## Mô hình ngân sách (mở rộng từ Amdahl của bài)
R = t_patched/t_clean ≈ 1 + Δt_mask/t_total, với Δt_mask = chi phí dựng K mask − chi phí ở baseline. So với bài cũ (số hạng
NMS), M1 thay bằng số hạng mask. Khớp post hoc trên chính M1 và (nếu có) kiểm tiến cứu trên tấn công thật M3.

## Quy tắc quyết định
- Có K* ≤ 300 cho ít nhất một (model, retina) → bề mặt tấn công TRONG TẦM (còn chờ M3 chứng minh tấn công đẩy K tới đó).
- Không có K* ≤ 300 ở ô nào → kết quả âm cho ô đó, báo kèm MDE (không nói "không có tác động" trống).
- Mô hình ngân sách: báo số ô đạt/không đạt và lý do hụt (như E9 phần B). Không dùng chữ "kiểm chứng" nếu < quá nửa ô đạt.

## Tiêu chí chất lượng phiên (định trước; chặt hơn M0 vì là bảng cho bài)
1. Ghim pass đầu phiên (effective ≥ 0,5). 2. ≤ 2/N điểm tham chiếu bị cờ > 3%. 3. Sàn nhiễu total trong phiên ≤ 2 × MDE A/A.
Không đạt → loại phiên, chạy lại khi máy rảnh thật (như E11b).

## Điều kiện tiên quyết (người dùng/harness)
- [ ] Trọng số: `yolov8n-seg.pt` (có), `yolo11n-seg.pt` (người dùng tải nếu dùng — quy tắc 7). Ghi SHA vào manifest.
- [ ] Viết `studies/m1_mask_positive/{run,analyze}.py` + decode seg (copy có nguồn) + test hồi quy. Smoke 2 ảnh (scratchpad) ước lượng thời gian.

## Checklist chuẩn bị máy (như M0/E11b)
- [ ] Cắm sạc; power plan Balanced (`381b4222-...`), ghi `powercfg /getactivescheme`.
- [ ] Tắt trình duyệt, IDE/indexer, sync cloud, Docker, antivirus quét theo lịch; hoãn Windows Update; tắt sleep.
- [ ] Trợ lý không chạy việc gì khác trong lúc đo (quy tắc 9).

## Lệnh (điền sau khi viết script; chạy từ `sponge-sim`, SAU khi tag)
```
python -X utf8 studies/m1_mask_positive/run.py --run-dir results/m1/M1_P1cpu2_<ngày>
python -X utf8 studies/m1_mask_positive/analyze.py --run-dir results/m1/M1_P1cpu2_<ngày>
```
Ước lượng: điền sau smoke (nhiều hơn M0 do decode/NMS thật + 50 ảnh + 2 model).
