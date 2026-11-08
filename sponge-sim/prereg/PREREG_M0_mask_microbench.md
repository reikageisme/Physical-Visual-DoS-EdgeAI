# PREREG_M0 — Micro-benchmark chi phí dựng mask theo K (cổng go/no-go cho hướng phân đoạn)

Soạn 04/11/2026. CHƯA tag. Tag khi quyết định chạy: `prereg-m0-<ngày>` (commit + tag TRƯỚC khi đo).
Chỉ cần CPU laptop — KHÔNG phụ thuộc GPU/thiết bị biên, nên chạy được ngay trong lúc chờ thu xếp phần cứng.

## Lý do
Bài hiện tại (Q3) cho kết quả âm trên 1 lõi CPU vì NMS chỉ chiếm ~2–4% toàn trình (trung vị P1 0,0207,
`results/e5/20261021T151412/summary.json`), nên đẩy số ứng viên lên 340–389× chỉ làm toàn trình tăng 2,9–3,4%.
Mô hình ngân sách Amdahl của chính bài chỉ ra điều kiện cần: một tấn công chỉ làm chậm toàn trình khi công đoạn nó
đánh chiếm một tỷ phần đủ lớn. Câu hỏi của hướng mới: trong model **phân đoạn**, công đoạn dựng mask từng đối tượng
(O(K), nặng theo độ phân giải khi bật `retina_masks`) có phải công đoạn đủ lớn đó không.

**M0 là cổng go/no-go, KHÔNG phải kết quả cho bài.** Nếu ở K=300 chi phí dựng mask vẫn nhỏ hơn sàn nhiễu đo được
thì bỏ hướng YOLO-seg trên CPU (chuyển sang trục GPU hoặc hướng dự phòng SAM). Không đo đối chứng dương đầy đủ,
không sinh tấn công, không kết luận khoa học từ M0.

## Giả thuyết
- H0 (null): tỷ phần thời gian dựng mask trên toàn trình KHÔNG tăng đáng kể theo K; tại K=300 nó ≤ sàn nhiễu.
- H1: tỷ phần đó tăng theo K, và tại K=300 (nhất là khi `retina_masks=True`) vượt sàn nhiễu đủ để một tấn công
  đẩy K từ nền sạch lên 300 có thể làm toàn trình chậm phát hiện được.

## Endpoint
- **Chính:** `frac_mask@K = t_mask_stage / t_total`, đo tại mỗi K, cho từng cấu hình `retina_masks` ∈ {False, True}.
  Đại lượng quyết định: `Δfrac = frac_mask@300 − frac_mask@(K nền sạch)`.
- **Phụ (mô tả):** t_mask_stage tuyệt đối (ms) theo K; độ dốc t_mask theo K; t_total theo K.
- Tách công đoạn, KHÔNG cộng hai trung vị rời: tiền xử lý, forward, giải mã, lọc tau, NMS, **dựng mask**, tổng.

## Thiết kế
- **Model:** YOLOv8n-seg (chính). Thêm YOLO11n-seg nếu có trọng số. YOLO26-seg (không NMS) và SAM/MobileSAM để
  giai đoạn sau, KHÔNG thuộc M0.
- **Độ phân giải đầu vào:** 640×640 (khớp phần 640 px của bài). Ghi rõ nếu pipeline dùng letterbox.
- **K (số đối tượng giữ lại):** {0, 10, 50, 100, 300}. Điều khiển K bằng **tiêm K detection tổng hợp** (hộp + hệ số mask)
  tại ranh giới giải mã → dựng mask, cùng kiểu đối chứng dương như E1 tiêm N_τ. Dựng dữ liệu tiêm NGOÀI vùng bấm giờ
  (giống `backends.prepare()`), chỉ bấm giờ đúng công đoạn dựng mask + toàn trình. Hộp tiêm phủ các diện tích khác nhau
  (ghi phân bố diện tích), vì chi phí cắt/phóng mask phụ thuộc diện tích hộp.
- **retina_masks:** chạy cả False (mask ở độ phân giải proto ~160×160) và True (phóng lên full-res 640). Đây là đòn bẩy chính.
- **Lặp:** 30 lần/điều kiện (như E1/E5). Thứ tự điều kiện ngẫu hóa theo khối (Latin square), không chạy hết K này rồi mới K khác.
- **Ghim lõi:** profile P1_cpu2 (logical 2), verify_pin bằng tải torch ở đầu phiên (effective ≥ 0,5).
- **Sàn nhiễu:** chèn khối A/A (cùng K, cùng điều kiện, đo hai lần) để lấy MDE trong cùng phiên; benchmark tham chiếu
  đầu/giữa/cuối, cờ nếu trôi > 3%.

## Quy tắc quyết định go/no-go (định trước, tránh quyết sau khi thấy số)
Xét ở cấu hình bất lợi nhất cho H1 đã đo (CPU P1_cpu2), với `retina_masks=True`:
- **GO** nếu `Δfrac` (từ K nền sạch lên K=300) **≥ 2 × MDE** của A/A trong phiên **VÀ ≥ 5% tuyệt đối** của toàn trình.
  (MDE = sàn thống kê; 5% = ngưỡng ý nghĩa thực tiễn đặt trước của nhóm.)
- **NO-GO cho YOLO-seg CPU** nếu `Δfrac < 2 × MDE` hoặc `< 5%`. Khi đó: không theo YOLO-seg trên CPU; cân nhắc
  chỉ-GPU, hoặc chuyển sang hướng dự phòng SAM. Ghi kết quả âm nguyên vẹn, kèm MDE (không nói "không có tác động" trống).
- **BIÊN** (đạt MDE nhưng < 5%, hoặc chỉ đạt khi retina_masks=True): ghi là "có tín hiệu, phụ thuộc cấu hình",
  đo thêm một model seg thứ hai trước khi quyết.
Cả ba nhánh retina_masks=False được báo cạnh bên để biết đòn bẩy retina đóng góp bao nhiêu.

## Tiêu chí chất lượng phiên (như E11b)
1. Verify ghim đạt đầu phiên (effective ≥ 0,5). 2. ≤ 3 điểm tham chiếu bị cờ drift > 3%.
3. Sàn nhiễu toàn trình trong phiên ≤ 2 × MDE của A/A chuẩn. Không đạt → phiên bị loại, chạy lại khi máy rảnh thật.

## Điều kiện tiên quyết (người dùng làm; harness không dùng mạng — quy tắc 7)
- [ ] Tải trọng số `yolov8n-seg.pt` (và `yolo11n-seg.pt` nếu muốn) về repo; ghi SHA-256 vào manifest. Đây là thao tác
      mạng một lần, người dùng tự chạy, KHÔNG để harness tải.
- [ ] Viết `studies/m0_mask_microbench/run.py` + `analyze.py` + test hồi quy (tiêm K tất định; hàm đo chỉ tham chiếu
      vùng dựng mask; ảnh/dữ liệu tiêm dựng trước). Chạy smoke 2–3 ảnh (scratchpad, không báo cáo) để ước lượng thời gian.

## Checklist chuẩn bị máy (người dùng tự làm; như E11b)
- [ ] Cắm sạc, pin ≥ 80%; chế độ nguồn giữ Balanced (`381b4222-...`); ghi `powercfg /getactivescheme` vào PROGRESS.
- [ ] Tắt trình duyệt, IDE (kể cả indexer), sync cloud, Docker, Ollama/LM Studio; hoãn Windows Update + quét antivirus.
- [ ] Tắt sleep/tắt màn hình tự động; không chạm máy trong lúc đo.
- [ ] Trợ lý KHÔNG chạy việc gì khác trong lúc đo (quy tắc 9).

## Lệnh (điền sau khi viết script; chạy từ `sponge-sim`, SAU khi tag)
```
python -X utf8 studies/m0_mask_microbench/run.py --run-dir results/m0/M0_P1cpu2_<ngày>
python -X utf8 studies/m0_mask_microbench/analyze.py results/m0/M0_P1cpu2_<ngày>
```
Ước lượng thời gian: điền sau smoke. Dự kiến một buổi (CPU, 5 K × 2 retina × 30 lặp × 2 model).
