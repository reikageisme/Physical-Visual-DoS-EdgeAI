# PREREG E9 — Tiêm ứng viên vào toàn pipeline (4 backend, nhiều ảnh) và bộ dự đoán N + K

Trạng thái: **ĐÃ DUYỆT 02/11/2026** (người dùng: "Duyệt PREREG_E9, tag và chạy full, máy rảnh."). Tag `prereg-e9-20261102` đặt TRƯỚC khi đo. Soạn 02/11/2026.
Tên E9 (không dùng E8 vì `attacks_provenance.py` đã nhắc "E8" cho một kế hoạch cũ).
Lý do: thầy nhận xét phát hiện "N tăng ~300 lần, toàn trình chỉ +2–6%" dựa trên dữ liệu lịch sử chưa ghim lõi và gần kết luận của EVADE
(`docs/REVIEW_THAY_NAM.md` mục 4); `REVIEW/04_q1_gap_analysis.md` §2 yêu cầu so "bộ dự đoán chỉ-từ-N" với "bộ dự đoán thêm backend/cap/hình học"
trên dữ liệu giữ ra. Hiện có: tiêm toàn pipeline chỉ 1 ảnh, 2 backend, A = 2100 (E1 phần B); dự đoán chỉ-từ-N (E6). Khoảng trống G5 trong
`docs/REVIEW_TOAN_BO_20261102.md`.

## Câu hỏi
- **Phần A**: Trong pipeline đầy đủ đã ghim lõi (forward thật + lọc τ + NMS), cần bao nhiêu ứng viên để độ trễ TOÀN TRÌNH tăng vượt sàn nhiễu A/A,
  cho từng backend × hình học × giới hạn top-k × độ phân giải? Câu trả lời là một ngưỡng N* (hoặc "vượt lưới"), không phải một bội số độ trễ.
- **Phần B**: Thêm một đại lượng hình học K (số hộp NMS giữ lại) vào bộ dự đoán thời gian NMS có giảm sai số dự đoán trên tập ứng viên THẬT so với
  chỉ dùng N không, và giảm bao nhiêu? Đây là phép phân rã (geometry giải thích bao nhiêu), KHÔNG phải một bộ dự đoán triển khai được, vì K chỉ biết
  sau khi chạy NMS. Bộ dự đoán dùng được trước NMS (ví dụ IoU trung bình giữa các hộp) là việc sau, không thuộc PREREG này.

## Phạm vi
Laptop i5-1335U, profile `P1_cpu2` (CPU logic 2), Windows Balanced (`381b4222-f694-41f0-9685-ff5bb260df2e`) giữ nguyên, YOLOv8n float32,
tensor chuẩn bị sẵn (resize trực tiếp). Không patch, không camera. Điện thoại: KHÔNG thuộc PREREG này (chưa qua kiểm chứng ghim).

## Phần A — Thiết kế đo
- Mã: `sponge-sim/studies/e9_pipeline/run.py`, `analyze.py` (đã viết 02/11; tag phủ cả mã). Phần B của `analyze.py` đã tự kiểm trên dữ liệu tổng hợp (`--selftest-b`, PASS) và CHƯA chạy trên dữ liệu E6/E6-A1 thật. Dùng lại `inject.make`, `backends`,
  `provenance.tau_filter`, `timing.timed`, `hw.verify_pin` như E1; không sửa mã cũ của E1/E6.
- Ảnh: 20 ảnh đầu của danh sách giữ ra cố định của E5 (`studies/e5_protocol/run.py:IMAGE_IDS[:20]`), cùng 20 ảnh cho cả hai độ phân giải.
  Ảnh chỉ quyết định bước forward; ứng viên được tiêm thay đầu ra đã giải mã (như E1 phần B).
- 320 px (A = 2100): N ∈ {0, 100, 300, 1000, 2100}. 640 px (A = 8400): N ∈ {0, 300, 1000, 3000, 8400}.
- Hình học: disjoint, dense_cluster, mixed (`inject.py`, như E1). N = 0 không có hình học; ô N = 0 (mẫu số của mọi tỷ số) được đo 3 lần ở 3 vị trí ngẫu nhiên trong mỗi ảnh, trung vị gộp cả 3. nc = 80.
  (Thêm sau smoke 2 ảnh ngày 02/11, trước khi tag: thời gian toàn trình dao động khoảng ±20% giữa các lần lặp, một ô mẫu số là không đủ.)
- Cap ∈ {None, 300}; ô cap = 300 chỉ khi N > 300.
- Backend: torch_pure, tv_nms, tv_batched, ultra_native (như E1).
- Mỗi ô (độ phân giải, ảnh, N, hình học, cap): 1 warm-up + 5 lần (320 px), 1 warm-up + 3 lần (640 px); trong ô, 4 backend xen kẽ theo khối Latin
  (seed 9103); thứ tự ô trong ảnh hoán vị (seed 9102); thứ tự ảnh cố định; 320 px đo trước 640 px.
- Mỗi lần ghi: total_ns (forward + τ + NMS trong MỘT lần gọi), và riêng nms_ns, tau_ns, nms_input, kept đo ở lần gọi kế tiếp ngoài vùng total
  (để có tỷ phần NMS; không cộng các trung vị công đoạn thành toàn trình).
- Seed tiêm: như E1 (`inject_seed_base + crc32("A|N|nc|geometry") mod 100000`), cùng tập ứng viên cho mọi ảnh; SHA256 ghi theo phép đo.
- Tham chiếu xung nhịp mỗi 2 ảnh và cuối phiên; cờ drift 3% ghi lại, giữ trong phân tích chính, kèm phân tích độ nhạy bỏ khối bị cờ.
- Verify ghim: cận trên + cận dưới effective_cores ≥ 0,5; FAIL thì không chạy. Kiểm verify lại ở cuối phiên và ghi kết quả (không dùng để loại dữ liệu).
- Ước lượng thời gian theo smoke 2 ảnh (02/11; 1,2 phút cho 320 px và 3,5 phút cho 640 px với 2 ảnh): 20 ảnh ≈ 12 phút (320 px) + 35 phút (640 px), cộng ô N = 0 lặp lại. Smoke lần 2 (sau khi thêm 3 ô N = 0; 1,3 + 5,5 phút cho 2 ảnh) cho ước lượng ≈ 70 phút cho 20 ảnh; dự trù 60–80 phút.
  Smoke lần 2: ghim PASS đầu 0,93, cuối 0,63 (vẫn qua cận dưới 0,5); điểm tham chiếu cuối lệch 37% (bị cờ). Với 2 ảnh, tỷ số ở 640 px dao động mạnh (0,83–1,39) nên không đọc gì từ smoke; đây là rủi ro nhiễu ở 640 px cần ghi khi báo cáo.
  Smoke: thư mục scratchpad `e9_smoke` (không phải kết quả); ghim PASS đầu và cuối (0,83 và 0,85), 1/3 điểm tham chiếu bị cờ.

## Phần A — Endpoint (đơn vị suy luận: ảnh; mỗi ô của mỗi ảnh tóm bằng trung vị các lần; CI = bootstrap theo ảnh, 20.000 lần, seed 20261021)
- **A-Chính 1 (ngưỡng N* ở 320 px)**: với mỗi backend, hình học disjoint, cap None: tỷ số toàn trình R(N) = total(N) / total(0) theo ảnh rồi trung bình
  hình học (log), CI95%. N* = N nhỏ nhất trong lưới có cận dưới CI > 1 + MDE_total(320 px). MDE_total(320 px) = 0,55%
  (`results/e5/AA_P1cpu2_20261101/summary.json:profiles.P1_cpu2.stages.total.mde_ratio_pct`). Không có N nào đạt thì báo "N* > 2100 ở độ nhạy này".
- **A-Chính 2 (ngưỡng N* ở 640 px)**: như trên với A = 8400 và MDE_total(640 px) = 1,9% (`results/e5/AA640_P1cpu2_20261102/summary.json`, cùng selector).
  Ghi kèm: A/A 640 px có 8/11 điểm tham chiếu bị cờ drift; MDE này là cận trên thận trọng của độ nhạy.
- **A-Phụ** (mô tả, không ngưỡng đạt/không): N* cho dense_cluster và mixed; tỷ số toàn trình cap 300 / cap None tại N lớn nhất; tỷ phần NMS trong toàn trình
  theo N; tỷ lệ lần lặp vượt mốc 100 ms (mốc do nhóm chọn, không phải yêu cầu ứng dụng) kèm khoảng Wilson; so N* với N của ứng viên thật đo ở E6
  (trung vị khoảng 90 ở 320 px).
- Quy tắc đọc: cận dưới CI > 1 + MDE thì ghi "phát hiện tăng toàn trình ở độ nhạy này"; ngược lại "không phát hiện, MDE = x%". Không dùng "không ảnh hưởng",
  "tương đương".

## Phần B — Bộ dự đoán N + K (dùng dữ liệu đã có và dữ liệu đang đo)
- Dữ liệu huấn luyện (chỉ dùng để ước lượng hệ số): E1 micro `results/e1/E1_P1cpu2_20261101/raw_P1_cpu2.jsonl`, nc = 80, cap None, mọi A và hình học;
  mỗi ô tóm bằng trung vị nms_ns; K = kept (cố định trong ô vì tập ứng viên tất định), N = nms_input.
- Mô hình, khớp riêng từng backend bằng bình phương tối thiểu trên thang log, các ô N > 0:
  - M_N: ln t = a + b·ln N
  - M_NK: ln t = a + b·ln N + c·ln max(K, 1)
  Không thêm số hạng, không chọn mô hình theo dữ liệu kiểm tra. Đặc tả này được cố định ở đây trước khi khớp M_NK lần nào.
- Dữ liệu kiểm tra:
  - **Kiểm tra 1 (đã thấy một phần, gắn nhãn post hoc)**: E6 320 px `results/e6/E6_P1cpu2_20261102/raw_P1_cpu2.jsonl`, τ = 0,01, mọi nhánh 320 px kể cả clean;
    mỗi (ảnh, nhánh, seed, backend) tóm bằng trung vị nms_ns, N = nms_input, K = kept. Sai số của dự đoán CHỈ-TỪ-N (nội suy đường cong) đã được xem;
    M_N và M_NK chưa được khớp hay đánh giá trên tập này.
  - **Kiểm tra 2 (xác nhận, dữ liệu chưa phân tích)**: E6-A1 640 px `results/e6/E6A1_P1cpu2_20261102/` (clean640, C2, C4), cùng cách tóm.
    Phân tích của PREREG_E6_addendum_clean640 (tỷ số patch/clean, sai số N-only A1-3) có thể chạy trước; M_N/M_NK không được khớp lại trên E6-A1.
- **B-Chính (Kiểm tra 2)**: với mỗi backend, Δ = trung vị |ln(đo/dự đoán)| của M_N trừ của M_NK trên các đơn vị của E6-A1; CI95% bootstrap hai tầng
  (ảnh × seed; clean640 dùng chung cho mọi cột seed như E6), 20.000 lần, seed 20261021.
  Quy tắc đọc: cận dưới CI của Δ > 0 thì ghi "thêm K làm giảm sai số dự đoán ở backend này"; CI chứa 0 thì "không phát hiện cải thiện".
  Không báo "N đủ để dự đoán" từ CI chứa 0.
- **B-Phụ**: cùng đại lượng trên Kiểm tra 1 (nhãn post hoc); R² trên thang log; phân vị 90 của sai số; hệ số b, c kèm CI bootstrap theo ô của E1;
  kiểm tra chéo trong E1 theo hình học (khớp trên disjoint + dense_cluster, đánh giá trên mixed), chỉ mô tả.

## Điều kiện chạy (Phần A)
Máy rảnh thật (không dùng tay, cắm sạc, Balanced, đóng trình duyệt/IDE indexer/OneDrive), sau khi E6-A1 xong; không có tác vụ nào khác của trợ lý
chạy song song (bài học từ drift của A/A 640 px). Người dùng xác nhận trước khi chạy đầy đủ.

## Báo cáo
- Bài: một bảng N* theo backend × hình học × độ phân giải (thay Bảng pipeline 1 ảnh hiện tại, giữ bảng cũ ở phụ lục), một đoạn so N* với N thật,
  và một bảng sai số M_N / M_NK theo backend. Giữ mọi kết quả âm.
- Mọi số truy được tới `results/e9/<run>/summary.json` và các tệp nguồn ở trên. Sai lệch ghi DEVIATIONS.md.

## Rủi ro đã biết
- MDE_total đo trong A/A với ranh giới toàn trình có bước tải ảnh; ranh giới E9 là tensor đã chuẩn bị (forward + τ + NMS). Nếu cần MDE đúng ranh giới,
  phải chạy A/A riêng (chưa đăng ký ở đây); khi đó ghi DEVIATIONS và không đổi quy tắc đọc.
- K là đầu ra của NMS (vòng tròn nếu dùng để dự đoán triển khai); Phần B chỉ là phân rã.
- Các phát hiện review mã chưa sửa (PROGRESS.md 02/11) không ảnh hưởng đường mã E9, trừ metadata `ultra_impl` (ghi đúng phiên bản Ultralytics thay vì dựa vào trường này).
