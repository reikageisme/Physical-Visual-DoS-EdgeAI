# PREREG E5 — Giao thức đo độ trễ trên CPU laptop ghim core

Ngày: 21/10/2026. Nhánh `harness/step1`. Tag sau khi commit file này: `prereg-e5-20261021`.
Script: `sponge-sim/studies/e5_protocol/run.py` (đo), `analyze.py` (phân tích). Cấu hình đầy đủ: `CONFIG` trong `run.py`
(hash ghi ở `manifest.json:config_sha256` của mỗi run).

## Mục đích
Thiết lập và định lượng sai số đo của giao thức sẽ dùng cho E1/E2b: ghim affinity thật, đo từng công đoạn,
A/A (cùng điều kiện, hai nhãn) để biết nhiễu đo và MDE cho tỷ số độ trễ, cờ trôi xung nhịp.
Không phải thí nghiệm tấn công. Không có patch.

## Phạm vi và từ ngữ
"Mô phỏng hạn chế tài nguyên trên CPU laptop ghim core" (i5-1335U, Windows 11). Tensor chuẩn bị bằng `load_image`
(resize trực tiếp 320×320, không letterbox). Không có camera, không ARM/edge thật. Một laptop, một model (YOLOv8n).

## Đã biết trước khi đo (chỉ smoke kỹ thuật)
Smoke 4 ảnh × 2 lặp × 3 profile chạy ở scratchpad để kiểm script; KHÔNG phân tích để báo cáo, không dùng chọn tham số
ngoài việc xác nhận script chạy và ước lượng thời gian (~25 phút cho đầy đủ). Smoke cho thấy tham chiếu cuối phiên E1
lệch +14,4% (cơ chế cờ trôi hoạt động).

## Thiết kế
- Profile CPU (đọc topology từ OS bằng `GetLogicalProcessorInformationEx`; mỗi core vật lý lấy 1 logical, tránh HT anh em):
  `P1` = logical {0} (P-core 0); `E1` = {4}; `E4` = {4,5,6,7}. intra-op threads = |allowed|, interop = 1.
  Mỗi profile chạy trong subprocess riêng, ghim TRƯỚC khi torch tạo thread pool; xác minh bằng tải torch với 2×|allowed| thread:
  pass khi CPU-time/wall ≤ |allowed| + 0,15 và mean cpu_percent trên CPU được phép ≥ 70%.
  Thứ tự profile: hoán vị seed 5021.
- Ảnh: 100 ID lấy bằng `random.Random(20261021).sample(sorted(500 ID X1-500), 100)`; danh sách đầy đủ ở `run.py:IMAGE_IDS`.
- Mỗi ảnh: 1 warm-up (bỏ), rồi 20 lần đo: nhãn A/B theo chuỗi ABBA hoặc BAAB lặp (chọn ngẫu nhiên theo ảnh, seed 5022+cpu0).
  A và B là CÙNG điều kiện (ảnh sạch, τ = 0,01, không cap, IoU 0,45).
- Mỗi lần đo ghi riêng các công đoạn (ns, `perf_counter_ns`, gc tắt): `pre` (load_image), `forward`, `decode`, `tau`,
  `nms_torch_pure`, `nms_tv_nms`, và `total` = một lần gọi load→forward→decode→tau→torch_pure NMS đo trực tiếp
  (không cộng công đoạn, không cộng trung vị).
- Khối 10 ảnh; benchmark tham chiếu (matmul 256×256 ×20, trung vị 15 lần) đầu phiên, đầu mỗi khối và cuối phiên.
  |Δ| > 3% so với đầu phiên → cờ.

## Endpoint
- **Chính (theo profile, cho `forward` và `total`)**: d_i = ln(median_A,i / median_B,i), i = ảnh (n = 100).
  Báo mean d, exp(mean d), CI95% bootstrap percentile theo ảnh (20.000 lần, seed 20261021), và MDE
  = (t₀.₉₇₅,₉₉ + t₀.₈,₉₉)·SD(d)/√100 đổi sang % (xấp xỉ t trung tâm).
  Kỳ vọng giao thức: CI chứa 1. Nếu CI không chứa 1 → giao thức có thiên lệch theo vị trí/thời gian; báo nguyên trạng,
  không sửa bằng cách bỏ điểm đo.
  **Không** diễn giải "CI chứa 1" thành "A = B" hay "tương đương"; chỉ dùng để định cỡ nhiễu đo (MDE).
- **Phụ**: cùng thống kê cho `pre`, `decode`, `tau`, `nms_torch_pure`, `nms_tv_nms`; CV trong-ảnh (trung vị theo ảnh);
  số điểm tham chiếu bị cờ; kết quả verify ghim; tỷ phần NMS torch_pure / total (mô tả).
- **Độ nhạy**: lặp endpoint chính sau khi bỏ các khối có cờ trôi (nếu có). Báo cả hai, không thay kết quả chính.
- Holm: không áp dụng (endpoint chính là định cỡ nhiễu, không kiểm định giả thuyết hiệu ứng).

## Tiêu chí loại
Chỉ loại ảnh nếu đọc file lỗi (ghi lại). Không loại outlier theo giá trị. Profile có verify ghim FAIL: vẫn giữ raw,
gắn nhãn "ghim không xác minh", không dùng làm profile cho E1/E2b.

## Quyết định chuyển tiếp (định trước)
E1/E2b dùng profile có verify pass; chọn P1 và E4 nếu cả hai pass. MDE của E5 được ghi vào PREREG_E1/E2b như mức
nhiễu đo; hiệu ứng nhỏ hơn MDE không được diễn giải là "không có".

## Đầu ra
`sponge-sim/results/e5/<run_id>/`: `manifest.json`, `raw_{P1,E1,E4}.jsonl`, `summary.json`, `run.log`.
Mọi số báo cáo trích `summary.json:<selector>` hoặc raw JSONL.
