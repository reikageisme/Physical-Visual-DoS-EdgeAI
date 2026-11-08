# PREREG E7: sinh patch SG-GA với tập tối ưu 100 ảnh (320/640 px, ngưỡng 0,25) trên GPU thuê

Trạng thái: **NHÁP, chờ duyệt** (chưa tag). Tag dự kiến: `prereg-e7-<ngày>`.
Mã: `sponge-sim/studies/e7_sggar/run.py` (một lệnh), `make_bundle.py` (gói độc lập), `src/sponge_sim/attacks_provenance.py` (bản copy có header nguồn).
Yêu cầu thầy: "tăng tập tối ưu lên tối thiểu 100 ảnh, thêm thử nghiệm ngưỡng 0,25 và ảnh 640 px" (`docs/REVIEW_THAY_NAM.md`, yêu cầu 3).

## Mục đích và giới hạn việc làm ở máy GPU
Tạo patch từ thuật toán SG-GA **đã có** (không phải thuật toán mới) ở quy mô lớn hơn, để làm case study cho câu hỏi trung tâm
"số ứng viên N có dự đoán được độ trễ không". Máy GPU **chỉ sinh patch** (lưu tensor + SHA256); **không đo độ trễ** ở đó.
Độ trễ đo trên laptop và điện thoại theo giao thức ghim core (PREREG_E6). Không có kết luận thống kê nào rút ra ở máy GPU.

## Cái gì giữ nguyên, cái gì đổi (so với lượt 4 ảnh trước)
| Mục | Lượt cũ | E7 |
|---|---|---|
| Thuật toán | `extended_validation.optimize` (SG-GA: quần thể 20, 5 elite, giao chéo ngang, đột biến, EOT đầy đủ, vị trí saliency Laplacian) | **Giống hệt** (bản copy `attacks_provenance.optimize`) |
| Ảnh tối ưu | 4 ảnh | **100 ảnh** từ `coco128/train2017`, chọn `random.Random(20261101).sample(sorted(128 tên file), 100)`, danh sách cố định trong `run.py:TRAIN_FILES`, SHA256 từng ảnh trong `manifest.json` |
| Ngân sách | 2.500 lần forward ảnh = 625 ứng viên × 4 ảnh | **62.500 lần forward = 625 ứng viên × 100 ảnh** (cùng số ứng viên, cùng số thế hệ ≈ 31); checkpoint ở 10.000 / 25.000 / 62.500 |
| Độ phân giải | 320 px | 320 px **và 640 px** (resize trực tiếp, không letterbox) |
| Ngưỡng mục tiêu | 0,01 | 0,01 **và 0,25** (tham số `threshold` đã có sẵn trong `objective`, chỉ chưa được truyền) |
| Thiết bị | CPU | GPU, float32 (ép bằng gán `victim.use_half=False; victim.model=victim.model_float`, không sửa file gốc) |

Ba điều chỉnh kỹ thuật không đổi công thức, đã khai ở đầu `attacks_provenance.py`: canvas là tham số (bản gốc hardcode 320 trong `find_locations`),
tham số ngưỡng được truyền, tham số thiết bị. Kiểm chứng: 15 test tổng quát hóa + 2 test chậm khớp từng bit patch đã lưu (seed 0, 400 queries, CPU, 2 luồng).
**Hệ quả đã khai trước:** RNG CUDA khác RNG CPU nên patch GPU không trùng byte với lượt CPU dù cùng thuật toán và seed; so sánh với patch cũ chỉ ở mức thống kê.

## Điều kiện (seed 0–9 cho mỗi điều kiện; vòng lặp seed-major nên dừng giữa chừng vẫn cân bằng)
| Mã | Canvas | Patch | Ngưỡng tối ưu | Phương pháp | Vai trò |
|---|---|---|---|---|---|
| C1 | 320 | 64 px (4,0% diện tích) | 0,01 | SG-GA | chính |
| C2 | 640 | 128 px (4,0%, bằng diện tích C1) | 0,01 | SG-GA | chính (độ phân giải) |
| C3 | 320 | 64 px | **0,25** | SG-GA | chính (ngưỡng) |
| B1 | 320 | 64 px | 0,01 | GA vị trí tâm | baseline có sẵn |
| B2 | 320 | 64 px | 0,01 | GA vị trí ngẫu nhiên | baseline có sẵn |
| B3 | 320 | 64 px | 0,01 | tìm kiếm ngẫu nhiên (vị trí saliency) | baseline có sẵn |
| B4 | 320 | 64 px | 0,01 | gradient tanh (white-box, Overload-inspired) | baseline có sẵn; thầy nêu SG-GA thua baseline gradient |
| C4 | 640 | 64 px (1,0%) | 0,01 | SG-GA | phụ (kích thước tuyệt đối) |
Ghi chú: C2 giữ diện tích bằng C1 nên so sánh "độ phân giải" không lẫn với "diện tích"; C4 tách riêng hiệu ứng diện tích.

## Dữ liệu và tách tập
Tập tối ưu (100 ảnh `train2017`) tách khỏi tập đánh giá (100 ảnh `val2017` của E5, đã cố định từ 21/10/2026): khác tập gốc COCO, không giao nhau.
Patch bị khóa sau khi SHA256 được ghi (`patches_index.json`); mọi phép đo sau này dùng đúng các tensor đó, không tối ưu lại.

## Đầu ra và kiểm chứng
`runs/e7/manifest.json` (git, SHA256 mã + trọng số + từng ảnh, phiên bản torch, tên GPU, cấu hình), `runs/e7/<điều kiện>/seed<k>/{patch_q*.npy, patch_q*.png, checkpoint_q*.json, resume.pt, complete.json}`,
`runs/e7/patches_index.json`. Checkpoint JSON gồm `patch_sha256`, vị trí chèn, thời gian, `best_training_fitness`.
Sau khi mang về: laptop kiểm lại SHA256 từng tensor và so với `patches_index.json` trước khi dùng (thiếu/lệch thì dừng).

## Ước lượng giờ GPU (CHƯA đo; script tự in ETA sau đơn vị đầu tiên)
Tổng: 8 điều kiện × 10 seed × 62.500 lần forward = 5 triệu lần forward ảnh + nhiễu EOT. Giả định thô: GPU 16 GB cho YOLOv8n float32 hàng trăm đến vài nghìn ảnh/giây;
EOT kornia trên GPU chi phối ở 640 px. Ước lượng **1–2,5 giờ cho cả 8 điều kiện**; nhóm chính C1–C3 (30 đơn vị) khoảng 35–60% thời gian đó; C4 và B4 là phần tốn nhất ở 640/gradient.
Thứ tự chạy là seed-major (seed 0 cho mọi điều kiện, rồi seed 1, ...). **Sau khi seed 4 hoàn tất, mọi patch mà E6 cần (seed 0-4 của các nhánh) đã có**; seed 5-9 là phần bổ sung. Nếu thiếu giờ: dừng sau seed 4. Nếu phải cắt điều kiện, cắt theo thứ tự định trước C4, B3, B2, B1, B4 (giữ C1, C2, C3). Không đổi thứ tự sau khi bắt đầu.

## Rủi ro đã thấy trước
- Hết bộ nhớ GPU với 100 ảnh 640 px một lô: nếu lỗi OOM thì dừng, báo lại (không tự chia nhỏ lô vì sẽ đổi thứ tự RNG so với thiết kế; nếu phải chia thì ghi DEVIATIONS và chạy lại từ đầu điều kiện đó).
- Phiên bản torch/CUDA trên máy thuê khác laptop: ghi trong manifest; không ảnh hưởng tính hợp lệ của patch (patch là dữ liệu cố định), có ảnh hưởng nếu có ai so sánh quỹ đạo tối ưu giữa máy.
- Máy thuê cần mạng để cài thư viện; việc đó nằm ngoài laptop và điện thoại, không vi phạm quy tắc mạng của dự án (chỉ khai trong báo cáo).
- Đây vẫn là thuật toán hộp xám dùng điểm số trước NMS; không phải tấn công mạng thật. Bài không phát hành patch tối ưu (xem Responsible release trong README).

## Điều kiện dừng và báo cáo
Dừng khi hết giờ thuê hoặc xong. Báo: số đơn vị (điều kiện × seed) hoàn tất, ETA thực tế, lỗi nếu có. Mang cả thư mục `runs/e7/` về.
