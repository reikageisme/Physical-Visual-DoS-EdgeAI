# PREREG_M2 — Trục thiết bị: ngưỡng K* và tỷ số toàn trình theo thiết bị (CPU / GPU / ARM)

Soạn 04/11/2026. CHƯA tag. Tag khi quyết định chạy từng thiết bị: `prereg-m2-<thiết bị>-<ngày>` (commit + tag TRƯỚC khi đo).

## Lý do
M1 (trên 1 lõi CPU) cho K* và tỷ số toàn trình theo K cho công đoạn dựng mask. Nhưng chi phí dựng mask **song song hoá tốt trên GPU**:
mô hình ngân sách dự đoán tấn công hiệu quả khi lan truyền thuận nhanh NHƯNG mask vẫn tuần tự/đắt. M2 kiểm giả định đó bằng cách
đo CÙNG giao thức M1 trên nhiều thiết bị. Đây là trục "điều kiện" (hướng C của đề tài): khi nào lỗ hổng mask thành vấn đề sẵn sàng.

## Giả thuyết
- H1: K* và tỷ số toàn trình phụ thuộc thiết bị. Trên CPU (laptop, ARM) mask chiếm tỷ phần lớn → K* thấp. Trên GPU mask song song hoá →
  tỷ phần nhỏ hơn, K* cao hơn hoặc không có trong tầm max_det.
- Đăng ký trước CẢ HAI kết cục đều chấp nhận: GPU vẫn chậm (mask là nút cổ chai cả trên GPU) HOẶC GPU miễn nhiễm (chỉ CPU/ARM mới lỗ hổng).

## Endpoint
Giống M1 (M1-P1 K*, M1-P2 ngân sách, M1-P3 liều–đáp ứng), tính RIÊNG cho từng thiết bị (mỗi thiết bị một run-dir, một tag).
So sánh K* và tỷ số K=300 giữa thiết bị (mô tả, không gộp).

## Thiết bị và cách chạy (DÙNG LẠI mã M1, chỉ đổi cờ)
Mã `studies/m1_mask_positive/{run,analyze}.py` đã hỗ trợ `--device`. KHÔNG sửa logic đo.
1. **CPU laptop** (đã có M1/M1b, cần lại bản đạt cổng): `--device cpu` trên môi trường ổn định (xem PREREG_M1b; bỏ `--cooldown-s`).
2. **GPU (Linux thuê)**: `--device cuda`. Đo có `torch.cuda.synchronize` quanh vùng đo (đã cài trong `timing.timed(sync=...)`).
   Chuẩn bị: cài torch+cuda, ultralytics; đặt `SPONGE_REPO_ROOT`; chép dataset `outputs/datasets/coco_val2017_extended` + `yolov8n-seg.pt`.
   Ghim lõi host không quyết định trên GPU (chỉ ghi verify, không dừng). Máy rảnh, không chia sẻ GPU.
3. **ARM điện thoại Galaxy A36** (`--device cpu`): theo giao thức điện thoại trong CLAUDE.md và PREREG_E5_phone (tắt mạng di động,
   xác nhận máy rảnh, log ra `~/sponge-sim`, kết quả về `results/phone_a36/`). hw_linux lo ghim lõi. Chạy nền rồi lấy kết quả.

## Tiêu chí chất lượng (mỗi thiết bị, định trước)
- CPU/ARM: như M1 (≤2 cờ drift, sàn nhiễu ≤2×MDE, ghim pass).
- GPU: ≤2 cờ drift (reference CPU host vẫn theo dõi ổn định host); sàn nhiễu A/A GPU ≤ 2×MDE; ghi rõ model GPU, driver, CUDA, tần số.
Không đạt → loại phiên, chạy lại (ghi DEVIATIONS).

## Giới hạn đã biết (ghi vào bài)
- Con số độ trễ phụ thuộc model GPU/ARM cụ thể; GPU thuê phải là loại ĐẠI DIỆN triển khai, không phải GPU khủng (GPU càng mạnh,
  mask càng song song → hiệu ứng càng teo → thiên về "không có tác động"). Ghi rõ model thiết bị cho mọi bảng.
- K bị chặn ở max_det=300 ở mọi thiết bị.

## Lệnh mẫu (SAU khi tag, máy rảnh; trên từng thiết bị)
```
# GPU (Linux)
SPONGE_REPO_ROOT=/path/to/repo python -X utf8 studies/m1_mask_positive/run.py --run-dir results/m2/M2gpu_<ngày> --device cuda --n-images 50
python -X utf8 studies/m1_mask_positive/analyze.py --run-dir results/m2/M2gpu_<ngày>
# ARM (trên điện thoại, nền)
python -X utf8 studies/m1_mask_positive/run.py --run-dir results/phone_a36/M2arm_<ngày> --device cpu --n-images 30
```
