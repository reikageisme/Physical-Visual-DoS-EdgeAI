# Sponge Edge-AI Attack (NCKH Project)

Dự án Nghiên cứu Khoa học (NCKH): Đánh giá lỗ hổng bảo mật của mô hình Edge-AI (SSDLite MobileNetV3) đối với hình thức tấn công năng lượng bằng mảng dữ liệu đặc biệt (Sponge Data).

## Tổng quan (Overview)
Các thiết bị Edge-AI (như Raspberry Pi, Jetson Nano, hoặc Camera AI) thường bị giới hạn về xử lý (CPU/RAM/Tản nhiệt). Thuật toán này sử dụng **Giải thuật Di truyền (Genetic Algorithm)** để sinh ra các miếng dán vật lý (Sponge Patches). Khi Camera ghi nhận miếng dán này, mô hình phát hiện vật thể (SSD) thay vì lọc bỏ các Hộp giới hạn (Bounding Boxes) sẽ bị ép phải sinh ra hàng ngàn Hộp. Sự quá tải phép tính này dẫn đến:
1. **CPU đạt 100%** và nghẽn cổ chai.
2. **Nhiệt độ chip tăng**, dẫn đến Thermal Throttling.
3. **Tụt khung hình (FPS Drop)**, làm sập hệ thống xử lý thời gian thực.

## Cài đặt mô trường
```bash
pip install -r requirements.txt
```

## Các Bước Chạy Thực Nghiệm

### 1. Khởi chạy GA tạo Sponge Patch
Bạn có thể cấu hình số thế hệ (Generations) và quần thể (Population) qua các `arguments` trên Terminal:
```bash
python main_train.py --pop 50 --gen 100 --size 64
```
*Kết quả sẽ được xuất ra thư mục `outputs/` dưới dạng ảnh `sponge_patch_g100_p50.png`*

### 2. Thực nghiệm Physical DoS Test
Xài webcam cá nhân hoặc laptop camera để giả lập chạy xử lý AI thời gian thực (được giới hạn khung hình nhân tạo ~10 FPS để phản ánh Edge Device).

#### A. Chế độ Bình thường (Chạy kiểm tra gốc)
```bash
python test_physical_dos.py --cam 0
```

#### B. Chế độ Áp Patch số (Digital Overlay Attack)
Không cần in ra giấy, bạn áp thẳng Patch vừa Train lên góc camera để xem hệ thống gục ngã ra sao:
```bash
python test_physical_dos.py --cam 0 --patch outputs/sponge_patch_g100_p50.png
```

### 3. Trực quan hóa báo cáo NCKH (Đồ thị)
Trong quá trình test DoS, thư mục `logs/` sẽ tự ghi nhận file CSV theo thời gian thực (FPS, CPU load, RAM, Temp). Bạn chạy tệp sau để vẽ thành biểu đồ đưa vào Bài báo cáo NCKH:
```bash
python utils/plot_results.py
```
*Script sẽ tự tìm file CSV mới nhất trong thư mục log và kết xuất file ảnh `.png` gồm: Biểu đồ tụt FPS và Biểu đồ nghẽn CPU.*
