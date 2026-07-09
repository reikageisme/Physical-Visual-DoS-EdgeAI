# Sponge Patch — Physical Visual Denial-of-Service on Edge-AI Systems

<p align="center">
  <a href="https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/README.md">🇺🇸 English</a> | 
  <a href="https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/README.vi.md">🇻🇳 Tiếng Việt</a>
</p>

---

## 📖 Tổng quan Dự án

**Sponge Patch** là một framework nghiên cứu được thiết kế để đánh giá và trình diễn các cuộc tấn công **Physical Visual Denial-of-Service (Từ chối Dịch vụ Thị giác Vật lý - DoS)** nhắm vào các hệ thống phát hiện vật thể Edge-AI, đặc biệt là YOLOv8n. Khác với các cuộc tấn công đối kháng (adversarial attacks) thông thường nhằm mục đích làm giảm độ chính xác của mô hình (ví dụ: giấu vật thể hoặc phân loại sai), mục tiêu của tấn công Visual DoS là làm cạn kiệt tài nguyên tính toán của mô hình nạn nhân.

Bằng cách sử dụng **Thuật toán Di truyền Dựa trên Độ nổi bật (Saliency-Guided Genetic Algorithm - GA)**, chúng tôi tạo ra các miếng dán (patch) đối kháng vật lý có khả năng tạo ra một lượng khổng lồ các bounding box giả (false positives). Khi mô hình nạn nhân xử lý các đầu vào này, bước hậu xử lý — cụ thể là Non-Maximum Suppression (NMS) — bị quá tải, gây ra sự gia tăng đột biến về độ trễ. Trên các thiết bị Edge có tài nguyên hạn chế, điều này dẫn đến rớt khung hình (frame drops), hệ thống không phản hồi và dẫn đến tình trạng Từ chối Dịch vụ (DoS) hoàn toàn.

### Các Tính năng Chính
- **Tối ưu hóa Dựa trên Độ nổi bật (Saliency-Guided)**: Tập trung Thuật toán Di truyền vào các khu vực có độ nổi bật cao nhất để tối đa hóa hiệu quả của patch và khả năng ngụy trang thị giác.
- **Tính mạnh mẽ trong Môi trường Thực tế (Physical-World Robustness)**: Sử dụng Expectation Over Transformation (EOT) để đảm bảo patch vẫn giữ được hiệu quả khi được in ra và camera thu hình thực tế dưới nhiều góc độ, ánh sáng và nhiễu khác nhau.
- **Mô phỏng Hệ thống Edge Toàn diện**: Đánh giá tác động của cuộc tấn công trên các giới hạn phần cứng khác nhau, mô phỏng các môi trường như Raspberry Pi, Intel NUC, và Jetson Nano.
- **Bộ Công cụ Đánh giá Chuyên sâu**: Bao gồm các thử nghiệm đa hạt giống (multi-seed) để đảm bảo tính thống kê, các nghiên cứu cắt bỏ (ablation studies), và so sánh với các baseline (Random, Checkerboard).

---

## 🎯 Mô hình Mối đe dọa (Threat Model) & Kịch bản Tấn công

Hiểu rõ các khả năng của kẻ tấn công là điều cực kỳ quan trọng. Dự án này chủ yếu hoạt động dưới mô hình mối đe dọa **Gray-box** trong quá trình tạo patch, và mô phỏng quá trình triển khai thực tế khi kẻ tấn công chỉ có quyền truy cập vật lý vào trường nhìn (FOV) của camera.

| Cấp độ Mô hình Mối đe dọa | Quyền truy cập của Kẻ tấn công | Thành phần / Cách sử dụng trong Framework |
| :--- | :--- | :--- |
| **Gray-box (Chính)** | Lấy điểm confidence score gốc và tọa độ bounding box trước khi qua NMS từ một mô hình local. Không có quyền truy cập vào gradient bên trong mạng nơ-ron. | Được sử dụng bởi `main_train.py` và Thuật toán Di truyền để tối ưu hóa patch mà không cần lan truyền ngược (backpropagation). |
| **Observable Black-box** | Chỉ nhận được các detection sau NMS và số đo độ trễ suy luận tổng thể. | Được sử dụng bởi `ObservableFitness` trong `sponge_fitness.py` để đánh giá tác động thực tế trên thiết bị Edge. |
| **White-box (Baseline)** | Quyền truy cập đầy đủ vào kiến trúc và gradient của mô hình. | Được sử dụng bởi `fast_train.py` (PGD) để thiết lập giới hạn trên (upper bound) lý thuyết cho cuộc tấn công. |

---

## 🚀 Cài đặt & Bắt đầu Nhanh

### 1. Yêu cầu Hệ thống
Đảm bảo bạn đã cài đặt Python 3.9+. Clone repository và cài đặt các thư viện phụ thuộc:

```bash
git clone https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI.git
cd Physical-Visual-DoS-EdgeAI
pip install -r requirements.txt
```

### 2. Chạy Mô phỏng Toàn bộ Pipeline
Bạn có thể mô phỏng toàn bộ chu trình tấn công (từ việc huấn luyện patch cho đến đánh giá tác động của nó lên độ trễ NMS) mà không cần dùng đến camera thực tế.

```bash
# Chạy demo nhanh (~2 phút)
python simulate_edge_server.py --quick

# Thử nghiệm đầy đủ: huấn luyện một patch bằng GA và test cả hai kịch bản sạch/bị tấn công
python simulate_edge_server.py --frames 200 --train --pop 20 --gen 30

# Mô phỏng giới hạn phần cứng cụ thể (ví dụ: Raspberry Pi)
python simulate_edge_server.py --frames 200 --train --edge-profile raspberry_pi

# Đánh giá sử dụng một patch đã huấn luyện từ trước
python simulate_edge_server.py --frames 200 --patch outputs/sponge_patch.png
```

---

## 🔬 Bộ Thử nghiệm Đánh giá Chuyên sâu

Framework cung cấp các module độc lập để đánh giá một cách khắt khe tính hiệu quả, tính mạnh mẽ và các biện pháp phòng thủ tiềm năng của cuộc tấn công.

### Đánh giá Tính Thống kê (Statistical Rigor)
Để đảm bảo cuộc tấn công ổn định, chúng tôi cung cấp script đánh giá đa hạt giống để tính toán giá trị trung bình và độ lệch chuẩn qua nhiều lần chạy tối ưu hóa.
```bash
python experiments/multi_seed_experiment.py --n-seeds 10 --pop 20 --gen 30
```

### Ablation Studies (Nghiên cứu Cắt bỏ)
Phân tích sự tương quan giữa kích thước vật lý của adversarial patch với sự gia tăng độ trễ NMS.
```bash
python experiments/ablation_patch_size.py
```

### So sánh Baseline
So sánh Sponge Patch đã được tối ưu hóa với các patch đối chứng tiêu chuẩn (Nhiễu ngẫu nhiên và Họa tiết bàn cờ) để xác nhận độ hiệu quả của Thuật toán Di truyền.
```bash
python experiments/baseline_comparison.py --patch outputs/sponge_patch.png
```

### Đánh giá Các phương pháp Phòng thủ (Defense Evaluations)
Thử nghiệm patch đối phó với các kỹ thuật giảm thiểu tiêu chuẩn, như tăng ngưỡng confidence (confidence threshold) hoặc giới hạn số lượng detection tối đa (`max_det`).
```bash
python experiments/defense_evaluation.py --patch outputs/sponge_patch.png
```

---

## 🎛️ Huấn luyện & Thử nghiệm Thực tế

### Huấn luyện Patch Độc lập
Nếu bạn chỉ muốn tạo ra một patch để sử dụng sau (ví dụ: để in ra giấy), bạn có thể chạy script huấn luyện độc lập.

```bash
# Huấn luyện một patch kích thước 64x64px (~4% khung hình 320x320)
python main_train.py --size 64 --pop 30 --gen 50 --seed 42

# Huấn luyện sử dụng tỷ lệ phần trăm diện tích so với khung hình
python main_train.py --area-pct 4.0 --resolution 320 --seed 42
```

### Thử nghiệm với Camera Thực tế
Thử nghiệm patch đã tạo trong kịch bản thế giới thực bằng webcam của bạn.

```bash
# Chạy baseline sạch (không có tấn công)
python test_physical_dos.py --scenario clean

# Chạy tấn công chèn kỹ thuật số (overlay patch lên video stream)
python test_physical_dos.py --patch outputs/sponge_patch.png --scenario digital_attack

# Chạy mô phỏng headless (bỏ qua camera, sử dụng các frame được tạo giả lập)
python test_physical_dos.py --simulate --frames 200 --scenario digital_attack
```

---

## 📊 Phân tích Cấu hình Phần cứng & Trực quan hóa Kết quả

### Profile Phần cứng Edge
Framework có thể tự động bóp băng thông tài nguyên hệ thống để mô phỏng việc triển khai YOLOv8n trên các phần cứng bị hạn chế.

| Tên Profile | Luồng CPU | GPU Hỗ trợ | Mục tiêu Mô phỏng |
| :--- | :--- | :--- | :--- |
| `raspberry_pi` | 1 | ❌ | Raspberry Pi 4 |
| `intel_nuc` | 2 | ❌ | Intel NUC Edge Server |
| `jetson_nano` | 2 | ✅ | NVIDIA Jetson Nano |
| `full_server` | Toàn bộ | ✅ | Desktop / Server không giới hạn |

### Trực quan hóa Kết quả
Sau khi chạy các mô phỏng, dữ liệu telemetry sẽ được lưu trong thư mục `logs/`. Sử dụng công cụ vẽ biểu đồ để trực quan hóa sự suy giảm hiệu suất.

```bash
# Vẽ biểu đồ từ file log mô phỏng gần nhất
python utils/plot_results.py

# Vẽ biểu đồ phân tích độ trễ chi tiết (Tiền xử lý vs. Forward Pass vs. NMS)
python utils/plot_results.py --file logs/resource_log_xxx.csv --breakdown
```

---

## 📁 Kiến trúc Dự án

Sơ lược về codebase để giúp bạn dễ dàng theo dõi:

```text
Physical-Visual-DoS-EdgeAI/
├── simulate_edge_server.py       # Entry point chính cho mô phỏng toàn diện (end-to-end)
├── main_train.py                 # Script độc lập để tối ưu hóa patch bằng GA
├── test_physical_dos.py          # Thử nghiệm camera thực tế và mô phỏng headless
│
├── core/
│   ├── victim_model.py           # Wrapper YOLOv8 với các hàm profiling NMS
│   ├── sponge_fitness.py         # Hàm Fitness để đánh giá độ trễ GrayBox/BlackBox
│   └── eot_transforms.py         # Data Augmentation EOT dựa trên Kornia để chống chịu nhiễu vật lý
│
├── attack/
│   └── genetic_algo.py           # Triển khai Thuật toán Di truyền Dựa trên Độ nổi bật (Saliency-Guided GA)
│
├── experiments/                  # Các script xác thực tính thống kê và cắt bỏ
│   ├── multi_seed_experiment.py  
│   ├── ablation_patch_size.py    
│   ├── baseline_comparison.py    
│   └── defense_evaluation.py     
│
├── utils/
│   ├── monitor.py                # Công cụ theo dõi tài nguyên phần cứng và ghi nhận độ trễ
│   └── plot_results.py           # Trực quan hóa dữ liệu từ các file CSV logs
│
└── outputs/                      # Các patch được tạo ra (bao gồm file để in A4) và đồ thị
```
