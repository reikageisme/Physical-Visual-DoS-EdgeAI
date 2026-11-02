# Đánh giá YOLOv13 cho định vị miếng dán đối kháng vật lý trên AdvT-shirt-1K

**Tác giả:** Phạm Tuấn Anh, Mai Quốc Bảo — Trường Đại học Công nghệ TP.HCM (HUTECH)  
**Giảng viên hướng dẫn:** Đinh Phương Nam  
**Ngày báo cáo:** 05 tháng 10 năm 2026

## Tóm tắt

Miếng dán đối kháng in có thể làm suy giảm tính toàn vẹn của bộ phát hiện vật thể khi đưa mô hình từ môi trường số ra thế giới thực. Nghiên cứu này đánh giá YOLOv13 trên tác vụ định vị hộp miếng dán, đồng thời đối sánh có kiểm soát với YOLOv8, YOLOv10, YOLO11 và YOLOv12. Các split có nhãn của bộ dữ liệu NCKH gốc hiện chưa có ảnh và nhãn, vì vậy AdvT-shirt-1K được dùng như tập proxy bên ngoài. Bản mirror có 1.123 ảnh và 1.548 hộp miếng dán; split xác định theo seed 42 gồm 785/169/169 ảnh train/validation/test. Mười lăm mô hình Nano/Small/Large được fine-tune với đầu vào 416×416 và lịch huấn luyện đồng nhất. YOLOv13s đạt 83,96% mAP@0.5:0.95 và 99,45% mAP@0.5 trên test, với 9,55 triệu tham số, 9,15 GFLOPs, độ trễ GPU 3,16 ms/ảnh và CPU 38,95 ms/ảnh. YOLOv12s đạt 83,81% và 99,41%, với độ trễ GPU 2,91 ms/ảnh; đây là cấu hình có mAP@0.5:0.95 validation cao nhất trong 15 cấu hình và do đó được chọn báo cáo chính. YOLOv13s cao hơn YOLOv12s 0,16 điểm phần trăm mAP@0.5:0.95 trên test nhưng tăng 0,25 ms độ trễ GPU. Kết quả chỉ chứng minh khả năng định vị trên proxy, không trực tiếp chứng minh khả năng chống tấn công vật lý, quá tải NMS hay hiệu quả trên dữ liệu đề tài gốc.

**Từ khóa:** phát hiện vật thể; miếng dán đối kháng vật lý; định vị patch; YOLOv13; AdvT-shirt-1K; robustness.

> **Đối chiếu nguồn số liệu:** cặp 83,81% mAP@0.5:0.95 và 99,41% mAP@0.5 thuộc YOLOv12s. YOLOv13s có kết quả tương ứng là 83,96% và 99,45% trong benchmark CSV.

## 1. Giới thiệu và mục tiêu nghiên cứu

Miếng dán đối kháng vật lý được in và đặt trong cảnh thật để gây nhầm lẫn cho bộ phát hiện. Đây là rủi ro đối với tính toàn vẹn của nhận thức thị giác trong phương tiện tự hành và thiết bị biên. Định vị vùng patch có thể hỗ trợ hệ thống nhận diện khu vực nghi vấn và kích hoạt kiểm tra bổ sung. Tuy nhiên, phát hiện được hộp patch không đồng nghĩa mô hình đã chống được tấn công hoặc ngăn được suy giảm dịch vụ. [1], [2]

YOLOv13 đề xuất HyperACE để khai thác tương quan đặc trưng bậc cao giữa nhiều vị trí và tỉ lệ; FullPAD phân phối đặc trưng đã tăng cường xuyên suốt pipeline; DS-C3k2 dùng các khối tích chập tách biệt theo chiều sâu nhằm giảm chi phí tính toán. Đây là giả thuyết kiến trúc từ công trình YOLOv13. Hiệu quả trên AdvT-shirt-1K phải được xác định bằng phép đo tại chỗ, không suy diễn trực tiếp từ kết quả COCO. [3]

Nghiên cứu trả lời ba câu hỏi:

- **RQ1 — Chất lượng định vị:** YOLOv13 định vị hộp miếng dán ra sao trên split AdvT-shirt-1K được chia lại?
- **RQ2 — Chi phí suy luận:** độ chính xác đi kèm độ trễ, FPS, tham số và FLOPs nào so với baseline cùng quy mô?
- **RQ3 — Mức độ phòng thủ:** các phép đo hiện có có chứng minh khả năng chống tấn công vật lý hoặc DoS/NMS-overload trong bài toán NCKH gốc không?

## 2. Thiết lập thực nghiệm

### 2.1. Dữ liệu và nhãn

AdvT-shirt-1K gồm ảnh áo có miếng dán trong bối cảnh vật lý, với nhãn hộp cho người và patch. Nghiên cứu này chỉ dùng nhãn patch và một lớp duy nhất. Bản mirror hiện có 1.123 ảnh, thấp hơn con số 1.131 ảnh mà nguồn tác giả nêu; chênh lệch tám ảnh chưa được giải quyết và được xem là hạn chế về tính toàn vẹn nguồn dữ liệu. [1], [2]

Nguồn mirror không cung cấp split phù hợp cho quy trình này. Nhóm tạo split xác định 70/15/15 theo seed 42, phân tầng trong năm thư mục nguồn: train 785 ảnh/1.106 hộp; validation 169 ảnh/220 hộp; test 169 ảnh/222 hộp. Tổng cộng 1.548 hộp. Split được tạo ở mức ảnh, không theo người mặc, thiết kế patch hoặc phiên chụp; vì vậy ảnh tương quan có thể xuất hiện ở nhiều split. Hai nhãn lệch biên 5×10⁻⁷ do làm tròn đã được chặn về [0,1] trong bản split dẫn xuất.

Ảnh gốc rộng 856–2056 pixel và cao 1044–1598 pixel; mọi mô hình nhận đầu vào 416×416. Dataset YAML proxy là dataset/adv_tshirt_patch_detection.yaml. Cấu hình chuẩn dataset/data.yaml dùng path: ./dataset; các split train/validation/test của đề tài hiện có 0 ảnh và 0 nhãn, tên lớp object vẫn là placeholder và không được dùng để huấn luyện.

### 2.2. Phần cứng và phần mềm

| Thành phần | Cấu hình |
|---|---|
| CPU | Intel Core i5-14600KF |
| GPU | NVIDIA GeForce RTX 5060, 7,96 GiB VRAM |
| RAM | 31,79 GiB |
| CUDA | Runtime 12.8 |
| Python | 3.14.7 |
| PyTorch | 2.11.0+cu128 |
| Ultralytics | 8.3.63 |
| Môi trường | project_yolov13/.venv |
| Cấu hình YOLO | project_yolov13/.config |
| Dữ liệu, runs, weights | project_yolov13/dataset, runs, weights |

Các đường dẫn dữ liệu, cache tải dữ liệu, cấu hình YOLO, checkpoint, log và kết quả nằm dưới thư mục dự án. Cây dataset được kiểm tra không có symlink/junction. Không có phép đo trên Jetson hay thiết bị edge chuyên dụng.

### 2.3. Cấu hình huấn luyện

| Siêu tham số | Giá trị |
|---|---|
| Kích thước đầu vào | 416×416 |
| Batch size | 4 |
| Epoch tối đa / patience | 50 / 10 |
| Optimizer | SGD |
| Learning rate ban đầu | 0,01 |
| Scheduler | tuyến tính, lr cuối bằng 0,01 lần lr đầu |
| Warm-up | 2 epoch |
| Weight decay | 0,0005 |
| Seed / deterministic | 42 / bật |
| HSV | h=0,015; s=0,7; v=0,4 |
| Hình học | translate=0,1; scale=0,5; horizontal flip=0,5 |
| Mosaic | 1,0 |
| MixUp / CopyPaste | 0 / 0 |

Mười lăm cấu hình dùng cùng split, seed, pretrained initialization tương ứng, augmentation và lịch huấn luyện. Mỗi cấu hình chỉ chạy một seed nên chưa có ước lượng phương sai hoặc kiểm định ý nghĩa thống kê.

### 2.4. Quy trình chọn mô hình và đo lường

Validation được dùng chọn mô hình/checkpoint; test chỉ dùng báo cáo cuối. YOLOv12s đạt validation mAP@0.5:0.95 cao nhất toàn bộ 15 cấu hình (84,78%), và đạt 83,81% mAP@0.5:0.95 cùng 99,41% mAP@0.5 trên test. Trong so sánh cùng scale, baseline được chọn là model có validation mAP@0.5:0.95 cao nhất ở scale đó.

Latency là thời gian suy luận trung bình mỗi ảnh theo Ultralytics validator tại batch 4, trên CPU/GPU; không gồm thời gian nạp mô hình. FPS* được tính bằng 1000/latency_ms, là nghịch đảo latency bình quân trên ảnh, không phải phép đo throughput video độc lập. Params và FLOPs do profiler Ultralytics đo ở 416 pixel. Các phép đo phụ thuộc thiết bị và runtime, không so trực tiếp với TensorRT FP16 trên Tesla T4.

<!-- LANDSCAPE -->

## 3. Kết quả và benchmark

### 3.1. So sánh trên test split

**Bảng 1.** Mười lăm mô hình fine-tune trên test split AdvT-shirt-1K dẫn xuất. P/R/mAP tính theo phần trăm; latency là ms/ảnh ở batch 4; FPS* suy ra từ latency.
| Mô hình | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | FLOPs (G) | CPU (ms) | CPU FPS* | GPU (ms) | GPU FPS* |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | 98.18 | 99.10 | 99.47 | 83.70 | 3.01 | 3.46 | 11.87 | 84.27 | 1.30 | 769.73 |
| YOLOv10n | 95.77 | 98.65 | 99.32 | 83.02 | 2.71 | 3.55 | 13.68 | 73.09 | 2.01 | 497.88 |
| YOLO11n | 97.95 | 99.10 | 99.37 | 83.65 | 2.59 | 2.72 | 11.87 | 84.23 | 1.69 | 590.53 |
| YOLOv12n | 97.24 | 99.55 | 99.41 | 82.52 | 2.55 | 2.73 | 13.35 | 74.93 | 2.49 | 402.31 |
| YOLOv13n | 99.52 | 98.65 | 99.49 | 79.49 | 2.46 | 2.69 | 16.89 | 59.19 | 3.08 | 324.38 |
| YOLOv8s | 96.63 | 100.00 | 99.33 | 83.72 | 11.14 | 12.10 | 34.81 | 28.73 | 2.12 | 472.52 |
| YOLOv10s | 96.03 | 96.85 | 99.14 | 81.65 | 8.07 | 10.46 | 33.61 | 29.75 | 2.54 | 393.68 |
| YOLO11s | 99.10 | 98.65 | 99.47 | 82.18 | 9.43 | 9.10 | 30.70 | 32.57 | 2.34 | 427.76 |
| YOLOv12s | 97.76 | 98.41 | 99.41 | 83.81 | 9.22 | 9.07 | 33.61 | 29.75 | 2.91 | 343.73 |
| YOLOv13s | 98.21 | 99.10 | 99.45 | 83.96 | 9.55 | 9.15 | 38.95 | 25.67 | 3.16 | 316.55 |
| YOLOv8l | 98.20 | 98.10 | 99.42 | 83.94 | 43.63 | 69.88 | 165.67 | 6.04 | 6.42 | 155.88 |
| YOLOv10l | 96.42 | 97.06 | 99.04 | 82.68 | 25.77 | 53.74 | 140.35 | 7.13 | 6.30 | 158.81 |
| YOLO11l | 98.81 | 98.20 | 99.45 | 83.91 | 25.31 | 36.87 | 108.12 | 9.25 | 5.09 | 196.28 |
| YOLOv12l | 98.84 | 98.20 | 99.45 | 84.25 | 26.92 | 38.23 | 124.92 | 8.01 | 6.74 | 148.46 |
| YOLOv13l | 98.21 | 98.61 | 99.44 | 83.84 | 27.57 | 37.61 | 144.76 | 6.91 | 8.53 | 117.25 |

    % Requires \usepackage{booktabs,graphicx}
    \begin{table*}[t]
    \centering
    \scriptsize
    \caption{Kết quả fine-tune trên test split AdvT-shirt-1K dẫn xuất.}
    \label{tab:advtshirt-yolov13}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lrrrrrrrrrr}
    \toprule
    Model & P (\%) & R (\%) & mAP@0.5 (\%) & mAP@0.5:0.95 (\%) & Params (M) & FLOPs (G) & CPU ms & CPU FPS* & GPU ms & GPU FPS* \\
    \midrule
    YOLOv8n & 98.18 & 99.10 & 99.47 & 83.70 & 3.01 & 3.46 & 11.87 & 84.27 & 1.30 & 769.73 \\
    YOLOv10n & 95.77 & 98.65 & 99.32 & 83.02 & 2.71 & 3.55 & 13.68 & 73.09 & 2.01 & 497.88 \\
    YOLO11n & 97.95 & 99.10 & 99.37 & 83.65 & 2.59 & 2.72 & 11.87 & 84.23 & 1.69 & 590.53 \\
    YOLOv12n & 97.24 & 99.55 & 99.41 & 82.52 & 2.55 & 2.73 & 13.35 & 74.93 & 2.49 & 402.31 \\
    YOLOv13n & 99.52 & 98.65 & 99.49 & 79.49 & 2.46 & 2.69 & 16.89 & 59.19 & 3.08 & 324.38 \\
    YOLOv8s & 96.63 & 100.00 & 99.33 & 83.72 & 11.14 & 12.10 & 34.81 & 28.73 & 2.12 & 472.52 \\
    YOLOv10s & 96.03 & 96.85 & 99.14 & 81.65 & 8.07 & 10.46 & 33.61 & 29.75 & 2.54 & 393.68 \\
    YOLO11s & 99.10 & 98.65 & 99.47 & 82.18 & 9.43 & 9.10 & 30.70 & 32.57 & 2.34 & 427.76 \\
    YOLOv12s & 97.76 & 98.41 & 99.41 & 83.81 & 9.22 & 9.07 & 33.61 & 29.75 & 2.91 & 343.73 \\
    YOLOv13s & 98.21 & 99.10 & 99.45 & 83.96 & 9.55 & 9.15 & 38.95 & 25.67 & 3.16 & 316.55 \\
    YOLOv8l & 98.20 & 98.10 & 99.42 & 83.94 & 43.63 & 69.88 & 165.67 & 6.04 & 6.42 & 155.88 \\
    YOLOv10l & 96.42 & 97.06 & 99.04 & 82.68 & 25.77 & 53.74 & 140.35 & 7.13 & 6.30 & 158.81 \\
    YOLO11l & 98.81 & 98.20 & 99.45 & 83.91 & 25.31 & 36.87 & 108.12 & 9.25 & 5.09 & 196.28 \\
    YOLOv12l & 98.84 & 98.20 & 99.45 & 84.25 & 26.92 & 38.23 & 124.92 & 8.01 & 6.74 & 148.46 \\
    YOLOv13l & 98.21 & 98.61 & 99.44 & 83.84 & 27.57 & 37.61 & 144.76 & 6.91 & 8.53 & 117.25 \\
    \bottomrule
    \end{tabular}}
    \end{table*}


### 3.2. Đánh đổi độ chính xác và tốc độ

**Bảng 2.** YOLOv13 so với baseline cùng scale có validation mAP@0.5:0.95 cao nhất. Test chỉ được dùng sau khi chọn baseline.

| Quy mô | Baseline tốt nhất theo validation | Val mAP95 (%) | Baseline test mAP95 (%) | YOLOv13 | YOLOv13 val mAP95 (%) | YOLOv13 test mAP95 (%) | Δ test (điểm %) | Baseline GPU (ms) | YOLOv13 GPU (ms) | Δ latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Nano | YOLOv8n | 83.85 | 83.70 | YOLOv13n | 81.07 | 79.49 | -4.21 | 1.30 | 3.08 | +1.78 |
| Small | YOLOv12s | 84.78 | 83.81 | YOLOv13s | 84.72 | 83.96 | +0.16 | 2.91 | 3.16 | +0.25 |
| Large | YOLOv12l | 84.69 | 84.25 | YOLOv13l | 84.37 | 83.84 | -0.41 | 6.74 | 8.53 | +1.79 |

<!-- PORTRAIT -->


Ở Nano, YOLOv8n được chọn theo validation; YOLOv13n thấp hơn 4,21 điểm phần trăm mAP@0.5:0.95 trên test và có latency GPU cao hơn. Ở Small, YOLOv13s cao hơn YOLOv12s 0,16 điểm phần trăm nhưng chậm hơn 0,25 ms/ảnh (3,16 so với 2,91 ms). Ở Large, YOLOv13l thấp hơn YOLOv12l 0,41 điểm phần trăm và chậm hơn 1,79 ms/ảnh. Benchmark này không cho thấy YOLOv13 vượt trội đồng đều trên mọi scale.

![Hình 1. mAP@0.5:0.95 và độ trễ GPU của baseline được chọn cùng YOLOv13](adv_tshirt_patch_results/advpatch_tradeoff_selected_latency.png)

![Hình 2. mAP@0.5:0.95 và FLOPs của baseline được chọn cùng YOLOv13](adv_tshirt_patch_results/advpatch_tradeoff_selected_flops.png)

<!-- LANDSCAPE -->

### 3.3. Ablation các thành phần kiến trúc

Ablation chạy trên YOLOv13-N với cùng pretrained initialization, split, seed, lịch và augmentation. Bảng dùng metric validation; Δ so với model đầy đủ. Biến thể no-DS-C3k2 thay thế 20 convolution mục tiêu.

**Bảng 3.** Ablation YOLOv13-N trên validation; latency đo trên GPU.



| Biến thể | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | FLOPs (G) | GPU (ms) | Δ mAP95 (điểm %) | Δ Params (M) | Δ FLOPs (G) | Δ latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv13-n-full | 98.47 | 97.27 | 99.29 | 81.07 | 2.45 | 2.62 | 5.44 | +0.00 | +0.00 | +0.00 | +0.00 |
| YOLOv13-n-no-hyperace | 98.88 | 98.64 | 99.42 | 80.67 | 2.36 | 2.56 | 6.21 | -0.39 | -0.09 | -0.06 | +0.77 |
| YOLOv13-n-no-fullpad | 97.31 | 98.50 | 99.25 | 83.19 | 2.45 | 2.62 | 6.38 | +2.13 | +0.00 | +0.00 | +0.93 |
| YOLOv13-n-no-dsc3k2 | 98.07 | 99.09 | 99.43 | 83.57 | 5.35 | 5.65 | 5.97 | +2.50 | +2.90 | +3.03 | +0.53 |

![Hình 3. Độ chính xác và latency khi lần lượt loại bỏ hoặc thay thế thành phần](adv_tshirt_patch_results/advpatch_ablation_map_latency.png)

<!-- PORTRAIT -->


Bỏ HyperACE làm mAP@0.5:0.95 giảm 0,39 điểm phần trăm; latency đồng thời tăng 0,77 ms. Bỏ FullPAD làm mAP tăng 2,13 điểm nhưng latency tăng 0,93 ms. Thay DS-C3k2 làm mAP tăng 2,50 điểm, song tham số tăng 2,90 triệu, FLOPs tăng 3,03 G và latency tăng 0,53 ms. Vì vậy, ablation không chứng minh cả ba thành phần đều cải thiện đồng thời độ chính xác và tốc độ trên proxy. Một seed chỉ cung cấp bằng chứng mô tả, chưa hỗ trợ kết luận nhân quả phổ quát.

## 4. Thảo luận

### 4.1. Trả lời câu hỏi nghiên cứu

**RQ1 — Chất lượng định vị.** YOLOv13s đạt 83,96% mAP@0.5:0.95 và 99,45% mAP@0.5 trên test. YOLOv12s là cấu hình cao nhất theo validation trên toàn benchmark, đạt 83,81% mAP@0.5:0.95 trên test. Chênh lệch giữa hai mô hình Small là +0,16 điểm phần trăm cho YOLOv13s trên test.

**RQ2 — Chi phí suy luận.** So với YOLOv12s, YOLOv13s có thêm 0,33 triệu tham số, 0,08 GFLOPs, và chậm hơn 0,25 ms/ảnh GPU, đổi lại mAP@0.5:0.95 cao hơn 0,16 điểm trên test. Đây là đánh đổi nhỏ ở scale Small, không phải cải thiện FPS. Kết quả không thay thế benchmark edge; cần đo trên thiết bị đích, với pipeline tiền/hậu xử lý, stream và công suất thực tế.

**RQ3 — Khả năng phòng thủ.** Thí nghiệm chỉ đánh giá định vị hộp patch. Không đo attack success rate, suy giảm recall của đối tượng, số proposal sau NMS, tải NMS hoặc khả năng khái quát hóa sang patch chưa thấy. Do đó, chưa có bằng chứng rằng YOLOv13 chống được tấn công vật lý hoặc ngăn DoS/NMS-overload.

### 4.2. Ý nghĩa các khối YOLOv13

HyperACE hướng đến liên kết nhiều vị trí đặc trưng cùng lúc thay vì chỉ quan hệ từng cặp; FullPAD truyền thông tin tăng cường qua các giai đoạn; DS-C3k2 nhắm giảm phép tính tích chập. Đây là lý do kiến trúc khiến YOLOv13 đáng khảo sát cho patch có tương quan giữa họa tiết, áo và người. Tuy nhiên, bằng chứng địa phương cho thấy bỏ HyperACE làm giảm nhẹ mAP, còn bỏ FullPAD hoặc thay DS-C3k2 lại tăng mAP trong cấu hình và seed này, đồng thời tăng latency hoặc chi phí. Không thể quy chênh lệch giữa các họ mô hình cho riêng một thành phần vì chúng còn khác backbone/head và pretrained weights.

### 4.3. Tính hợp lệ và giới hạn

- **Nội tại:** cùng split, ảnh vào, optimizer, augmentation, seed và validator tăng khả năng đối sánh; khác biệt pretrained weights và early stopping vẫn có thể ảnh hưởng thứ hạng.
- **Split:** chia theo ảnh trong thư mục nguồn, không theo người/patch/session; ảnh tương quan có thể rơi vào nhiều phân vùng.
- **Bên ngoài:** AdvT-shirt-1K là proxy định vị patch trên áo, không thay thế dataset SpongePatch/Physical-Visual DoS của đề tài.
- **Tốc độ:** latency/FPS phụ thuộc GPU, driver, runtime và batch; FPS* là nghịch đảo latency/ảnh, không phải throughput video.
- **Ablation:** một seed, chưa có khoảng tin cậy; cần chạy nhiều seed và test trên thiết kế/người chụp chưa thấy.
- **Triển khai:** chưa đo năng lượng, camera thật, xử lý đầu-cuối hay thiết bị edge chuyên dụng.

Công trình YOLOv13 công bố kết quả COCO để đánh giá kiến trúc. Các số COCO dùng dữ liệu, nhãn, thiết bị và giao thức khác nên không được ghép với kết quả local thành bảng xếp hạng trực tiếp. [3]

## 5. Kết luận và hướng phát triển

Nghiên cứu fine-tune và benchmark 15 cấu hình YOLOv8/10/11/12/13 Nano/Small/Large cho định vị patch trên split dẫn xuất AdvT-shirt-1K. YOLOv13s đạt 83,96% mAP@0.5:0.95 và 99,45% mAP@0.5; so với YOLOv12s, mức tăng là 0,16 điểm phần trăm mAP@0.5:0.95 cùng 0,25 ms latency GPU tăng thêm. Mô hình có validation mAP cao nhất toàn benchmark là YOLOv12s, với test mAP@0.5:0.95 83,81% và mAP@0.5 99,41%. Vì vậy, cặp số 83,81%/99,41% phải được gắn với YOLOv12s.

Kết luận chỉ áp dụng cho định vị patch trên proxy; thí nghiệm chưa xác lập tính kháng đối kháng, hiệu quả ngăn quá tải NMS hoặc lợi thế trên dữ liệu NCKH gốc. Bước tiếp theo là hoàn thiện nhãn đề tài, chia test độc lập theo người/thiết kế patch, lặp nhiều seed, đo attack success và tải NMS trực tiếp, sau đó đánh giá end-to-end trên phần cứng edge.

## Tài liệu tham khảo

[1] AdvT-shirt-1K, dataset card và giấy phép Apache-2.0. [Hugging Face](https://huggingface.co/datasets/xingjunm/Adv-T-shirt).

[2] W. Wang et al., “BadPatch: Diffusion-Based Generation of Physical Adversarial Patches,” ICCV 2025 Workshop. [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Wang_BadPatch_Diffusion-Based_Generation_of_Physical_Adversarial_Patches_ICCVW_2025_paper.html). Nguồn tác giả nêu 1.131 ảnh. [Trang dự án](https://github.com/Wwangb/AdvT-shirt-1K).

[3] M. Lei et al., “YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception,” arXiv:2506.17733, 2025. [Bài báo](https://arxiv.org/abs/2506.17733).

## Phạm vi dữ liệu

Bản mirror có 1.123 ảnh và 1.548 hộp patch theo kiểm kê; nguồn tác giả nêu 1.131 ảnh. Dataset đề tài chưa có ảnh/nhãn trong các split. Mọi kết quả ở đây là kết quả proxy AdvT-shirt-1K, không phải kết quả xác nhận trên bộ dữ liệu NCKH gốc. Chi tiết nguồn và giấy phép nằm ở results/external_dataset_sources.md.
