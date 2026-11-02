# Báo cáo khoa học: Thực nghiệm theo mốc thời gian huấn luyện

**So sánh YOLO11n (baseline) và YOLOv13n trên benchmark định vị patch AdvT-shirt-1K**  
Ngày tổng hợp: 10/10/2026 19:52 SE Asia Standard Time

## Tóm tắt

Nghiên cứu khảo sát mối quan hệ giữa thời gian fine-tuning và độ chính xác của YOLO11n và YOLOv13n trên benchmark định vị patch dẫn xuất từ AdvT-shirt-1K. Sau pilot 10 phút mỗi mô hình, các mốc được điều chỉnh từ lưới ban đầu theo giờ sang 5, 10, 15, 20, 25 và 30 phút. Mỗi mô hình được huấn luyện một lần với cùng lịch tối đa 50 epoch, seed 42 và patience 10; checkpoint tốt nhất theo mAP@0.5:0.95 validation đạt được trước mỗi mốc được đánh giá trên test. Theo tiêu chí sớm nhất nằm trong 0,5 điểm phần trăm so với đỉnh validation, sweet spot là **30 phút** cho YOLO11n và **10 phút** cho YOLOv13n. Tại các mốc này, test mAP@0.5:0.95 lần lượt là **83.65%** và **79.49%** (YOLOv13n − YOLO11n: -4.16 điểm phần trăm). Các kết quả chỉ áp dụng cho benchmark patch bổ sung; tập dữ liệu gốc của dự án chưa có ảnh/nhãn để huấn luyện.

## 1. Thiết lập thực nghiệm

### 1.1 Phần cứng và môi trường

| Thành phần | Cấu hình |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 (7.96 GiB VRAM) |
| CPU | Intel(R) Core(TM) i5-14600KF; 14 nhân vật lý, 20 luồng |
| RAM | 31.79 GiB |
| Môi trường | `.venv`; Python 3.14.7, PyTorch 2.11.0+cu128, Ultralytics 8.3.63 |
| Thiết lập | CUDA 0, AMP, 416×416, batch 4, 0 workers, seed 42 |

### 1.2 Dữ liệu và phạm vi kết luận

Benchmark là split cố định của AdvT-shirt-1K cho phát hiện/định vị patch: train 785 ảnh/1.106 box, validation 169 ảnh/220 box, test 169 ảnh/222 box. Manifest nguồn ghi nhận split này được dẫn xuất cục bộ; đây không phải split benchmark chính thức. Kết quả vì vậy phản ánh bài toán định vị patch trên split này.

Audit `dataset/data.yaml` ghi nhận train/validation/test đều có 0 ảnh và tên lớp `object` là placeholder (trạng thái: `not_ready`). Không thể suy rộng kết quả sang nhiệm vụ gốc trước khi cung cấp tập dữ liệu hợp lệ và ánh xạ lớp đã xác minh.

### 1.3 Mô hình và giao thức thời gian

YOLO11n được dùng làm baseline cỡ nano, YOLOv13n là mô hình cần so sánh. Cả hai được khởi tạo từ pretrained weights và dùng cùng SGD (`lr0=0,01`, momentum 0,937, weight decay 0,0005), ảnh 416, batch 4, AMP, augmentation, seed 42, tối đa 50 epoch và patience 10. Một trajectory cố định cho mỗi mô hình được lưu theo epoch; Ultralytics `time` không bật trong lượt chính để các mốc dùng chung lịch learning rate. Mốc 5/10/15/20/25/30 phút được ánh xạ tới epoch cuối cùng có thời gian tích lũy không vượt mốc. Ở mỗi mốc, chọn checkpoint có validation mAP@0.5:0.95 cao nhất trong các epoch đã hoàn tất; nếu run đã dừng sớm, các mốc dài hơn dùng checkpoint cuối sẵn có và được ghi nhận theo thời gian thực tế.

Pilot cap 10 phút đạt 17 epoch/600,9 giây với YOLO11n (mAP validation tốt nhất 80,88%) và 13 epoch/607,6 giây với YOLOv13n (80,86%). Pilot phục vụ định vị thô; Ultralytics tự ước lượng số epoch theo cap thời gian. Các số pilot không gộp vào Bảng 1. Trong lượt thử ban đầu, YOLO11n cap 1 giờ dừng do patience sau 71 epoch/35,2 phút; cap 2 giờ bị ngắt ở epoch 11, còn cap 4/8 giờ và YOLOv13n chưa chạy. Các run này dùng cap thời gian với lịch tối đa 600 epoch nên được loại khỏi so sánh fixed-50 để tránh trộn lịch learning rate.

### 1.4 Chỉ số

Precision, recall, mAP@0.5 và mAP@0.5:0.95 trong Bảng 1 đo trên test ở batch 4 sau khi checkpoint được chọn bằng validation. Latency được đo riêng trên checkpoint sweet spot của mỗi kiến trúc, batch 1, ba lần; báo cáo trung vị tổng preprocess + inference + postprocess mỗi ảnh. Cùng latency đại diện cho các mốc của cùng kiến trúc; FPS = 1.000/latency. Parameters và FLOPs báo cáo theo kiến trúc tại 416 px. Loss tại mỗi mốc là tổng box + cls + DFL ở epoch cuối đã hoàn thành trước cap, không phải loss của checkpoint validation tốt nhất.

## 2. Kết quả

### Bảng 1. Độ chính xác test và tốc độ theo thời gian

| Mô hình | Mốc cap (phút) | Thời gian checkpoint (phút) | Epoch chọn | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Latency (ms/frame) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n (Baseline) | 5 | 4.5 | 6 | 96.62 | 99.10 | 99.01 | 77.95 | 15.27 | 65.5 |
| YOLO11n (Baseline) | 10 | 8.5 | 12 | 98.83 | 98.20 | 99.38 | 79.04 | 15.27 | 65.5 |
| YOLO11n (Baseline) | 15 | 13.3 | 19 | 98.12 | 98.65 | 99.43 | 81.25 | 15.27 | 65.5 |
| YOLO11n (Baseline) | 20 | 19.3 | 28 | 97.26 | 98.65 | 99.41 | 81.54 | 15.27 | 65.5 |
| YOLO11n (Baseline) | 25 | 24.7 | 36 | 97.32 | 98.20 | 99.44 | 82.48 | 15.27 | 65.5 |
| YOLO11n (Baseline) | 30 | 25.4 | 37 | 97.95 | 99.10 | 99.37 | 83.65 | 15.27 | 65.5 |
| YOLOv13n | 5 | 4.7 | 6 | 98.64 | 98.29 | 99.45 | 78.98 | 29.85 | 33.5 |
| YOLOv13n | 10 | 8.6 | 11 | 99.52 | 98.65 | 99.49 | 79.49 | 29.85 | 33.5 |
| YOLOv13n | 15 | 8.6 | 11 | 99.52 | 98.65 | 99.49 | 79.49 | 29.85 | 33.5 |
| YOLOv13n | 20 | 8.6 | 11 | 99.52 | 98.65 | 99.49 | 79.49 | 29.85 | 33.5 |
| YOLOv13n | 25 | 8.6 | 11 | 99.52 | 98.65 | 99.49 | 79.49 | 29.85 | 33.5 |
| YOLOv13n | 30 | 8.6 | 11 | 99.52 | 98.65 | 99.49 | 79.49 | 29.85 | 33.5 |

### Bảng 1a. Mã LaTeX: kết quả test

```latex
\begin{table}[htbp]
\centering
\caption{Kết quả trên test theo thời điểm huấn luyện; checkpoint được chọn theo validation.}
\label{tab:milestone-test}
\begin{tabular}{llrrrrrrrr}
\hline
Mô hình & Cap (phút) & Thực (phút) & Epoch & P (%) & R (%) & mAP50 (%) & mAP50:95 (%) & Latency (ms) & FPS \\
\hline
YOLO11n (Baseline) & 5 & 4.5 & 6 & 96.62 & 99.10 & 99.01 & 77.95 & 15.27 & 65.5 \\
YOLO11n (Baseline) & 10 & 8.5 & 12 & 98.83 & 98.20 & 99.38 & 79.04 & 15.27 & 65.5 \\
YOLO11n (Baseline) & 15 & 13.3 & 19 & 98.12 & 98.65 & 99.43 & 81.25 & 15.27 & 65.5 \\
YOLO11n (Baseline) & 20 & 19.3 & 28 & 97.26 & 98.65 & 99.41 & 81.54 & 15.27 & 65.5 \\
YOLO11n (Baseline) & 25 & 24.7 & 36 & 97.32 & 98.20 & 99.44 & 82.48 & 15.27 & 65.5 \\
YOLO11n (Baseline) & 30 & 25.4 & 37 & 97.95 & 99.10 & 99.37 & 83.65 & 15.27 & 65.5 \\
YOLOv13n & 5 & 4.7 & 6 & 98.64 & 98.29 & 99.45 & 78.98 & 29.85 & 33.5 \\
YOLOv13n & 10 & 8.6 & 11 & 99.52 & 98.65 & 99.49 & 79.49 & 29.85 & 33.5 \\
YOLOv13n & 15 & 8.6 & 11 & 99.52 & 98.65 & 99.49 & 79.49 & 29.85 & 33.5 \\
YOLOv13n & 20 & 8.6 & 11 & 99.52 & 98.65 & 99.49 & 79.49 & 29.85 & 33.5 \\
YOLOv13n & 25 & 8.6 & 11 & 99.52 & 98.65 & 99.49 & 79.49 & 29.85 & 33.5 \\
YOLOv13n & 30 & 8.6 & 11 & 99.52 & 98.65 & 99.49 & 79.49 & 29.85 & 33.5 \\
\hline
\end{tabular}
\end{table}
```

### Bảng 1b. Tài nguyên mô hình

| Mô hình | Mốc cap (phút) | Parameters (M) | FLOPs (G) | Epoch endpoint | Epoch checkpoint |
|---|---:|---:|---:|---:|---:|
| YOLO11n (Baseline) | 5 | 2.590 | 2.72 | 6 | 6 |
| YOLO11n (Baseline) | 10 | 2.590 | 2.72 | 14 | 12 |
| YOLO11n (Baseline) | 15 | 2.590 | 2.72 | 21 | 19 |
| YOLO11n (Baseline) | 20 | 2.590 | 2.72 | 28 | 28 |
| YOLO11n (Baseline) | 25 | 2.590 | 2.72 | 36 | 36 |
| YOLO11n (Baseline) | 30 | 2.590 | 2.72 | 46 | 37 |
| YOLOv13n | 5 | 2.460 | 2.69 | 6 | 6 |
| YOLOv13n | 10 | 2.460 | 2.69 | 13 | 11 |
| YOLOv13n | 15 | 2.460 | 2.69 | 20 | 11 |
| YOLOv13n | 20 | 2.460 | 2.69 | 21 | 11 |
| YOLOv13n | 25 | 2.460 | 2.69 | 21 | 11 |
| YOLOv13n | 30 | 2.460 | 2.69 | 21 | 11 |

### Bảng 1c. Loss tại cuối mốc thời gian

| Mô hình | Cap (phút) | Thời gian endpoint (phút) | Train loss tổng | Val loss tổng | Train box/cls/dfl | Val box/cls/dfl |
|---|---:|---:|---:|---:|---|---|
| YOLO11n (Baseline) | 5 | 4.5 | 2.703 | 2.289 | 0.948 / 0.699 / 1.056 | 0.803 / 0.532 / 0.954 |
| YOLO11n (Baseline) | 10 | 9.9 | 2.443 | 2.123 | 0.882 / 0.537 / 1.023 | 0.766 / 0.424 / 0.933 |
| YOLO11n (Baseline) | 15 | 14.6 | 2.341 | 2.020 | 0.858 / 0.480 / 1.002 | 0.722 / 0.378 / 0.920 |
| YOLO11n (Baseline) | 20 | 19.3 | 2.258 | 2.042 | 0.817 / 0.453 / 0.988 | 0.755 / 0.367 / 0.920 |
| YOLO11n (Baseline) | 25 | 24.7 | 2.169 | 1.938 | 0.784 / 0.419 / 0.966 | 0.712 / 0.325 / 0.900 |
| YOLO11n (Baseline) | 30 | 29.9 | 1.934 | 1.909 | 0.699 / 0.327 / 0.909 | 0.708 / 0.304 / 0.898 |
| YOLOv13n | 5 | 4.7 | 2.664 | 2.246 | 0.925 / 0.658 / 1.081 | 0.786 / 0.484 / 0.976 |
| YOLOv13n | 10 | 10.0 | 2.467 | 2.192 | 0.882 / 0.513 / 1.072 | 0.774 / 0.432 / 0.985 |
| YOLOv13n | 15 | 14.6 | 2.405 | 2.149 | 0.858 / 0.471 / 1.076 | 0.751 / 0.395 / 1.003 |
| YOLOv13n | 20 | 15.2 | 2.381 | 2.149 | 0.854 / 0.467 / 1.060 | 0.766 / 0.404 / 0.980 |
| YOLOv13n | 25 | 15.2 | 2.381 | 2.149 | 0.854 / 0.467 / 1.060 | 0.766 / 0.404 / 0.980 |
| YOLOv13n | 30 | 15.2 | 2.381 | 2.149 | 0.854 / 0.467 / 1.060 | 0.766 / 0.404 / 0.980 |

### Mã LaTeX cho bảng tài nguyên và loss

```latex
\begin{table}[htbp]
\centering
\caption{Chi phí tài nguyên và loss ở cuối ngân sách.}
\label{tab:milestone-resources-loss}
\begin{tabular}{llrrrrrr}
\hline
Mô hình & Cap (phút) & Params (M) & FLOPs (G) & Epoch cuối & Epoch chọn & Train loss & Val loss \\
\hline
YOLO11n (Baseline) & 5 & 2.590 & 2.72 & 6 & 6 & 2.703 & 2.289 \\
YOLO11n (Baseline) & 10 & 2.590 & 2.72 & 14 & 12 & 2.443 & 2.123 \\
YOLO11n (Baseline) & 15 & 2.590 & 2.72 & 21 & 19 & 2.341 & 2.020 \\
YOLO11n (Baseline) & 20 & 2.590 & 2.72 & 28 & 28 & 2.258 & 2.042 \\
YOLO11n (Baseline) & 25 & 2.590 & 2.72 & 36 & 36 & 2.169 & 1.938 \\
YOLO11n (Baseline) & 30 & 2.590 & 2.72 & 46 & 37 & 1.934 & 1.909 \\
YOLOv13n & 5 & 2.460 & 2.69 & 6 & 6 & 2.664 & 2.246 \\
YOLOv13n & 10 & 2.460 & 2.69 & 13 & 11 & 2.467 & 2.192 \\
YOLOv13n & 15 & 2.460 & 2.69 & 20 & 11 & 2.405 & 2.149 \\
YOLOv13n & 20 & 2.460 & 2.69 & 21 & 11 & 2.381 & 2.149 \\
YOLOv13n & 25 & 2.460 & 2.69 & 21 & 11 & 2.381 & 2.149 \\
YOLOv13n & 30 & 2.460 & 2.69 & 21 & 11 & 2.381 & 2.149 \\
\hline
\end{tabular}
\end{table}
```

### Bảng 2. Ma trận điểm tối ưu

Sweet spot là mốc cap nhỏ nhất mà checkpoint tốt nhất đến mốc đó nằm trong 0,5 điểm phần trăm mAP@0.5:0.95 validation so với đỉnh validation của toàn trajectory. Tiêu chí này chọn theo validation; test chỉ dùng để báo cáo hiệu năng của checkpoint đã chọn.

| Mô hình | Cap sweet spot (phút) | Thời gian checkpoint (phút) | Epoch | Val mAP@0.5:0.95 (%) | Test mAP@0.5:0.95 (%) | Đỉnh val toàn run (%) |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n (Baseline) | 30 | 25.4 | 37 | 83.72 | 83.65 | 83.72 |
| YOLOv13n | 10 | 8.6 | 11 | 81.13 | 79.49 | 81.13 |

#### Mã LaTeX cho Bảng 2

```latex
\begin{table}[htbp]
\centering
\caption{Sweet spot theo mAP validation trong dung sai 0,5 điểm phần trăm.}
\label{tab:sweet-spot}
\begin{tabular}{lrrrrr}
\hline
Mô hình & Cap (phút) & Thời gian ckpt (phút) & Epoch & Val mAP50:95 (%) & Test mAP50:95 (%) \\
\hline
YOLO11n (Baseline) & 30 & 25.4 & 37 & 83.72 & 83.65 \\
YOLOv13n & 10 & 8.6 & 11 & 81.13 & 79.49 \\
\hline
\end{tabular}
\end{table}
```

## 3. Phân tích và thảo luận

### 3.1 Đường cong mAP theo thời gian

![Đường cong mAP validation theo thời gian](map_growth.png)

Hình thể hiện mAP@0.5:0.95 validation sau mỗi epoch. Dấu sao là đỉnh validation của từng trajectory; các vạch dọc chỉ các mốc khảo sát. Các đường có thể dao động do tập validation nhỏ (169 ảnh), vì vậy lựa chọn checkpoint dùng validation tốt nhất tích lũy đến mốc thay vì coi một epoch cuối đơn lẻ là đại diện.

| Mô hình | Mốc (phút) | Best-so-far val mAP@0.5:0.95 (%) | Δ so với mốc trước (điểm %) | Mức tăng biên (điểm %/phút) | Epoch checkpoint |
|---|---:|---:|---:|---:|---:|
| YOLO11n (Baseline) | 5 | 77.86 | — | — | 6 |
| YOLO11n (Baseline) | 10 | 78.94 | 1.07 | 0.215 | 12 |
| YOLO11n (Baseline) | 15 | 80.62 | 1.69 | 0.337 | 19 |
| YOLO11n (Baseline) | 20 | 81.61 | 0.99 | 0.197 | 28 |
| YOLO11n (Baseline) | 25 | 82.79 | 1.18 | 0.237 | 36 |
| YOLO11n (Baseline) | 30 | 83.72 | 0.92 | 0.185 | 37 |
| YOLOv13n | 5 | 79.07 | — | — | 6 |
| YOLOv13n | 10 | 81.13 | 2.06 | 0.413 | 11 |
| YOLOv13n | 15 | 81.13 | 0.00 | 0.000 | 11 |
| YOLOv13n | 20 | 81.13 | 0.00 | 0.000 | 11 |
| YOLOv13n | 25 | 81.13 | 0.00 | 0.000 | 11 |
| YOLOv13n | 30 | 81.13 | 0.00 | 0.000 | 11 |

### 3.2 Sweet spot và hiệu quả thời gian

YOLO11n đạt đỉnh validation 83.72% ở epoch 37 (khoảng 25.4 phút đến checkpoint đầu tiên nằm trong dung sai 0,5 điểm phần trăm). YOLOv13n đạt đỉnh 81.13% ở epoch 11 (khoảng 8.6 phút đến vùng tương đương). Theo các cap khảo sát, điều này tương ứng sweet spot 30 phút và 10 phút. YOLOv13n cần ít thời gian hơn để đạt vùng đỉnh theo tiêu chí đã định; chênh lệch test mAP@0.5:0.95 tại sweet spot là -4.16 điểm phần trăm so với baseline. Tổng training dừng ở 15,25 phút với YOLOv13n và 30,40 phút với YOLO11n; checkpoint sweet spot xuất hiện lần lượt ở 8,6 và 25,4 phút. Latency batch 1 của YOLOv13n là 29,85 ms/frame (33,5 FPS), so với 15,27 ms/frame (65,5 FPS) của baseline. Do đó YOLOv13n đạt vùng validation đỉnh sớm hơn và có ít tham số/FLOPs, nhưng tại sweet spot có test mAP@0.5:0.95 thấp hơn và latency cao hơn trên GPU này.

Sau đỉnh validation, YOLO11n tiếp tục khoảng 5,0 phút và YOLOv13n khoảng 6,7 phút trước khi patience dừng run, nhưng không đạt mAP cao hơn. Các mốc dài hơn chỉ lặp checkpoint cuối. Đây là dấu hiệu diminishing returns trong cấu hình và split đã thử, không phải khẳng định phổ quát cho mọi dữ liệu.

### 3.3 Hội tụ, loss, overfitting và underfitting

**YOLO11n:** Loss train và validation cùng giảm hoặc ổn định; chưa thấy tín hiệu overfitting rõ. (Δ train loss theo các cửa sổ đầu/cuối: -1.119; Δ val loss: -0.730).

**YOLOv13n:** Loss train và validation cùng giảm hoặc ổn định; chưa thấy tín hiệu overfitting rõ. (Δ train loss theo các cửa sổ đầu/cuối: -0.701; Δ val loss: -0.409).

Train loss giảm từ 3.900 xuống 1.943 với YOLO11n và từ 3.935 xuống 2.381 với YOLOv13n. Hai run hoàn tất lần lượt 47 và 21 epoch; thời gian training tích lũy là 30.4 và 15.2 phút. Loss giảm cùng với mAP cao ở vùng đỉnh nên không thấy dấu hiệu underfitting rõ trong benchmark này. Cũng chưa có bằng chứng overfitting mạnh vì val loss giảm; tuy vậy validation chỉ có 169 ảnh và mAP dao động, nên nhận định cần xem là mô tả chứ không phải kiểm định thống kê.

### 3.4 Trần hiệu năng và khuyến nghị

Mốc khảo sát dày 5 phút phù hợp hơn lưới 1/2/4/8 giờ vì các trajectory đã đạt vùng đỉnh validation trong vài đến vài chục phút. Với cấu hình hiện tại, dùng 10 phút làm screening ban đầu cho YOLOv13n; dùng đến 30 phút khi cần kiểm tra đầy đủ vùng plateau của YOLO11n. Các nghiên cứu tiếp theo nên lặp ít nhất 3 seed, mở rộng sang split/dataset đúng của nhiệm vụ gốc, và đo latency riêng ở batch 1 trên thiết bị edge đích. Khi dữ liệu hoặc augmentation thay đổi, cần hiệu chỉnh lại các mốc.

## 4. Kết luận

Thử nghiệm pilot đã thay lưới cap dài theo giờ bằng các mốc 5–30 phút, dựa trên thời gian hội tụ quan sát được. Giao thức chính dùng chung một lịch 50 epoch cho mỗi mô hình và checkpoint từng epoch; các mốc được đánh giá theo best-so-far validation, sau đó báo cáo test. Sweet spot theo dung sai 0,5 điểm phần trăm là 30 phút với YOLO11n và 10 phút với YOLOv13n. Kết luận về mô hình chỉ có phạm vi của benchmark patch dẫn xuất; cần tập dữ liệu gốc hợp lệ và nhiều seed trước khi đưa ra khuyến nghị triển khai rộng.

## 5. Tệp và khả năng tái lập

- Kết quả theo mốc: `results/time_budget/milestone_records.json` và `.csv`.
- Cấu hình môi trường: `results/time_budget/milestone_environment.json`.
- Log: `results/time_budget/logs/milestones-20261010-*-fixed50.log`.
- CSV theo epoch/checkpoint: `runs/train/milestones-20261010-*-fixed50/`.
- Pilot: `results/time_budget/pilot/pilot_records.json` và thư mục log/checkpoint tương ứng.
- Manifest split: `results/adv_tshirt_patch_dataset_manifest.json`.
- Audit dữ liệu gốc: `results/dataset_audit_current.json`.

## Phụ lục. Cấu hình

```text
models = YOLO11n pretrained; YOLOv13n pretrained
milestones = 5, 10, 15, 20, 25, 30 minutes
one fixed-schedule run/model; epochs=50; patience=10; save_period=1
imgsz=416; batch=4; seed=42; optimizer=SGD; AMP=True; workers=0
dataset=dataset/adv_tshirt_patch_detection.yaml
checkpoint selection = highest validation mAP50:95 observed by each milestone
test evaluation = after validation-based selection; one test split, 169 images
```
