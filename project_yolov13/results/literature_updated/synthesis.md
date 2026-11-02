# Tổng hợp nguồn và đối sánh giao thức

Ngày truy xuất: 13/10/2026. Đây là rà soát tường thuật có chủ đích các kiến trúc liên quan; không phải tìm kiếm hệ thống đầy đủ hay bảng xếp hạng toàn bộ SOTA. Số liệu trong bảng là kết quả công bố, không phải kết quả thực nghiệm của dự án.

## Ma trận bài báo và phần mềm

| Nguồn / biến thể | Năm | Dữ liệu / lớp / split | AP50 (%) | AP50:95 (%) | Params (M) | FLOPs (G), 640px | Latency/FPS công bố |
|---|---:|---|---:|---:|---:|---:|---|
| [YOLO11n](https://docs.ultralytics.com/models/yolo11/) | 2024 | COCO2017 / 80 / val2017 | — | 39.5 | 2.6 | 6.5 | 1.5 ms |
| [YOLO11s](https://docs.ultralytics.com/models/yolo11/) | 2024 | COCO2017 / 80 / val2017 | — | 47 | 9.4 | 21.6 | 2.5 ms |
| [YOLOv12-N](https://arxiv.org/html/2502.12524v1) | 2025 | COCO2017 / 80 / val2017 | 56.7 | 40.6 | 2.6 | 6.5 | 1.64 ms |
| [YOLOv12-S](https://arxiv.org/html/2502.12524v1) | 2025 | COCO2017 / 80 / val2017 | 65 | 48 | 9.3 | 21.4 | 2.61 ms |
| [YOLOv13-N](https://arxiv.org/html/2506.17733v1) | 2025 | COCO2017 / 80 / val2017 | 57.8 | 41.6 | 2.5 | 6.4 | 1.97 ms |
| [YOLOv13-S](https://arxiv.org/html/2506.17733v1) | 2025 | COCO2017 / 80 / val2017 | 65.2 | 48 | 9 | 20.8 | 2.98 ms |
| [YOLOv10-N](https://arxiv.org/html/2405.14458v1) | 2024 | COCO2017 / 80 / val2017 | 53.8 | 38.5 | 2.3 | 6.7 | 1.84 ms |
| [YOLOv10-S](https://arxiv.org/html/2405.14458v1) | 2024 | COCO2017 / 80 / val2017 | 63 | 46.3 | 7.2 | 21.6 | 2.49 ms |
| [RT-DETR-R18](https://github.com/lyuwenyu/RT-DETR) | 2024 | COCO2017 / 80 / val2017 | 63.8 | 46.5 | 20 | 60 | 217 FPS |
| [RT-DETR-R50](https://github.com/lyuwenyu/RT-DETR) | 2024 | COCO2017 / 80 / val2017 | 71.3 | 53.1 | 42 | 136 | 108 FPS |
| [RT-DETRv2-S](https://arxiv.org/html/2407.17140v1) | 2024 | COCO2017 / 80 / val2017 | 64.9 | 47.9 | 20 | — | 217 FPS |

AP là COCO bounding-box AP; dấu — nghĩa là chưa xác định trong nguồn đã kiểm tra. YOLO11 được dẫn dưới dạng phần mềm chính thức vì nhà phát triển nói rõ chưa có bài báo nghiên cứu chính thức. Các hàng YOLOv12 cố định arXiv v1; YOLOv13 cố định arXiv v1. Hàng RT-DETR chỉ dùng COCO, không dùng kết quả có pretraining Objects365. GPU các hàng tốc độ là T4; YOLOv10/12/13 và RT-DETR dùng TensorRT FP16, còn bảng tài liệu YOLO11 ghi TensorRT10 và chưa xác nhận dtype trong trang đó. Batch, phạm vi preprocessing và số lần đo còn thiếu được giữ `null` trong JSON.

## Khác biệt phiên bản phải giải thích

- [YOLOv12 repo hiện tại](https://github.com/sunsmarterjie/yolov12) có Turbo; không đồng nhất với bản v1 đã dùng trong bài gốc và dự án.
- [YOLOv13, Table I và Sec. IV-A](https://arxiv.org/html/2506.17733v1) tái lập YOLO11/12 trên phần cứng huấn luyện của tác giả. Các mức chênh lệch trong paper đó phải tính theo baseline được tái lập, không ghép baseline từ tài liệu khác.
- [RT-DETRv2 repo](https://github.com/lyuwenyu/RT-DETR) có cập nhật S lên 48.1 AP; bảng này cố định technical report v1 (47.9).
- [BadPatch arXiv v5](https://arxiv.org/abs/2412.01440v5) có tiêu đề và ba tác giả hiện tại; citation DiffPatch cũ trong README bộ dữ liệu không nên chép nguyên làm nguồn mới.

## Bối cảnh vật lý và giới hạn về nhiệm vụ

[BadPatch, Sec. 4.1 và 4.5](https://arxiv.org/html/2412.01440v5) tạo patch để làm giảm khả năng phát hiện người, đánh giá ASR/AP trên các detector đã pretrain và xây dựng AdvT-shirt-1K 1,131 ảnh. AP/ASR tấn công này không cùng mục tiêu với việc tăng AP của mô hình fine-tune. [Adversarial T-shirt, ECCV2020](https://arxiv.org/abs/1910.11099) đặt nền tảng cho thử nghiệm patch trên áo có biến dạng khi người di chuyển. Hai nguồn này giúp mô tả threat model; không xác nhận rằng kiến trúc YOLOv13 đã chống được mọi patch hoặc gây/khắc phục DoS tính toán.

## Đối chiếu với thực nghiệm của dự án

Dự án dùng 11,123 ảnh (COCO subset 10,000 + mirror AdvT-shirt 1,123), chỉ một lớp person, train/val/test 7,786/1,669/1,668. Chỉ dùng val chọn cấu hình; test để đánh giá cấu hình đã khóa. Các paper detector đánh giá 80 lớp trên val2017 và thường huấn luyện đầy đủ hơn; không thể suy ra 47–48% của dự án bằng hay kém 47–48% trên COCO80 lớp. Thước đo dùng cùng dải IoU vẫn có phân phối dữ liệu và phép gộp lớp khác nhau.

Latency dự án ở RTX5060, PyTorch FP32, 416px, batch1, gồm preprocess/inference/postprocess. Latency công bố ở T4 TensorRT và 640px không dùng để tính hệ số tăng tốc của dự án. FPS suy từ 1,000/latency chỉ là phép quy đổi, không phải phép đo throughput độc lập. Đồ thị cần tách panel công bố và panel thực nghiệm; không chuẩn hóa radar rồi xếp hạng giữa các panel.

Có 1,429 ảnh test dự án xuất phát từ COCOtrain2017 và có thể đã xuất hiện khi pretrain; đây là holdout của fine-tuning, chưa là holdout của toàn bộ vòng đời mô hình. Split theo ảnh chưa loại trừ tương quan người/chuỗi. Pooled AP chủ yếu chịu ảnh COCO vì test Adv chỉ 168 ảnh; cần phân tầng clean/Adv và thử nghiệm patch chưa thấy nếu muốn kết luận về độ bền vật lý.

## Giải trình về ưu thế và tính độc đáo

[YOLOv13, Sec. III-B–D](https://arxiv.org/html/2506.17733v1) dùng hyperedge thích nghi để tổng hợp nhiều vị trí/thang đo; FullPAD phân phối đặc trưng; DS-C3k2 là biến thể C3k2 dùng depthwise-separable convolution. Đây là kiến trúc của Lei et al. Table III: đủ tunnel so với bỏ tunnel tăng AP95 47.1 lên 48.0. Table V: S dùng DS giảm 11.7 xuống 9.0M parameters và 27.1 xuống 20.8G FLOPs, cùng AP95 làm tròn 48.0. Đây là ablation của tác giả gốc; dự án chưa tự cô lập các thành phần. Tiềm năng ngữ cảnh không đồng nghĩa bảo đảm chống patch. Params/FLOPs thấp hơn không bảo đảm latency thấp hơn trên backend đang dùng.

Đóng góp có thể bảo vệ của dự án là quy trình áp dụng và tối ưu cho phát hiện người: dữ liệu có truy vết, lựa chọn trên validation, kiểm tra scheduler/loss/augmentation/sampling, đánh giá phân tầng và chi phí thực tế. Chỉ gọi là cải tiến thực nghiệm khi Agent tối ưu có số liệu kiểm chứng; chỉ quy nguyên nhân cho một thành phần khi có ablation giữ các yếu tố khác cố định. So sánh YOLOv13s với YOLO11n đồng thời thay dung lượng mô hình, nên không cô lập lợi ích của HyperACE. Kết quả một seed tối ưu sau tìm kiếm chưa thay thế bằng chứng năm seed ban đầu.

Một kết quả âm hoặc tốc độ thấp vẫn có giá trị nghiên cứu: nó xác định giới hạn triển khai trong môi trường Windows/SDPA và giúp chọn cấu hình theo yêu cầu ứng dụng. Lập luận thuyết phục cần nói rõ phạm vi đạt được, thay vì tuyên bố vượt trội phổ quát.
