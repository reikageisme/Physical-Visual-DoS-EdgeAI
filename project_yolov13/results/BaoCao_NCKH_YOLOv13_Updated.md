# Tối ưu giao thức chuyển giao và đối sánh khoa học YOLOv13 cho phát hiện người

Tác giả: Mai Quốc Bảo. Báo cáo gửi Hội đồng và giảng viên hướng dẫn. Cập nhật: 17/10/2026.

## Tóm tắt

Nghiên cứu kiểm tra cải thiện giao thức chuyển giao và fine tuning YOLOv13s cho person trên 11.123 ảnh. Sau sàng lọc 6 cấu hình bằng validation với seed 42, cấu hình transfer_tiny_lr được khóa và đối chiếu SGD ban đầu ở ba seed 100, 2026 và 777. Mỗi quỹ đạo gồm sáu epoch, ảnh 416, batch 8; ba đối chứng dùng prefix lịch sử đã audit và ba nhánh tối ưu được huấn luyện mới. Test mAP@0.5:0.95 của đối chứng là 47.63 ± 1.44 [44.05; 51.20]%, của nhánh tối ưu là 59.02 ± 0.07 [58.85; 59.18]%; chênh lệch ghép cặp là 11.39 ± 1.37 [7.98; 14.80] điểm phần trăm. CI ghép cặp nằm trên 0, hỗ trợ cải thiện trong ba seed đã đo.

Lựa chọn triển khai được khóa từ validation seed 42 là nhánh đã train transfer_tiny_lr. Quyết định này tách khỏi đánh giá test; không biến số test thành tiêu chí chọn cấu hình.

Latency trung bình của median từng seed nhánh tối ưu là 16.10 ± 0.17 [15.67; 16.54] ms/frame, PyTorch FP32 batch1 trên RTX5060. Diagnostic checkpoint COCO giữ head80 và lọc person cho validation AP95 57.21%. Audit tensor phát hiện cấu trúc YOLOv13s tái dựng trước khác checkpoint pretrained, cần khai báo khi diễn giải kết quả cũ. Đóng góp là kiểm tra chuyển giao, thích ứng giao thức và đánh giá có đối chứng; chưa là kiến trúc mới, tối ưu toàn cục, SOTA hoặc phòng thủ mọi patch. CI phản ánh biến thiên seed trên một split cố định; test đã được xem trong nghiên cứu trước.

Đối chiếu bổ sung sau khóa với checkpoint native không fine tune, giữ head 80 lớp và lọc person, cho test AP95 58.63% tại một điểm cố định, không CI đa seed. Vì vậy cải thiện so với SGD cũ không tự chứng minh fine tuning tốt hơn checkpoint pretrained. Các nguồn COCO trong test có thể từng nằm trong pretraining.

Từ khóa: YOLOv13; chuyển giao; HyperACE; phát hiện người; đối sánh giao thức; đa seed.

## 1 Đặt vấn đề và câu hỏi nghiên cứu

Nghiên cứu trước đạt test AP95 47,97% ở cấu hình được gọi YOLOv13s và sweet spot 10 phút theo checkpoint validation tốt nhất. Giá trị này chưa giải thích chất lượng chuyển giao hay khả năng chống patch. Cập nhật kiểm tra tensor, scheduler, loss, augmentation và sampling, rồi đo hiệu quả và chi phí bằng đối chứng mới.

Câu hỏi chính là cấu hình chọn chỉ bằng validation có cải thiện test AP so với SGD trên cùng dữ liệu và sáu epoch hay không. Câu hỏi bổ sung là cải thiện thuộc nguồn COCO hay AdvT-shirt, và chi phí phù hợp triển khai không. Nếu nhánh sửa cấu trúc pretrained thắng, phép so sánh đo tác dụng gộp sửa chuyển giao và recipe; chưa phân biệt nhân quả từng yếu tố.

## 2 Nghiên cứu liên quan và khác biệt giao thức

Rà soát có chủ đích các nguồn gốc liên quan, chưa là tổng quan hệ thống toàn bộ SOTA. YOLO11 [3] được dẫn như phần mềm chính thức; YOLOv12 [2], YOLOv13 [1], YOLOv10 [4], RT-DETR [5] và RT-DETRv2 [6] là các công trình kiến trúc. Bảng I giữ riêng số công bố. NR nghĩa là chưa xác định trong nguồn được kiểm tra.

Bảng I Số liệu công bố COCO80 lớp 640px

| Mô hình nguồn | AP50 % | AP95 % | Params M | FLOPs G | Tốc độ T4 | Backend dtype |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO11n [3] | NR | 39.5 | 2.6 | 6.5 | 1.5 ms | TensorRT10 / dtype NR |
| YOLO11s [3] | NR | 47 | 9.4 | 21.6 | 2.5 ms | TensorRT10 / dtype NR |
| YOLOv12-N [2] | 56.7 | 40.6 | 2.6 | 6.5 | 1.64 ms | TensorRT / FP16 |
| YOLOv12-S [2] | 65 | 48 | 9.3 | 21.4 | 2.61 ms | TensorRT / FP16 |
| YOLOv13-N [1] | 57.8 | 41.6 | 2.5 | 6.4 | 1.97 ms | TensorRT / FP16 |
| YOLOv13-S [1] | 65.2 | 48 | 9 | 20.8 | 2.98 ms | TensorRT / FP16 |
| YOLOv10-N [4] | 53.8 | 38.5 | 2.3 | 6.7 | 1.84 ms | TensorRT / FP16 |
| YOLOv10-S [4] | 63 | 46.3 | 7.2 | 21.6 | 2.49 ms | TensorRT / FP16 |
| RT-DETR-R18 [5] | 63.8 | 46.5 | 20 | 60 | 217 FPS | TensorRT / FP16 |
| RT-DETR-R50 [5] | 71.3 | 53.1 | 42 | 136 | 108 FPS | TensorRT / FP16 |
| RT-DETRv2-S [6] | 64.9 | 47.9 | 20 | NR | 217 FPS | TensorRT / FP16 |

Bảng I dùng COCO2017 val2017 gộp 80 lớp. GPU tốc độ là T4 nhưng backend, dtype và pipeline không đồng nhất; YOLO11 chỉ ghi TensorRT10, dtype chưa xác định. FPS từ 1000/latency là quy đổi, chưa là phép đo throughput. YOLOv12 và RT-DETRv2 khóa paper v1; không thay bằng Turbo hay checkpoint cập nhật. Baseline tác giả YOLOv13 tự tái lập có AP khác tài liệu chính thức, không ghép nguồn để suy mức cải thiện.

Bảng II Ma trận khác biệt giao thức

| Thiết lập | Dữ liệu lớp | Accuracy | Latency | Đối sánh |
| --- | --- | --- | --- | --- |
| Paper detector | COCO2017; 80 lớp; val2017 | 640px; AP điểm công bố | T4 TensorRT; dtype tùy nguồn | Bối cảnh; không xếp hạng trực tiếp địa phương |
| Báo cáo trước và đo lại | COCO 10.000 + Adv 1.123; person; test 1.668 | 416px; val chọn; 5 seed tại mốc riêng | Loại speed cũ và cohort retiming khác bối cảnh | Giữ accuracy và params; không xếp hạng speed n=5 |
| Hai nhánh cập nhật | Giữ split; person; test từng xem | 416px; 6 epoch; 3 seed ghép | 32 warmup + 500 frame; pipeline chung | Trực tiếp hai nhánh mới; thăm dò |

BadPatch [7] và Adversarial T-shirt [8] cung cấp bối cảnh patch vật lý làm người bị bỏ sót. AP hoặc attack success rate trong bài tấn công khác mục tiêu tăng AP detector. Ảnh patch trong tập dữ liệu chưa chứng minh mô hình chống patch chưa thấy hay giảm DoS tính toán; các kết luận đó cần threat model và thí nghiệm riêng.

## 3 Phương pháp và phạm vi đóng góp

### 3 1 Cơ sở HyperACE FullPAD và DS C3k2

HyperACE của Lei và cộng sự dùng hyperedge thích nghi để tổng hợp quan hệ nhiều vị trí và thang đo; FullPAD phân phối đặc trưng tới nhiều tầng; DS-C3k2 dùng depthwise separable convolution để giảm phép tính và tham số [1]. Cơ chế tạo tiềm năng linh hoạt về ngữ cảnh, chưa là bằng chứng chống patch. Những kiến trúc và tên thành phần thuộc tác giả YOLOv13.

Ablation gốc YOLOv13-S đạt AP95 47,1 khi bỏ tunnel và 48,0 khi đủ tunnel; tác giả mô tả bỏ tunnel làm mất tác dụng HyperACE. Đây là tác dụng gộp, chưa là factorial ablation từng phần. Table V gốc cho DS giảm params 11,7 xuống 9,0M và FLOPs 27,1 xuống 20,8G với AP95 cùng làm tròn 48,0 [1]. FLOPs không đo bộ nhớ và kernel; ít FLOPs không bảo đảm nhanh hơn trên Windows PyTorch.

### 3 2 Can thiệp vào chuyển giao và recipe

Ứng viên kiểm tra lr thấp, cosine scheduler, AdamW, trọng số box class DFL, mosaic mixup biến đổi hình học và sampling nguồn train. Các yếu tố gộp chưa cho attribution riêng. Ứng viên khôi phục pretrained sửa HyperACE về số hyperedge thực và chuyển hàng person vào classifier một lớp, cùng AdamW cosine; đây là sửa chuyển giao và recipe, không phải phát minh kiến trúc.

Bảng III Cấu hình trước sàng lọc

| Cấu hình | Optimizer scheduler | Can thiệp bổ sung |
| --- | --- | --- |
| original_sgd | SGD; lr0=0.01; lrf=0.01; linear; horizon=300; warmup=3.0 | Giao thức gốc |
| adamw_transfer_restored | AdamW; lr0=0.0003; lrf=0.1; cosine; horizon=6; warmup=1.0 | warmup_bias_lr=0.0003; close_mosaic=2 |
| transfer_tiny_lr | AdamW; lr0=1e-05; lrf=0.1; cosine; horizon=6; warmup=1.0 | warmup_bias_lr=1e-05; close_mosaic=2; mosaic=0.25; hsv_s=0.4; hsv_v=0.2; translate=0.05; scale=0.2; freeze=9 |
| low_sgd_cosine | SGD; lr0=0.001; lrf=0.1; cosine; horizon=6; warmup=1.0 | warmup_bias_lr=0.001; close_mosaic=2 |
| adamw_cosine | AdamW; lr0=0.0003; lrf=0.1; cosine; horizon=6; warmup=1.0 | warmup_bias_lr=0.0003; close_mosaic=2 |
| adamw_domain_loss_aug | AdamW; lr0=0.0003; lrf=0.1; cosine; horizon=6; warmup=1.0 | warmup_bias_lr=0.0003; close_mosaic=2; box=9.0; cls=0.7; dfl=1.8; mosaic=0.5; mixup=0.05; hsv_s=0.5; hsv_v=0.3; scale=0.3; degrees=5.0; translate=0.1 |

Đối chứng giữ horizon 300 nhưng dừng sau 6 epoch; cosine dùng horizon 6. Khác biệt là một can thiệp có chủ đích. Riêng ứng viên adamw_domain_loss_aug dùng sampling có hoàn lại, kỳ vọng 25% lượt ảnh chính thuộc Adv thay khoảng 10,10% train gốc; ảnh phụ mosaic vẫn lấy đều. Sáu epoch có 46.716 lượt chọn ảnh chính mỗi run; không bảo đảm mỗi ảnh xuất hiện sáu lần khi lấy mẫu có hoàn lại. Amendment sửa chuyển giao được khóa trước khi GPU đánh giá ứng viên và khai báo trong protocol.

Tìm kiếm có điều chỉnh dựa trên validation: bốn cấu hình ban đầu, ứng viên khôi phục chuyển giao sau audit CPU và một ứng viên thứ sáu theo điều kiện đã ghi trước khi đo ứng viên đó. Ứng viên transfer_tiny_lr dùng AdamW 1e-5, đóng băng chín module đầu, augmentation nhẹ và không đóng băng BN. Đây là thử nghiệm thích nghi hữu hạn, chưa là đăng ký trước bất biến từ đầu; không quy tác dụng cho một thành phần riêng.

## 4 Thiết lập và kiểm soát chất lượng

Dữ liệu 11.123 ảnh gồm COCO 10.000 ảnh chọn xác định và mirror Adv 1.123 ảnh khả dụng. Split theo nguồn và ảnh với seed 2026, tỷ lệ 70/15/15: train 7.786, val 1.669, test 1.668; COCO/Adv lần lượt 7.000/786, 1.500/169, 1.500/168. Chỉ lớp person. Byte trùng qua split bằng 0 nhưng chưa loại tương quan người hoặc chuỗi. Có 1.429 ảnh test từ COCO train2017 có thể đã xuất hiện trong pretraining; holdout chỉ độc lập với fine tuning.

Windows 11, RTX 5060 khoảng 7,96 GiB VRAM, Intel family 6 model 183 với 20 logical CPU, RAM 31,79 GiB. Python chỉ dùng project_yolov13/.venv: Python 3.14.7, Torch 2.11 CUDA 12.8, Ultralytics địa phương 8.3.63. Attention dùng SDPA fallback. Ảnh 416, batch 8, cùng train, sáu epoch; training_config lưu args đầy đủ. Không thay số đo bằng tốc độ TensorRT của bài báo.

Sàng lọc seed 42 có 6 trial chỉ dùng val. Bước 1 chọn epoch có AP95 native lớn nhất trong epoch 1 đến 6, validator AMP FP16 batch16. Bước 2 chỉ đánh giá lại checkpoint đã chọn bằng FP32 batch8 trên toàn validation và xếp hạng ứng viên theo AP95 này. Không đo FP32 mọi epoch, nên có thể bỏ sót checkpoint tối ưu theo FP32. transfer_tiny_lr thắng nhóm cải tiến, khóa tại 2026-10-17T06:15:11+00:00; SGD đối chứng không thuộc nhóm thắng. Các nhánh seed 100, 2026, 777 dùng cùng quy tắc chọn checkpoint rồi đánh giá test sau khóa. Test đã được xem trước nên đây là xác nhận thăm dò, chưa là holdout hoàn toàn mới.

Ba đối chứng SGD dùng đúng sáu epoch đầu của các lượt lịch sử seed100, 2026, 777. Audit seed42 cho thấy toàn bộ giá trị CSV ngoài thời gian và tensor EMA mỗi checkpoint sáu epoch trùng chính xác giữa lần chạy mới và cũ; args có ý nghĩa, SHA nguồn, pretrained và manifest cũng khớp. Chỉ chọn checkpoint trong prefix sáu epoch, không lấy best120 phút. Đối chứng được đo lại test và latency sau khóa trong phiên mới. Chi phí train lịch sử được ghi riêng, chi phí train tăng thêm bằng 0 cho ba prefix tái dùng; không mô tả thành sáu lượt xác nhận đều huấn luyện mới.

Kết quả trình bày mean ± sample SD và CI 95% Student t trong ngoặc vuông, n=3 và df=2. Δ ghép cùng seed. CI theo optimizer trên một split, không đo bất định dataset, người hoặc quá trình tìm cấu hình. AP95 theo IoU 0,50:0,05:0,95; Precision và Recall theo operating point validator. AP gộp 80 lớp của paper khác AP person.

Latency mới dùng protocol actual-batch1-txt-loader-v2: danh sách ảnh lưu thành tệp text để LoadImagesAndVideos đọc từng ảnh, thay vì truyền list Python. Callback và forward hook kiểm tra batch thực bằng 1 cho mọi lần dự đoán. Có 532 ảnh test chọn xác định: 32 warmup và 500 frame lấy mẫu, FP32, 416px; median tổng preprocess, inference, postprocess từng seed rồi tổng hợp. Không gồm decode và I/O; P95 lưu nhưng FPS từ median chưa bảo đảm thời gian thực ứng dụng. Params dựa số phần tử tensor; FLOPs từ THOP MACs nhân hai và extrapolation diện tích có thể thiếu custom operators.

Đính chính phép đo tốc độ: loader nhận list Python trước đây gom 532 ảnh thành một batch lớn dù tham số batch=1, nên mọi latency và FPS lịch sử theo cách đó bị đánh dấu vô hiệu và loại khỏi bảng, đồ thị hoặc kết luận tốc độ cập nhật. Loader cũ còn gây thiếu VRAM ở lần đánh giá đối chứng seed100. Accuracy dùng validator riêng; các số AP đã lưu không bị lỗi loader này. Full test đã tính nhưng chưa lưu trước lần OOM được đánh giá lại, không xem là một record hoàn tất. Mã sửa lưu accuracy trước đo tài nguyên. Speed hiện tại và diagnostic được đo lại với batch thực được kiểm tra; hồ sơ cũ giữ để truy vết, không thay bằng số paper.

Sau khi sửa batch, cùng checkpoint SGD YOLOv13s seed100 và cùng hash có median 18,973 ms trong phiên xác nhận ghép cặp, nhưng 32,142 ms khi đo lại lịch sử độc lập, lệch khoảng 69,4%. Đây là bất ổn theo bối cảnh runtime chưa được giải thích, không phải cải thiện do kiến trúc hay recipe. Vì vậy toàn bộ latency/FPS cohort lịch sử n=5, kể cả lần đo lại batch1 đúng, bị loại khỏi đối sánh tốc độ trực tiếp; chỉ giữ accuracy và params của chúng. Phân tích speed dùng hai nhánh n=3 trong cohort xác nhận chung; điểm native n=1 chỉ là diagnostic. CI theo seed chưa đo hết sai số giữa các phiên hoặc điều kiện hệ thống.

## 5 Kết quả

### 5 1 Chẩn đoán và đính chính cấu trúc

Pretrained head 80 lớp lọc person đúng tại NMS có validation P 84.15%, R 67.60%, AP50 78.41%, AP95 57.21%. Đây là val, không trực tiếp so với test 47,97% trước. Diagnostic 28,54% ban đầu gộp 80 lớp qua single_cls và lọc person sai; record giữ trạng thái invalid_excluded, không dùng làm baseline.

Audit CPU cho YOLOv13s: pretrained prototype_base thực [8,128], tái dựng YAML [16,128], context_net 1024 lên 2048; 6 tensor HyperACE và 6 classifier khác shape khiến khoảng 1.057.155 phần tử tensor mới không chuyển được. Bản khôi phục 8 edge có 9,024954M params, theo head_transfer_audit.json. Số cũ 9,553338M thuộc biến thể tái dựng thực đo, không gắn trực tiếp checkpoint chính thức. Nano có 4 edge thực và 4 edge tái dựng, không có lỗi HyperACE tương tự; hidden width classifier từ 80 xuống 64 còn khác. Không tổng quát hóa anomaly cho mọi YOLOv13.

Khôi phục cấu trúc và chuyển hàng person chưa tự xác định nguyên nhân AP thấp. Scheduler, learning rate, head và forgetting cùng ảnh hưởng; cần ablation để quy nguyên nhân. Báo cáo trước và số đo được giữ nguyên, đính chính phạm vi tên mô hình thực chạy.

Khởi tạo một lớp đã khôi phục 8 hyperedge và chuyển hàng person, chưa học epoch nào, đạt validation AP50 78.41% và AP95 57.21%. Epoch 0 không tham gia chọn ứng viên đã train. Theo amendment lựa chọn triển khai riêng trước test, checkpoint này được đối chiếu FP32 batch8 với ứng viên đã train thắng sàng lọc; n=1 và không gán CI seed.

Quy tắc triển khai nghiêm ngặt chọn AP95 validation lớn nhất đã ưu tiên nhánh train: 57.7000% so với khôi phục epoch 0 57.2068%, tăng 0.4932 pp sau 26.02 phút. Xét độ nhạy Pareto hậu nghiệm với dung sai thực dụng 0,5 pp đã dùng trong nghiên cứu thời gian trước, lợi ích nhỏ có thể làm phương án không train hấp dẫn hơn về chi phí. Phân tích độ nhạy này không thay quyết định khóa và không là tiêu chí chọn ban đầu; cần test mới để kiểm định giá trị ứng dụng.

Kiểm tra cùng FP32 batch8 cho checkpoint adamw_transfer_restored đã train có AP95 50.0004%; thay riêng buffer BN bằng giá trị khởi tạo làm AP95 còn 0.0449%. Audit xác nhận tensor ngoài BN giữ nguyên. Chênh lệch này không hỗ trợ thao tác đặt lại BN hậu huấn luyện cho checkpoint này, nên ứng viên tiny LR không đóng băng BN theo điều kiện đã khóa. Không suy ra nhân quả toàn quá trình học hoặc mọi mô hình từ một phép hoán đổi buffer. Đánh giá khởi tạo và đã train ở cả FP16 batch16 cũng cho AP95 giảm; sai khác precision đơn thuần không giải thích toàn bộ suy giảm. Mất tri thức pretrained vẫn là diễn giải cần ablation, không phải nguyên nhân đã xác lập.

Bảng IV Sàng lọc validation seed42 theo hai bước

| Cấu hình | Epoch chọn | AP95 native % | P FP32 % | R FP32 % | AP50 FP32 % | AP95 FP32 % | Train phút |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_sgd | 1 | 47.03 | 79.12 | 59.62 | 69.87 | 47.07 | 24.40 |
| adamw_transfer_restored | 6 | 49.97 | 81.04 | 61.74 | 72.47 | 50.00 | 20.90 |
| transfer_tiny_lr | 6 | 57.67 | 84.76 | 68.94 | 79.46 | 57.70 | 26.02 |
| low_sgd_cosine | 6 | 55.56 | 83.79 | 67.65 | 77.94 | 55.58 | 21.83 |
| adamw_cosine | 6 | 49.53 | 79.57 | 61.99 | 71.97 | 49.55 | 23.40 |
| adamw_domain_loss_aug | 6 | 47.75 | 80.90 | 61.37 | 72.10 | 47.76 | 29.38 |

Sàng lọc một seed không có CI đa seed. Epoch chọn có thể sớm hơn 6 vì lấy validation best. Cột native trích CSV train; FP32 đo lại checkpoint đó, là cơ sở xếp hạng cuối. Cấu hình thắng chỉ tốt nhất trong các ứng viên đã chạy, chưa là tối ưu toàn cục.

Bảng IVa Hội tụ và loss cuối epoch6

| Cấu hình | Seed | Giai đoạn | Train box cls dfl | Val box cls dfl | AP95 native epoch1 → 6 % | Phút tới epoch chọn |
| --- | --- | --- | --- | --- | --- | --- |
| original_sgd | 42 | screening | 1.503/1.433/1.492 | 1.476/1.345/1.452 | 47.03 → 29.51 | 7.47 |
| adamw_transfer_restored | 42 | screening | 0.992/0.734/1.122 | 1.022/0.748/1.097 | 42.83 → 49.97 | 20.90 |
| transfer_tiny_lr | 42 | screening | 0.866/0.579/1.023 | 0.875/0.553/0.991 | 57.35 → 57.67 | 26.02 |
| low_sgd_cosine | 42 | screening | 0.886/0.601/1.033 | 0.924/0.613/1.009 | 54.09 → 55.56 | 21.83 |
| adamw_cosine | 42 | screening | 0.994/0.745/1.128 | 1.022/0.753/1.104 | 41.84 → 49.53 | 23.40 |
| adamw_domain_loss_aug | 42 | screening | 1.156/0.935/1.338 | 1.351/1.073/1.377 | 38.08 → 47.75 | 29.38 |
| original_sgd | 100 | confirmation | 1.488/1.428/1.486 | 1.484/1.328/1.433 | 45.96 → 29.57 | 4.52 |
| transfer_tiny_lr | 100 | confirmation | 0.871/0.577/1.023 | 0.873/0.551/0.989 | 57.37 → 57.68 | 18.49 |
| original_sgd | 2026 | confirmation | 1.488/1.435/1.479 | 1.470/1.353/1.427 | 47.39 → 29.14 | 5.63 |
| transfer_tiny_lr | 2026 | confirmation | 0.869/0.574/1.021 | 0.877/0.550/0.989 | 57.36 → 57.62 | 12.27 |
| original_sgd | 777 | confirmation | 1.492/1.434/1.481 | 1.493/1.331/1.434 | 44.51 → 29.43 | 4.50 |
| transfer_tiny_lr | 777 | confirmation | 0.865/0.578/1.021 | 0.871/0.550/0.991 | 57.28 → 57.76 | 17.07 |

Loss là giá trị native cuối quỹ đạo, không phải loss của checkpoint chọn nếu epoch tốt nhất sớm hơn sáu. Train dùng augmentation, val không dùng cùng phép biến đổi, nên khoảng cách loss chưa tự chứng minh overfitting. AP giảm trong khi loss train giảm có thể gợi ý suy giảm chuyển giao hoặc không khớp mục tiêu; sáu epoch và một split chưa phân biệt chắc chắn overfitting, underfitting và forgetting, cũng chưa chứng minh trần hội tụ.

### 5 2 Đối chứng mới và ba seed

Bảng V Accuracy test sau khóa

| Nhánh | n | Precision % | Recall % | AP50 % | AP95 % |
| --- | --- | --- | --- | --- | --- |
| Đối chứng SGD | 3 | 80.18 ± 1.45 [76.59; 83.77] | 61.03 ± 1.07 [58.38; 63.67] | 70.61 ± 1.31 [67.35; 73.86] | 47.63 ± 1.44 [44.05; 51.20] |
| Nhánh tối ưu | 3 | 85.78 ± 0.23 [85.21; 86.34] | 69.67 ± 0.17 [69.24; 70.11] | 79.71 ± 0.06 [79.56; 79.87] | 59.02 ± 0.07 [58.85; 59.18] |
| Native pretrained | 1 | 85.91 | 68.07 | 78.76 | 58.63 |

Bảng VI Tài nguyên và chi phí

| Nhánh | Params M | FLOPs G | Train phút | Latency ms | FPS |
| --- | --- | --- | --- | --- | --- |
| Đối chứng SGD | 9.553 ± 0.000 | 9.15 ± 0.00 | 22.45 ± 0.62 [20.90; 24.00] | 16.52 ± 2.19 [11.09; 21.95] | 61.21 ± 7.69 [42.12; 80.31] |
| Nhánh tối ưu | 9.025 ± 0.000 | 8.87 ± 0.00 | 17.77 ± 0.71 [16.01; 19.52] | 16.10 ± 0.17 [15.67; 16.54] | 62.10 ± 0.67 [60.43; 63.77] |
| Native pretrained | 9.056 | 8.95 | 0 fine tune | 16.64 | 60.09 |

Hàng native pretrained là một checkpoint cố định head 80 lớp, lọc COCO person trong NMS đúng protocol; n=1 biểu thị một điểm chẩn đoán, không nhóm seed và không gán CI. Đo test và latency sau winner lock. Params/FLOPs khác nhánh một lớp; latency mới cùng batch1 FP32 nhưng dự đoán được lọc person. Chi phí pretraining từ nguồn ngoài không bằng 0; số 0 chỉ là fine tuning bổ sung của dự án.

Δ tối ưu trừ đối chứng AP95 là 11.39 ± 1.37 [7.98; 14.80] pp. CI ghép cặp nằm trên 0, hỗ trợ cải thiện trong ba seed đã đo. Nếu nhánh khôi phục cấu trúc thắng, Δ phản ánh gói can thiệp, chưa tách cấu trúc và recipe. Tổng runtime CSV của các trial được phân tích là 4.44 giờ, chưa cộng khởi tạo và đánh giá; nếu có tái dùng prefix đối chứng lịch sử, tổng này không phải toàn bộ chi phí tính toán mới của vòng cập nhật.

Chi phí train CSV của 9 quỹ đạo mới là 3.32 giờ; ba prefix đối chứng có thời gian lịch sử cộng 1.12 giờ nhưng không train mới. CSV không tính toàn bộ tải dữ liệu, khởi tạo, đánh giá, diagnostic, thời gian chờ và thời gian gián đoạn phiên làm việc. Ngân sách sáu epoch tương đương lượt dữ liệu, chưa tương đương thời gian đồng hồ hoặc dung lượng mô hình.

Bảng VII Test ghép cặp từng seed

| Seed | Epoch đối chứng | Epoch tối ưu | AP95 đối chứng % | AP95 tối ưu % | Δ pp |
| --- | --- | --- | --- | --- | --- |
| 100 | 1 | 6 | 47.60 | 59.03 | +11.43 |
| 2026 | 1 | 4 | 49.08 | 59.08 | +9.99 |
| 777 | 1 | 6 | 46.20 | 58.94 | +12.74 |

### 5 3 Phân tầng nguồn

Bảng VIII Test theo nguồn với ba seed

| Nhánh | Nguồn | Ảnh | Person | Ảnh âm | AP50 % | AP95 % |
| --- | --- | --- | --- | --- | --- | --- |
| Đối chứng | coco | 1500 | 4788 | 300 | 68.67 ± 1.47 [65.02; 72.32] | 44.93 ± 1.65 [40.83; 49.03] |
| Đối chứng | advtshirt | 168 | 667 | 0 | 84.30 ± 0.40 [83.32; 85.28] | 65.82 ± 0.20 [65.32; 66.33] |
| Tối ưu | coco | 1500 | 4788 | 300 | 78.28 ± 0.08 [78.10; 78.47] | 56.78 ± 0.05 [56.65; 56.91] |
| Tối ưu | advtshirt | 168 | 667 | 0 | 89.88 ± 0.06 [89.75; 90.02] | 75.06 ± 0.22 [74.51; 75.61] |

Pooled AP không bằng mean AP hai nguồn. Test COCO 1.500 ảnh so với Adv 168 nên pooled có thể che miền patch. Phân tầng giúp định vị cải thiện, chưa là causal defense: thiếu cặp cùng cảnh clean và attacked, patch mới và split theo người hoặc chuỗi.

### 5 4 Đối chiếu các mốc thời gian đã khảo sát

Bảng IX Đường cong thời gian thăm dò trước với validation AP95 phần trăm

| Mô hình | 5 phút | 10 phút | 20 phút | 30 phút | 60 phút | 120 phút | Sweet spot phút |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO11n | 36.7372 | 36.7372 | 36.7372 | 36.7372 | 36.8510 | 40.8488 | 120 |
| YOLOv12n | 23.7272 | 24.3246 | 25.2424 | 28.5548 | 33.8660 | 38.6968 | 120 |
| YOLOv13n | 39.0300 | 39.0300 | 39.0300 | 39.0300 | 39.0300 | 39.3978 | 5 |
| YOLOv13s | NR | 46.2738 | 46.2738 | 46.2738 | 46.2738 | 46.2738 | 10 |

Nguồn Bảng IX là trường sweet_spots trong results/fullscale_experiments/latency_records.json, chỉ dùng metadata accuracy, loại toàn bộ speed cũ. Mỗi ô là mean validation AP95 của checkpoint tốt nhất đã hoàn tất trong ngân sách với đủ năm seed. NR ở YOLOv13s 5 phút vì seed2026 chưa hoàn tất epoch đầu, nên thiếu nhóm đủ năm seed. Sweet spot là mốc sớm nhất có mean cách peak quan sát không quá 0,5 pp. Đây là các đường cong thăm dò trước, không phải lượt chạy mới 1, 2, 4 hoặc 8 giờ.

Checkpoint best của YOLOv13s giữ nguyên từ 10 đến 120 phút; YOLOv13n tăng thêm 0.3678 pp ở 120 so với 5 phút, nằm trong dung sai 0,5 pp. YOLO11n và YOLOv12n còn tăng tới 120 phút, nên chưa chứng minh đã hội tụ. Đường best-so-far phẳng có thể chỉ do giữ một checkpoint sớm; không đồng nghĩa loss hoặc mô hình ngừng học. 120 phút là mốc tối đa đã thử, chưa có bằng chứng cho kết luận phổ quát 4 hoặc 8 giờ. Cấu trúc tái dựng và recipe của đường cong trước khác nhánh khôi phục hiện tại, vì vậy không chuyển nguyên sweet spot cũ sang mô hình sửa.

Khuyến nghị khảo sát bắt đầu bằng diagnostic epoch0, các mốc sớm 5, 10, 20 và 30 phút, rồi mở tới 60 hoặc 120 phút khi validation còn tăng vượt dung sai. Với ứng viên tiny LR hiện tại, sáu epoch và diagnostic epoch0 là bằng chứng trực tiếp; cần lập đường cong ngân sách riêng sau sửa chuyển giao trước khi đề xuất mốc dừng cho nghiên cứu tiếp.

## 6 Hình ảnh và đánh đổi

![Hình 1 Đối sánh AP, FPS và params](figures/comparison_baseline_vs_proposed.png)

Hình 1 Đối sánh AP, FPS và params. Accuracy và params lịch sử n=5 được giữ; speed cũ và cohort retiming lệch bối cảnh bị loại. Hai nhánh mới n=3 có CI 95% theo seed; speed mới FP32 416px RTX5060. Native80 lọc person là điểm n=1, không SD/CI. Panel COCO80 lớp 640px là bối cảnh khác giao thức. Nguồn: fullscale accuracy, final_summary.json, retiming và benchmark_matrix.json. Ảnh 300 DPI.

![Hình 2 AP95 và latency test person, batch1 thực FP32 416px, không decode/I/O](figures/tradeoff_map_vs_latency.png)

Hình 2 AP95 và latency test person, batch1 thực FP32 416px, không decode/I/O. Chỉ hai nhánh ghép cặp n=3 có CI 95% và diagnostic native80 n=1 không CI xuất hiện. Mọi điểm speed lịch sử n=5 bị loại vì loader cũ và drift cohort. Ngưỡng 33,3 ms chỉ tham khảo 30 FPS, không kết luận tăng tốc. Nguồn: final_summary.json và audit timing. Ảnh 300 DPI.

Nhánh tối ưu có AP95 59.02% và latency đo lại 16.10 ms. Chỉ mô tả chênh lệch speed giữa hai nhánh n=3 trong cùng cohort xác nhận; chưa xác lập tăng tốc do recipe, nhất là khi đã quan sát drift giữa các phiên. Điểm lịch sử n=5 chỉ dùng accuracy và params. Các nhóm khác ngân sách và dung lượng chưa cô lập HyperACE. Cần benchmark lặp theo phiên, kiểm soát trạng thái GPU và export trên thiết bị đích, đo tail p95 cùng decode I/O trước cam kết 30 FPS.

## 7 Ưu nhược điểm và giải trình

Sáu chủ đề giải trình dưới đây do nghiên cứu phân rã từ góp ý tổng quát về AP thấp, tính độc đáo và đối sánh. Chúng không phải sáu câu hỏi nguyên văn hoặc nhận xét cụ thể đã được cung cấp bởi Hội đồng.

Bảng X Ưu nhược điểm và mức chứng cứ

| Thành phần | Ưu điểm | Hạn chế | Chứng cứ |
| --- | --- | --- | --- |
| HyperACE FullPAD | Tương quan đa vị trí đa thang | Backend và dataset ảnh hưởng; chưa chứng minh chống patch | Cơ chế và ablation gốc [1]; chưa ablation địa phương |
| DS C3k2 | Giảm params và FLOPs | Không bảo đảm latency | Table V gốc [1]; pipeline đo local |
| Khôi phục pretrained | Giữ shape tương thích | Gộp cùng recipe; chưa causal | Audit tensor và trial mới |
| Scheduler optimizer | Thích ứng horizon | 6 khác 300; chưa tối ưu toàn cục | Cùng 6 epoch, seed ghép cặp |
| Loss aug sampling | Ưu tiên miền Adv và điều chỉnh loss | Gộp yếu tố; lặp cảnh có thể overfit | Chỉ toàn cấu hình; cần ablation tiếp |
| Ba seed | Đo biến thiên optimizer | CI rộng, chưa đo bất định dataset | Thăm dò n=3 |
| COCO Adv | Cảnh thường và áo patch | Pretrain exposure, tương quan frame | Phân tầng, chưa causal defense |

### 7 1 Vì sao kết quả trước thấp

Diagnostic person đúng cho AP cao hơn, còn head và shape HyperACE khác khi tái dựng; lịch học và forgetting cũng ảnh hưởng. Giữ số trước và khai báo erratum; chưa quy một nguyên nhân khi chưa có ablation.

### 7 2 Đóng góp độc đáo

Kiến trúc thuộc YOLOv13 gốc. Đóng góp dự án là audit chuyển giao, recipe person, khóa cấu hình bằng val, seed ghép cặp, phân tầng và đo chi phí. Không nhận kiến trúc gốc là phát minh mới.

### 7 3 Tại sao hướng đi tốt hơn

Lý thuyết hỗ trợ ngữ cảnh bậc cao. Bằng chứng gói can thiệp là Δ AP95 11.39 ± 1.37 [7.98; 14.80] pp. CI ghép cặp nằm trên 0, hỗ trợ cải thiện trong ba seed đã đo. Chưa quy riêng HyperACE, head hoặc scheduler.

### 7 4 Có vượt SOTA

Chưa có cơ sở SOTA. COCO80 lớp 640px T4 khác person416 RTX5060. Cần cùng dataset, class mapping, backend và dung lượng mô hình.

### 7 5 Diễn giải FLOPs và tốc độ thế nào

Attention kernels, bộ nhớ, depthwise, graph và postprocess tùy backend; FLOPs chưa đo thời gian. Trước hết phải sửa loader để xác nhận batch thực. Chỉ dùng speed mới đã kiểm tra, loại mọi số lịch sử vô hiệu và benchmark export trên thiết bị đích.

### 7 6 Thực nghiệm tiếp theo

Ablation khôi phục cấu trúc, head và recipe; tách loss aug sampling; thêm baseline small, seed mới, holdout theo người hoặc chuỗi. Dùng patch mới, clean attacked ghép cặp và ASR nếu kết luận robustness; workload tài nguyên và latency đầu cuối nếu kết luận DoS.

## 8 Hạn chế và tái lập

Val và test tái dùng gây selection bias; khóa mới không tạo test hoàn toàn mới. CI ba seed trên split cố định chưa đo generalization quần thể. Exposure COCO và ảnh tương quan; mirror 1.123 khác 1.131 ảnh công bố. Sáu epoch chưa chứng minh saturation; cosine lịch 6 so với đối chứng lịch 300 dừng sớm. Gói can thiệp cấu trúc, initialization và recipe cần ablation. Lựa chọn hai bước chỉ xếp hạng checkpoint native tốt nhất, chưa tìm tối ưu mọi epoch theo FP32. Ứng viên thứ sáu và đối chiếu epoch0 có thích nghi từ validation; đối chứng dùng prefix lịch sử đã audit, không hoàn toàn chạy đồng thời với nhánh mới.

Phiên thực nghiệm bị gián đoạn giữa ngày 13 và ngày 17/10/2026. Ứng viên tiny LR được chạy lại thành quỹ đạo sáu epoch hoàn chỉnh; hai epoch đầu so với prefix gián đoạn trùng mọi trường CSV ngoài thời gian theo interruption_replay_audit.json. Prefix gián đoạn không tham gia xếp hạng hoặc gộp thêm như seed mới. Hai epoch hoàn tất của lượt bỏ dở dùng ít nhất 7.51 phút, chưa gồm phần epoch 3 đang chạy và overhead. Runtime các quỹ đạo chọn là phần chi phí xác định được, chưa bao gồm toàn bộ tính toán, diagnostic và đánh giá.

Checkpoint và log cục bộ, protocol.json, winner_lock.json, final_summary.json, audit tensor và diagnostic invalid được giữ. Hash manifest, pretrained và đầu vào hình cho phép truy vết. Số liệu cũ giữ nguyên. Git chứa artifact compact, không gồm ảnh dataset hoặc checkpoint lớn.

## 9 Kết luận

transfer_tiny_lr tốt nhất trong các ứng viên đã train được sàng lọc. Test AP95 59.02% so với SGD 47.63%; Δ 11.39 ± 1.37 [7.98; 14.80] pp. CI ghép cặp nằm trên 0, hỗ trợ cải thiện trong ba seed đã đo. Dùng làm điểm khởi đầu vòng xác nhận, lấy split mới và benchmark edge. Kết luận giới hạn trong thiết lập đã đo.

Theo quy tắc validation AP95 lớn nhất, lựa chọn triển khai đã khóa là checkpoint đã train của transfer_tiny_lr. Điểm epoch0 nếu được đo test chỉ là đối chiếu chẩn đoán n=1, chưa thay quyết định. Cần xác nhận lựa chọn này trên holdout mới, có phân tầng patch và tốc độ đầu cuối ở thiết bị đích; không suy ra thời gian tối ưu phổ quát từ sáu epoch.

So với điểm chẩn đoán native pretrained AP95 58.63%, chênh lệch trung bình nhánh tối ưu là +0.39 pp, chưa có kiểm định đa seed tương ứng cho checkpoint native. Khuyến nghị giữ phương án không fine tune như một baseline thực dụng, và chỉ bỏ nó khi dữ liệu mới cùng chi phí thực tế hỗ trợ lựa chọn khác.

## Tài liệu tham khảo

[1] Mengqi Lei, Siqi Li, Yihong Wu, Han Hu, You Zhou, Xinhu Zheng, Guiguang Ding, Shaoyi Du, Zongze Wu, Yue Gao, “YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception,” arXiv preprint, 2025. https://arxiv.org/html/2506.17733v1 (truy xuất 13/10/2026).

[2] Yunjie Tian, Qixiang Ye, David Doermann, “YOLOv12: Attention-Centric Real-Time Object Detectors,” arXiv v1; subsequent NeurIPS 2025 conference version differs, 2025. https://arxiv.org/html/2502.12524v1 (truy xuất 13/10/2026).

[3] Glenn Jocher, Jing Qiu, “Ultralytics YOLO11,” official software documentation; no formal official research paper, 2024. https://docs.ultralytics.com/models/yolo11/ (truy xuất 13/10/2026).

[4] Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding, “YOLOv10: Real-Time End-to-End Object Detection,” NeurIPS 2024; metrics pinned to arXiv v1 and official repository, 2024. https://arxiv.org/html/2405.14458v1 (truy xuất 13/10/2026).

[5] Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen, “DETRs Beat YOLOs on Real-time Object Detection,” CVPR 2024; official repository metrics, 2024. https://arxiv.org/abs/2304.08069 (truy xuất 13/10/2026).

[6] Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu, “RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer,” arXiv technical report, 2024. https://arxiv.org/html/2407.17140v1 (truy xuất 13/10/2026).

[7] Zhixiang Wang, Xingjun Ma, Yu-Gang Jiang, “BadPatch: Diffusion-Based Generation of Physical Adversarial Patches,” arXiv v5; ICCV 2025 Findings workshop listing, 2025. https://arxiv.org/html/2412.01440v5 (truy xuất 13/10/2026).

[8] Kaidi Xu, Gaoyuan Zhang, Sijia Liu, Quanfu Fan, Mengshu Sun, Hongge Chen, Pin-Yu Chen, Yanzhi Wang, Xue Lin, “Adversarial T-shirt! Evading Person Detectors in A Physical World,” ECCV 2020, 2020. https://arxiv.org/abs/1910.11099 (truy xuất 13/10/2026).

## Phụ lục bằng chứng

Markdown kèm LaTeX booktabs theo thông lệ IEEE. Word là báo cáo một cột có trang ngang cho bảng và hình, chưa là mẫu camera ready IEEE hai cột.

Bằng chứng: results/optimization/protocol.json

Bằng chứng: results/optimization/winner_lock.json

Bằng chứng: results/optimization/deployment_choice_lock.json

Bằng chứng: results/optimization/final_summary.json

Bằng chứng: results/optimization/pretrained_validation.json

Bằng chứng: results/optimization/pretrained_validation_unfiltered_invalid.json

Bằng chứng: results/optimization/head_transfer_audit.json

Bằng chứng: results/optimization/historical_control_reuse_audit.json

Bằng chứng: results/optimization/bn_controlled_diagnostic.json

Bằng chứng: results/optimization/latency_protocol_correction.json

Bằng chứng: results/optimization/historical_latency_retimes.json

Bằng chứng: results/literature_updated/benchmark_matrix.json

Bằng chứng: results/literature_updated/source_evidence.json

Bằng chứng: results/figures/figures_manifest.json

Bằng chứng: results/fullscale_experiments/latency_records.json

Bằng chứng: results/updated_research/metric_integrity_audit.json

## Mã LaTeX bảng IEEE

Dùng gói booktabs. Bảng rộng cần chỉnh nhãn hoặc trang ngang khi tích hợp manuscript.

```latex
\begin{table*}[t]
\centering
\caption{Bảng I Số liệu công bố COCO80 lớp 640px}
\label{tab:updated-1}
\begin{tabular}{lcccccc}
\toprule
Mô hình nguồn & AP50 \% & AP95 \% & Params M & FLOPs G & Tốc độ T4 & Backend dtype \\
\midrule
YOLO11n [3] & NR & 39.5 & 2.6 & 6.5 & 1.5 ms & TensorRT10 / dtype NR \\
YOLO11s [3] & NR & 47 & 9.4 & 21.6 & 2.5 ms & TensorRT10 / dtype NR \\
YOLOv12-N [2] & 56.7 & 40.6 & 2.6 & 6.5 & 1.64 ms & TensorRT / FP16 \\
YOLOv12-S [2] & 65 & 48 & 9.3 & 21.4 & 2.61 ms & TensorRT / FP16 \\
YOLOv13-N [1] & 57.8 & 41.6 & 2.5 & 6.4 & 1.97 ms & TensorRT / FP16 \\
YOLOv13-S [1] & 65.2 & 48 & 9 & 20.8 & 2.98 ms & TensorRT / FP16 \\
YOLOv10-N [4] & 53.8 & 38.5 & 2.3 & 6.7 & 1.84 ms & TensorRT / FP16 \\
YOLOv10-S [4] & 63 & 46.3 & 7.2 & 21.6 & 2.49 ms & TensorRT / FP16 \\
RT-DETR-R18 [5] & 63.8 & 46.5 & 20 & 60 & 217 FPS & TensorRT / FP16 \\
RT-DETR-R50 [5] & 71.3 & 53.1 & 42 & 136 & 108 FPS & TensorRT / FP16 \\
RT-DETRv2-S [6] & 64.9 & 47.9 & 20 & NR & 217 FPS & TensorRT / FP16 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Bảng IV Sàng lọc validation seed42 theo hai bước}
\label{tab:updated-2}
\begin{tabular}{lccccccc}
\toprule
Cấu hình & Epoch chọn & AP95 native \% & P FP32 \% & R FP32 \% & AP50 FP32 \% & AP95 FP32 \% & Train phút \\
\midrule
original\_sgd & 1 & 47.03 & 79.12 & 59.62 & 69.87 & 47.07 & 24.40 \\
adamw\_transfer\_restored & 6 & 49.97 & 81.04 & 61.74 & 72.47 & 50.00 & 20.90 \\
transfer\_tiny\_lr & 6 & 57.67 & 84.76 & 68.94 & 79.46 & 57.70 & 26.02 \\
low\_sgd\_cosine & 6 & 55.56 & 83.79 & 67.65 & 77.94 & 55.58 & 21.83 \\
adamw\_cosine & 6 & 49.53 & 79.57 & 61.99 & 71.97 & 49.55 & 23.40 \\
adamw\_domain\_loss\_aug & 6 & 47.75 & 80.90 & 61.37 & 72.10 & 47.76 & 29.38 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Bảng IVa Hội tụ và loss cuối epoch6}
\label{tab:updated-3}
\begin{tabular}{lcccccc}
\toprule
Cấu hình & Seed & Giai đoạn & Train box cls dfl & Val box cls dfl & AP95 native epoch1 → 6 \% & Phút tới epoch chọn \\
\midrule
original\_sgd & 42 & screening & 1.503/1.433/1.492 & 1.476/1.345/1.452 & 47.03 → 29.51 & 7.47 \\
adamw\_transfer\_restored & 42 & screening & 0.992/0.734/1.122 & 1.022/0.748/1.097 & 42.83 → 49.97 & 20.90 \\
transfer\_tiny\_lr & 42 & screening & 0.866/0.579/1.023 & 0.875/0.553/0.991 & 57.35 → 57.67 & 26.02 \\
low\_sgd\_cosine & 42 & screening & 0.886/0.601/1.033 & 0.924/0.613/1.009 & 54.09 → 55.56 & 21.83 \\
adamw\_cosine & 42 & screening & 0.994/0.745/1.128 & 1.022/0.753/1.104 & 41.84 → 49.53 & 23.40 \\
adamw\_domain\_loss\_aug & 42 & screening & 1.156/0.935/1.338 & 1.351/1.073/1.377 & 38.08 → 47.75 & 29.38 \\
original\_sgd & 100 & confirmation & 1.488/1.428/1.486 & 1.484/1.328/1.433 & 45.96 → 29.57 & 4.52 \\
transfer\_tiny\_lr & 100 & confirmation & 0.871/0.577/1.023 & 0.873/0.551/0.989 & 57.37 → 57.68 & 18.49 \\
original\_sgd & 2026 & confirmation & 1.488/1.435/1.479 & 1.470/1.353/1.427 & 47.39 → 29.14 & 5.63 \\
transfer\_tiny\_lr & 2026 & confirmation & 0.869/0.574/1.021 & 0.877/0.550/0.989 & 57.36 → 57.62 & 12.27 \\
original\_sgd & 777 & confirmation & 1.492/1.434/1.481 & 1.493/1.331/1.434 & 44.51 → 29.43 & 4.50 \\
transfer\_tiny\_lr & 777 & confirmation & 0.865/0.578/1.021 & 0.871/0.550/0.991 & 57.28 → 57.76 & 17.07 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Bảng V Accuracy test sau khóa}
\label{tab:updated-4}
\begin{tabular}{lccccc}
\toprule
Nhánh & n & Precision \% & Recall \% & AP50 \% & AP95 \% \\
\midrule
Đối chứng SGD & 3 & 80.18 $\pm$ 1.45 [76.59; 83.77] & 61.03 $\pm$ 1.07 [58.38; 63.67] & 70.61 $\pm$ 1.31 [67.35; 73.86] & 47.63 $\pm$ 1.44 [44.05; 51.20] \\
Nhánh tối ưu & 3 & 85.78 $\pm$ 0.23 [85.21; 86.34] & 69.67 $\pm$ 0.17 [69.24; 70.11] & 79.71 $\pm$ 0.06 [79.56; 79.87] & 59.02 $\pm$ 0.07 [58.85; 59.18] \\
Native pretrained & 1 & 85.91 & 68.07 & 78.76 & 58.63 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Bảng VI Tài nguyên và chi phí}
\label{tab:updated-5}
\begin{tabular}{lccccc}
\toprule
Nhánh & Params M & FLOPs G & Train phút & Latency ms & FPS \\
\midrule
Đối chứng SGD & 9.553 $\pm$ 0.000 & 9.15 $\pm$ 0.00 & 22.45 $\pm$ 0.62 [20.90; 24.00] & 16.52 $\pm$ 2.19 [11.09; 21.95] & 61.21 $\pm$ 7.69 [42.12; 80.31] \\
Nhánh tối ưu & 9.025 $\pm$ 0.000 & 8.87 $\pm$ 0.00 & 17.77 $\pm$ 0.71 [16.01; 19.52] & 16.10 $\pm$ 0.17 [15.67; 16.54] & 62.10 $\pm$ 0.67 [60.43; 63.77] \\
Native pretrained & 9.056 & 8.95 & 0 fine tune & 16.64 & 60.09 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Bảng VII Test ghép cặp từng seed}
\label{tab:updated-6}
\begin{tabular}{lccccc}
\toprule
Seed & Epoch đối chứng & Epoch tối ưu & AP95 đối chứng \% & AP95 tối ưu \% & $\Delta$ pp \\
\midrule
100 & 1 & 6 & 47.60 & 59.03 & +11.43 \\
2026 & 1 & 4 & 49.08 & 59.08 & +9.99 \\
777 & 1 & 6 & 46.20 & 58.94 & +12.74 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Bảng VIII Test theo nguồn với ba seed}
\label{tab:updated-7}
\begin{tabular}{lcccccc}
\toprule
Nhánh & Nguồn & Ảnh & Person & Ảnh âm & AP50 \% & AP95 \% \\
\midrule
Đối chứng & coco & 1500 & 4788 & 300 & 68.67 $\pm$ 1.47 [65.02; 72.32] & 44.93 $\pm$ 1.65 [40.83; 49.03] \\
Đối chứng & advtshirt & 168 & 667 & 0 & 84.30 $\pm$ 0.40 [83.32; 85.28] & 65.82 $\pm$ 0.20 [65.32; 66.33] \\
Tối ưu & coco & 1500 & 4788 & 300 & 78.28 $\pm$ 0.08 [78.10; 78.47] & 56.78 $\pm$ 0.05 [56.65; 56.91] \\
Tối ưu & advtshirt & 168 & 667 & 0 & 89.88 $\pm$ 0.06 [89.75; 90.02] & 75.06 $\pm$ 0.22 [74.51; 75.61] \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Bảng IX Đường cong thời gian thăm dò trước với validation AP95 phần trăm}
\label{tab:updated-8}
\begin{tabular}{lccccccc}
\toprule
Mô hình & 5 phút & 10 phút & 20 phút & 30 phút & 60 phút & 120 phút & Sweet spot phút \\
\midrule
YOLO11n & 36.7372 & 36.7372 & 36.7372 & 36.7372 & 36.8510 & 40.8488 & 120 \\
YOLOv12n & 23.7272 & 24.3246 & 25.2424 & 28.5548 & 33.8660 & 38.6968 & 120 \\
YOLOv13n & 39.0300 & 39.0300 & 39.0300 & 39.0300 & 39.0300 & 39.3978 & 5 \\
YOLOv13s & NR & 46.2738 & 46.2738 & 46.2738 & 46.2738 & 46.2738 & 10 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Bảng X Ưu nhược điểm và mức chứng cứ}
\label{tab:updated-9}
\begin{tabular}{lccc}
\toprule
Thành phần & Ưu điểm & Hạn chế & Chứng cứ \\
\midrule
HyperACE FullPAD & Tương quan đa vị trí đa thang & Backend và dataset ảnh hưởng; chưa chứng minh chống patch & Cơ chế và ablation gốc [1]; chưa ablation địa phương \\
DS C3k2 & Giảm params và FLOPs & Không bảo đảm latency & Table V gốc [1]; pipeline đo local \\
Khôi phục pretrained & Giữ shape tương thích & Gộp cùng recipe; chưa causal & Audit tensor và trial mới \\
Scheduler optimizer & Thích ứng horizon & 6 khác 300; chưa tối ưu toàn cục & Cùng 6 epoch, seed ghép cặp \\
Loss aug sampling & Ưu tiên miền Adv và điều chỉnh loss & Gộp yếu tố; lặp cảnh có thể overfit & Chỉ toàn cấu hình; cần ablation tiếp \\
Ba seed & Đo biến thiên optimizer & CI rộng, chưa đo bất định dataset & Thăm dò n=3 \\
COCO Adv & Cảnh thường và áo patch & Pretrain exposure, tương quan frame & Phân tầng, chưa causal defense \\
\bottomrule
\end{tabular}
\end{table*}
```
