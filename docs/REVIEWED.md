# REVIEW CHI TIẾT

## Bản thảo: “Tấn công đối kháng vật lý Sponge Patch: Khai thác lỗ hổng cạn kiệt tài nguyên trên hệ thống Edge Server bằng thuật toán tối ưu không gian”


## 1. Tóm tắt đánh giá

**Ý tưởng đáng theo đuổi, nhưng bằng chứng thực nghiệm và định nghĩa threat model hiện chưa đủ để bảo vệ các claim chính của bài.**


## 2. Đóng góp được tuyên bố của bài

Nhóm em tuyên bố các đóng góp chính:

1. Đề xuất Sponge Patch nhằm gây Visual Denial-of-Service trên Edge Server thông qua việc làm tăng số bounding boxes trước NMS.
2. Kết hợp Saliency Map với Genetic Algorithm để tối ưu hóa patch trong điều kiện tài nguyên hạn chế.
3. Đề xuất fitness function tập trung vào số lượng và điểm tin cậy của raw boxes, thay vì tối ưu misclassification/cross-entropy.
4. Trình bày lập luận toán học về nút thắt cổ chai NMS với độ phức tạp `O(N^2)`.
5. Kiểm chứng trên pipeline IP Camera/Edge Server cấu hình cũ, bao gồm digital injection, physical preliminary testing, ablation và transferability.

Trong các đóng góp trên, đóng góp 1 và 4 là thuyết phục nhất về mặt ý tưởng. Đóng góp 2 có tiềm năng, nhưng cần thực nghiệm nghiêm ngặt hơn. Đóng góp 3 và 5 hiện còn yếu vì liên quan trực tiếp đến mâu thuẫn black-box và physical claim.


## 3. Điểm mạnh

### 3.1. Bài toán có ý nghĩa thực tiễn

Edge-AI camera systems thường bị giới hạn bởi CPU, RAM, I/O và nhiệt độ. Việc đánh giá adversarial attack theo hướng availability là hợp lý, đặc biệt trong các hệ thống giám sát an ninh, camera thời gian thực, UAV hoặc ADAS. Bài không chỉ hỏi “mô hình có bị nhận diện sai không?”, mà hỏi “pipeline có còn duy trì được FPS thời gian thực không?”. Đây là câu hỏi có giá trị hệ thống hơn.

### 3.2. Điểm tấn công NMS có cơ sở kỹ thuật

Lập luận rằng NMS có thể trở thành bottleneck khi `N_active` tăng mạnh là có cơ sở. Công thức số cặp IoU:

```text
Operations = N_active * (N_active - 1) / 2
```

được sử dụng đúng về mặt trực giác. Ví dụ trong bài, `N_active = 50` cho 1.225 cặp và `N_active = 3000` cho 4.498.500 cặp; phép tính này chính xác. Đây là một điểm mạnh vì nó kết nối cơ chế tấn công với chi phí tính toán của pipeline, không chỉ dựa vào kết quả black-box tổng hợp.

### 3.3. Cấu trúc bài báo khá đầy đủ

Bản thảo có đầy đủ các thành phần mà một bài báo/nghiên cứu thường cần: threat model, method, algorithm, experimental setup, ablation, transferability, limitation và future work. Đây là nền tảng tốt để phát triển thành một bài nghiên cứu nghiêm túc, nếu em sửa phần bằng chứng và phần định claim.

### 3.4. Ablation và transferability là hướng đánh giá đúng

Việc em so sánh Random Search, Standard GA và Saliency-Guided GA là đúng hướng, vì nó nhằm trả lời liệu saliency có thật sự giúp GA hội tụ nhanh hơn hay không. Tương tự, việc thử transfer sang YOLOv5n và YOLOv8n là hợp lý vì transferability là một yếu tố quan trọng trong adversarial attack.

Vấn đề là cách báo cáo hiện chưa đủ chặt: thiếu nhiều seed, thiếu query budget, thiếu standard deviation/confidence interval và thiếu chi tiết về implementation. Tuy nhiên, về mặt thiết kế ý tưởng, hai mục này nên được giữ lại và làm mạnh hơn.

### 3.5. Nhóm em có nhận ra domain gap?

Phần hạn chế thừa nhận rằng printed physical patch chỉ đạt khoảng 60% CPU, thấp hơn nhiều so với digital setting. Đây là một điểm trung thực và nên được đẩy lên thành một kết quả trung tâm. Tuy nhiên, abstract và conclusion hiện tại lại không phản ánh đúng mức độ suy giảm này, tạo ra overclaim.



## 4. Các vấn đề nghiêm trọng

### Major Issue 1: Threat model “strict black-box” không nhất quán với fitness function

Đây là vấn đề nghiêm trọng nhất của bài.

Mục 3.1 khẳng định kẻ tấn công hoạt động trong kịch bản **Strict Black-box**: không có quyền truy cập kiến trúc mô hình, trọng số, gradient và chỉ có thể tương tác bằng cách gửi input. Tuy nhiên, Algorithm 1 yêu cầu:

```text
B_raw <- f(X_adv) (bỏ qua NMS)
```

và fitness function ở mục 3.4.2 cần các điểm tin cậy `C_i` của raw anchors trước NMS. Điều này có nghĩa kẻ tấn công phải đọc được output nội bộ của detector trước bước post-processing. Đây không phải strict black-box trong các hệ thống camera thương mại, vì người dùng bên ngoài thường chỉ thấy:

- final detections sau NMS;
- latency;
- FPS;
- trạng thái CPU/RAM nếu có side channel;
- hoặc hành vi quan sát được của hệ thống.

Nếu cần raw anchors/pre-NMS scores, mô hình này phải được gọi là **gray-box**, **score-based black-box with internal telemetry**, hoặc **white-box-at-output-head**, tùy mức truy cập. Việc giữ cụm “strict black-box” sẽ không thuyết phục với reviewer Q1/Q2.

**Tác động:** Nếu threat model sai, toàn bộ đóng góp “black-box hoàn toàn” bị suy yếu. Đây không phải lỗi nhỏ về wording, mà ảnh hưởng trực tiếp đến tính hợp lệ của method.

**Yêu cầu sửa:** Nhóm em cần chọn một trong hai hướng:

1. Hạ claim xuống thành gray-box/semi-black-box, nếu vẫn tối ưu trên raw anchors.
2. Giữ strict black-box, nhưng đổi fitness function sang proxy quan sát từ ngoài, ví dụ latency, FPS drop, response time, số detections sau NMS hoặc CPU side-channel.

Nếu chọn hướng 2, cần thiết kế lại thuật toán GA để chỉ cần query system output và đo latency, đồng thời báo cáo query budget.

### Major Issue 2: Claim “physical attack” vượt quá bằng chứng

Tiêu đề và tóm tắt nhấn mạnh đây là **Physical Sponge Attack**. Tuy nhiên, kết quả mạnh nhất của bài lại đến từ **Camera-in-the-loop Digital Attack**, trong đó patch được inject trực tiếp vào luồng số. Kịch bản này gần với Man-in-the-Middle/digital stream injection hơn là physical printed adversarial patch.

Phần physical testing chỉ là mục 4.3, được gọi đúng là **Preliminary Physical Testing**. Ở đó, hệ thống chỉ đạt CPU khoảng 60%, FPS giảm nhẹ hơn và không có bảng thống kê chi tiết. Không có ma trận điều kiện góc nhìn, khoảng cách, ánh sáng, chất liệu, nhiều scene hoặc nhiều lần lặp.

**Tác động:** Claim “physical DoS” chưa đủ bằng chứng. Kết quả hiện tại chỉ ủng hộ một claim yếu hơn:

> Digital Sponge Patch có thể gây resource pressure đáng kể; printed physical patch cho thấy tính khả thi sơ bộ nhưng chưa chứng minh DoS hoàn chỉnh.

**Yêu cầu sửa:** Bổ sung thực nghiệm vật lý đầy đủ hoặc hạ claim trong tiêu đề, abstract và conclusion. Nếu giữ claim physical, cần báo cáo:

- kích thước in thật theo cm;
- chất liệu in;
- máy in/loại giấy;
- khoảng cách camera;
- góc yaw/pitch/roll;
- độ sáng lux;
- độ phân giải camera;
- codec/năng lực nén;
- số lần lặp mỗi điều kiện;
- mean/std FPS, CPU, RAM, NMS latency;
- tỷ lệ thành công theo điều kiện.

### Major Issue 3: Mâu thuẫn về số liệu CPU/RAM/FPS

Tóm tắt và kết luận nói tấn công đẩy CPU lên 100%, chiếm dụng toàn bộ RAM và làm FPS tiệm cận 0. Nhưng số liệu Bảng 2:

- Clean stream: 15-21 FPS, CPU 52-64%, RAM ~1,2 GB.
- Attack ON: 5-11 FPS, CPU 67-78%, RAM ~4,8 GB.

Số liệu này cho thấy attack gây suy giảm đáng kể, nhưng **không phải CPU 100%**, **không phải RAM toàn bộ 8 GB**, và **không phải FPS tiệm cận 0**. Phần physical testing còn yếu hơn: CPU khoảng 60%.

Có thể con số 100% đến từ Hình 5 hoặc Bảng 4 trong ablation, nhưng em không tách rõ peak CPU, average CPU, digital injection, physical patch và training/application. Việc dùng con số 100% như kết luận tổng quát là overclaim.

**Yêu cầu sửa:** Lập một bảng tổng hợp theo kịch bản:

| Scenario | Patch type | CPU mean | CPU peak | RAM mean | FPS mean | NMS latency | n runs |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean | none | | | | | | |
| Digital injection | optimized patch | | | | | | |
| Printed patch | physical patch | | | | | | |
| Random texture | control | | | | | | |

Abstract và conclusion phải dùng số liệu đại diện, không chọn số peak nếu bảng chính báo cáo average.

### Major Issue 4: Mâu thuẫn kích thước patch

Mục 3.8 nói patch 64 x 64 pixel, chiếm khoảng 4% ảnh 320 x 320. Mục 4.1 nói patch 256 x 256 pixel, chiếm 15-20% khung hình khi áp dụng.

Đây là mâu thuẫn nghiêm trọng vì patch size ảnh hưởng đến:

- sức tấn công;
- tính kín đáo;
- khả năng in vật lý;
- so sánh với baseline;
- query budget cần hội tụ;
- domain gap;
- khả năng transferability.

Một patch 4% và một patch 15-20% là hai threat model thực tế khác nhau. Patch 15-20% có thể quá lớn để coi là stealthy physical attack.

**Yêu cầu sửa:** Nhóm em cần thống nhất:

- input resolution của detector là bao nhiêu;
- patch resolution tối ưu là bao nhiêu;
- patch được resize/scale thế nào khi đặt vào frame;
- diện tích patch tính theo input tensor hay frame camera gốc;
- thực nghiệm nào dùng 64 x 64, thực nghiệm nào dùng 256 x 256.

Nên bổ sung ablation theo patch area: 1%, 2%, 4%, 8% và 16%.

### Major Issue 5: Thiếu độ nghiêm ngặt thống kê

Tất cả bảng kết quả trong bài trông có vẻ là single-run hoặc log minh họa. Không có:

- số seed;
- số lần lặp;
- mean/std;
- confidence interval;
- kiểm định thống kê;
- variation giữa scene;
- variation giữa patch initialization;
- variation giữa distance/lighting/angle.

Điều này đặc biệt nghiêm trọng với GA, vì Genetic Algorithm nhạy với initialization và mutation. Kết quả “15 generations hội tụ” có thể là một seed may mắn nếu không có nhiều lần lặp.

**Yêu cầu sửa tối thiểu:**

- 5-10 seed cho Random Search, Standard GA và Saliency-Guided GA.
- Báo cáo mean ± std cho generations-to-convergence, training time, best fitness, CPU/FPS khi attack.
- Dùng paired test hoặc Wilcoxon/Mann-Whitney nếu phân phối không chuẩn.

**Yêu cầu mạnh hơn(optional):**

- nhiều video/scene;
- nhiều thiết bị Edge;
- nhiều camera/codec;
- nhiều detector và NMS implementation;
- defense evaluation.

### Major Issue 6: Ablation chưa tách được đóng góp thực sự của Saliency Map

Bảng 4 kết luận Saliency-Guided GA tốt hơn Standard GA và Random Search. Nhưng bài chưa nói rõ:

- các phương pháp có cùng query budget không;
- cùng patch area không;
- cùng EOT không;
- cùng generation/population/mutation rate không;
- cùng stopping criterion không;
- saliency map được tính bằng gradient hay heuristic;
- nếu dùng gradient thì có còn black-box không.

Nếu Saliency Map tính từ gradient của model, phương pháp không phải black-box. Nếu tính từ ảnh đầu vào bằng saliency/image processing độc lập, cần mô tả rõ. Nếu tính từ response của system qua nhiều query, cần báo cáo query cost.

**Yêu cầu sửa:** Định nghĩa rõ saliency mechanism và thêm ablation:

1. GA with random location.
2. GA with fixed center location.
3. GA with image saliency only.
4. GA with model-response saliency.
5. GA with oracle best location, đóng vai trò upper bound.

### Major Issue 7: Transferability claim quá rộng

Bảng 5 thử trên MobileNetV2 source, YOLOv5n và YOLOv8n target. Đây là bước đầu tốt, nhưng chưa đủ để kết luận lỗ hổng NMS là “điểm yếu kiến trúc mang tính hệ thống của ngành thị giác máy tính”.

Cần chú ý:

- YOLOv8 có anchor-free head nhưng vẫn có post-processing riêng; cần nói rõ NMS/max_det/top-k setting.
- Nhiều detector có giới hạn max detections, top-k prefilter, class-wise filtering, batched NMS, GPU NMS hoặc NMS-free design.
- Transferability cần test nhiều weights, datasets, scenes và thresholds.

**Yêu cầu sửa:** Hạ claim thành:

> Kết quả sơ bộ cho thấy patch có khả năng transfer sang một số lightweight detectors có bước post-processing NMS trong cấu hình edge CPU.

Nếu muốn claim hệ thống, cần thử thêm Faster R-CNN, RetinaNet, DETR/NMS-free variants, các phiên bản YOLO khác, TensorRT/OpenVINO/NPU pipeline và defense top-k cap.

### Major Issue 8: Chưa đo riêng NMS latency

Bài lập luận rằng NMS là bottleneck, nhưng kết quả chính chủ yếu báo cáo CPU/RAM/FPS tổng. CPU/FPS tổng có thể bị ảnh hưởng bởi:

- camera I/O;
- OpenCV buffer;
- Docker overhead;
- preprocessing;
- inference;
- rendering dashboard;
- memory allocation;
- network MJPEG;
- thermal throttling.

Phần hạn chế thừa nhận I/O blocking có thể làm CPU “nghỉ chờ” camera. Điều này càng cho thấy cần profile riêng.

**Yêu cầu sửa:** Báo cáo latency breakdown:

```text
Frame capture / decode
Preprocessing
Model forward pass
Confidence filtering
NMS
Rendering / logging
Total frame time
```

Nếu NMS chiếm tỷ lệ lớn khi attack, lập luận sẽ mạnh. Nếu tổng FPS giảm chủ yếu do I/O hoặc memory, claim NMS bottleneck cần được điều chỉnh.

### Major Issue 9: Thiếu baseline và negative controls

Hiện bài so sánh với clean stream và một vài phương pháp tối ưu, nhưng chưa có controls quan trọng:

- random texture cùng kích thước;
- checkerboard/high-frequency pattern;
- QR code;
- natural poster/printed image;
- adversarial patch tối ưu misclassification;
- patch không EOT;
- patch không saliency;
- patch có cùng tổng năng lượng/frequency.

Không có các baseline này, chưa biết Sponge Patch thật sự đặc biệt hay bất kỳ texture phức tạp cao nào cũng làm model sinh nhiều boxes.

### Major Issue 10: Thiếu defense và ethical/dual-use discussion

Đây là bài offensive security nhắm vào camera giám sát và Edge-AI. Theo tiêu chuẩn, bài cần có mục **Defense/Mitigation** và **Ethical Considerations**. Hiện tại, bài chủ yếu mô tả tấn công; trong future work còn đề cập UAV, ADAS, face recognition, tác chiến điện tử/an ninh quốc phòng. Nếu không có dual-use discussion, venue an ninh có thể đánh giá đây là thiếu trách nhiệm công bố.

**Cần bổ sung defense evaluation tối thiểu:**

- giới hạn top-k candidates trước NMS;
- tăng confidence threshold;
- max detections cap;
- entropy/frequency detector cho patch bất thường;
- frame dropping/backpressure control;
- async capture/inference pipeline;
- early-exit NMS hoặc batched NMS;
- monitor sudden candidate-box explosion.

**Cần bổ sung ethics:**

- mục tiêu nghiên cứu là nâng cao phòng thủ;
- không công bố artifacts có thể dùng trực tiếp nếu chưa có mitigation;
- khuyến nghị responsible disclosure với nhà vận hành hệ thống;
- giới hạn các ứng dụng dual-use.

### Major Issue 11: References chưa đạt chuẩn 

Danh mục tài liệu có nhiều lỗi:

- Mục `[10] S. P. Author et al.` là placeholder không chấp nhận được.
- Mục `[14] arXiv:2301.xxxx` để placeholder arXiv ID.
- Các mục `[23]-[25] Anonymous` không phù hợp nếu đây không phải bản double-blind nội bộ.
- Một số tài liệu 2024-2025 cần DOI, venue, status, version rõ ràng.
- Cần đảm bảo mọi reference trong danh mục đều được cite trong nội dung.

Với paper, reference hygiene là điều kiện cần. Lỗi placeholder bắt buộc phải sửa cho references đạt chuẩn(APA 6/7th)


## 5. Các vấn đề nhỏ và văn phong

1. Mục 1.2 có câu lặp: “Kiểm chứng mức độ đe dọa thực tiễn” bị lặp hai lần.
2. Bảng và hình có lỗi cross-reference hệ thống: Bảng 1/2/3/4/5 bị gọi nhầm nhiều lần trong mục 4.
3. Cụm từ “đột phá”, “vượt trội”, “chính thức làm tê liệt hoàn toàn” nên thay bằng văn phong trung tính hơn.
4. “Chiếm dụng toàn bộ RAM” không phù hợp với số liệu ~4,8 GB/8 GB.
5. Cần phân biệt rõ raw boxes, active boxes, post-threshold boxes và post-NMS detections.
6. Hàm fitness nên ghi rõ `C_i(x_adv)` thay vì `C_i` để nhất quán với biến đầu vào.
7. GA representation chưa rõ: gen là pixel RGB trực tiếp hay tham số hóa khác? Crossover/mutation cụ thể là gì?
8. Nếu patch Hình 3 trông “mượt” và có khối màu lớn, cần giải thích liên hệ với high-frequency spatial noise.
9. Cần nói rõ Docker container và Web Dashboard có ảnh hưởng đến CPU/RAM/FPS không.
10. Cần thống nhất “MobileNetV2 + YOLO-Head”, “MobileNet-SSD” và “YOLO”, vì các cụm này hiện được dùng lẫn.


## 6. Câu hỏi bắt buộc cho nhóm của Tuấn Anh

1. Attack có thật sự là strict black-box không? Nếu có, làm sao tính fitness khi không đọc được raw anchors/pre-NMS confidence?
2. Kích thước patch thật sự trong thực nghiệm là 64 x 64 hay 256 x 256? Diện tích patch được tính trên input tensor hay frame camera gốc?
3. Con số CPU 100% đến từ thực nghiệm nào? Đây là mean, peak, digital injection hay physical printed patch?
4. Kết quả Bảng 2, Bảng 4, Bảng 5 được lặp lại bao nhiêu lần và với bao nhiêu seed?
5. Saliency Map được tính bằng gradient của model, saliency của ảnh đầu vào hay query response của system?
6. NMS implementation có `max_det`, `top_k`, `conf_thres`, `iou_thres`, class-wise filtering hay batched NMS không?
7. Khi đặt confidence threshold thực tế hơn, ví dụ 0.25 hoặc 0.5 thay vì 0.01, attack còn hiệu quả không?
8. Có đo riêng thời gian NMS trong total latency không?
9. Printed physical patch được test với bao nhiêu góc, khoảng cách, ánh sáng và chất liệu?
10. Những defense đơn giản như top-k cap, max detections cap, adaptive threshold có làm vỡ attack không?
11. Em có kế hoạch công bố code, Dockerfile, config và log thực nghiệm không?
12. Em có bổ sung dual-use/ethical discussion không, đặc biệt khi bài đề cập UAV, ADAS và hệ thống an ninh quốc phòng?

## 7. Đánh giá định lượng theo rubric Scopus

| Tiêu chí | Điểm | Nhận xét |
|---|---:|---|
| Novelty | 6.5/10 | Ý tưởng physical/edge availability attack có tiềm năng, nhưng gần với Phantom Sponges và cần xác định novelty chính xác hơn. |
| Technical Soundness | 4/10 | Threat model mâu thuẫn với fitness/raw anchors; patch size và claim CPU/FPS không nhất quán. |
| Experimental Rigor | 3.5/10 | Thiếu nhiều seed, std/CI, ma trận vật lý, baseline và latency breakdown. |
| Reproducibility | 3/10 | Thiếu code, config, seed, NMS implementation, Docker details và dataset/video details. |
| Claim-Evidence Alignment | 3/10 | Abstract/conclusion overclaim so với Bảng 2 và physical preliminary test. |
| Writing and Organization | 5.5/10 | Cấu trúc tốt, nhưng lỗi cross-reference, placeholder references và văn phong quá mạnh. |
| Practical Impact | 7/10 | Chủ đề có giá trị thực tiễn nếu được chứng minh chặt chẽ hơn. |
| Ethics and Defense | 2.5/10 | Gần như thiếu mục defense/mitigation và dual-use discussion. |

**Đánh giá tổng thể:** 4.5/10 ở trạng thái hiện tại. Có thể tăng lên khoảng 7/10 nếu nhóm em sửa threat model, làm lại thực nghiệm và hạ claim cho đúng bằng chứng.

## 8. Revision roadmap để đưa bài lên chuẩn journal quốc tế

### Priority 1 - Bắt buộc sửa

1. **Định nghĩa lại threat model.** Bỏ “strict black-box” nếu vẫn dùng raw anchors; hoặc thiết kế fitness chỉ dựa trên latency/output quan sát được.
2. **Chuẩn hóa tất cả claim trong abstract/conclusion.** Phân biệt digital injection và physical printed patch; không nói CPU 100% nếu số liệu chính là 67-78% hoặc 60%.
3. **Giải quyết mâu thuẫn patch size.** Thống nhất 64 x 64/256 x 256, diện tích %, kích thước in vật lý và scale factor.
4. **Bổ sung thống kê nhiều lần lặp.** Ít nhất 5-10 seed cho ablation và main attack.
5. **Đo riêng NMS latency.** Nếu không chứng minh NMS là bottleneck bằng profiling riêng, đóng góp về NMS sẽ không đủ mạnh.
6. **Làm sạch references.** Xóa placeholder, bỏ Anonymous không cần thiết, điền đúng DOI/arXiv/venue.

### Priority 2 - Nên sửa để đạt chuẩn đăng journal vững hơn

1. Thêm baseline: random texture, checkerboard, QR code, natural poster và high-frequency noise.
2. Thêm defense evaluation: top-k cap, max detections, threshold tuning và batched/optimized NMS.
3. Thêm physical test matrix: góc, khoảng cách, ánh sáng và chất liệu.
4. Sửa cross-reference bảng/hình toàn bộ bài.
5. Bổ sung reproducibility statement: code, Dockerfile, config, seed và model weights.

### Priority 3 - Cần có nếu hướng đăng Q2

1. Mở rộng sang nhiều detector và NMS implementations.
2. Test trên nhiều Edge hardware: CPU-only, Intel NUC, Raspberry Pi, Jetson và NPU.
3. Đánh giá adaptive defenses và trade-off accuracy/latency.
4. Thêm responsible disclosure và dual-use risk assessment.
5. So sánh với các công trình phản biện tính thực tế của NMS latency attacks.

## 9. Kết luận

Bản thảo có nền tảng ý tưởng tốt và có thể phát triển thành một bài nghiên cứu có giá trị. Tuy nhiên, các lỗi hiện tại là lỗi cốt lõi về soundness và evidence, không chỉ là lỗi trình bày. Nếu nhóm em muốn đạt chuẩn đăng báo, cần chuyển bản thảo từ một “proof-of-concept có claim mạnh” thành một “system/security paper có threat model chặt, thực nghiệm lặp lại, profiling rõ ràng, baseline/defense đầy đủ và kết luận trung thực với dữ liệu”. Nếu làm được có thể hướng đến đăng journal quốc tế Q2

**Lời khuyên của a:** nên sửa những gì a note trong file và viết lại bản thảo bằng LaTex nhé em, bài này khá hay, có thể sủa lại để đăng lên Q2(có thể thử sức Q1), 1 bài nghiên cứu Q1/Q2 trong hồ sơ rất có giá trị để apply các job liên quan sau này
