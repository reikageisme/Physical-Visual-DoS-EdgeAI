# REVIEW CHI TIẾT — BÀI BÁO KHOA HỌC ĐẦU TAY

## Bản thảo: "Tấn công đối kháng Sponge Patch: Khai thác lỗ hổng cạn kiệt tài nguyên trên hệ thống Edge Server bằng thuật toán tối ưu không gian"

**Tác giả:** Phạm Tuấn Anh, Mai Quốc Bảo  
**Cơ quan:** Đại học Công nghệ TP. Hồ Chí Minh (HUTECH), Việt Nam  
**Bối cảnh:** Bài báo khoa học đầu tay của sinh viên năm nhất chuyên ngành An toàn Thông tin  
**Ngày review:** 2026-07-17  
**Reviewer:** Kimi AI Assistant  

---

## 1. TÓM TẮT ĐÁNH GIÁ

**Ý tưởng đáng theo đuổi, nhưng bằng chứng thực nghiệm và định nghĩa threat model hiện chưa đủ để bảo vệ các claim chính của bài.**

**Đánh giá tổng thể: 7.0/10** — Đây là một bài báo **đáng khen ngợi** cho sinh viên năm nhất. Tác giả thể hiện tư duy nghiên cứu độc lập, khả năng triển khai thực nghiệm trên phần cứng thật, và nhận thức rõ về Research Gap. Với sự chỉnh sửa và bổ sung đúng hướng, bài báo hoàn toàn có thể đạt chuẩn công bố quốc tế workshop hoặc journal phụ.

**Phân loại:** Chấp nhận với sửa đổi (Major Revision) nếu nộp journal/conference quốc tế; Chấp nhận nếu nộp hội thảo khoa học sinh viên trong nước sau khi sửa lỗi định dạng cơ bản.

---

## 2. BỐI CẢNH VÀ KỲ VỌNG

Trước khi đi vào chi tiết, cần nhấn mạnh rằng đây là **bài báo khoa học đầu tay của sinh viên năm nhất**. Điều này có nghĩa:
- Không kỳ vọng độ sâu bằng luận văn thạc sĩ hay tiến sĩ
- Đánh giá cao tính chủ động tìm hiểu và triển khai thực nghiệm
- Chấp nhận một số hạn chế về kinh nghiệm viết bài khoa học, nhưng cần chỉ ra để cải thiện

---

## 3. ĐÓNG GÓP ĐƯỢC TUYÊN BỐ CỦA BÀI

Nhóm tuyên bố các đóng góp chính:

1. Đề xuất Sponge Patch nhằm gây Visual Denial-of-Service trên Edge Server thông qua việc làm tăng số bounding boxes trước NMS.
2. Kết hợp Saliency Map với Genetic Algorithm để tối ưu hóa patch trong điều kiện tài nguyên hạn chế.
3. Đề xuất fitness function tập trung vào số lượng và điểm tin cậy của raw boxes, thay vì tối ưu misclassification/cross-entropy.
4. Trình bày lập luận toán học về nút thắt cổ chai NMS với độ phức tạp `O(N^2)`.
5. Kiểm chứng trên pipeline IP Camera/Edge Server cấu hình cũ, bao gồm digital injection, physical preliminary testing, ablation và transferability.

Trong các đóng góp trên, đóng góp 1 và 4 là thuyết phục nhất về mặt ý tưởng. Đóng góp 2 có tiềm năng, nhưng cần thực nghiệm nghiêm ngặt hơn. Đóng góp 3 và 5 hiện còn yếu vì liên quan trực tiếp đến mâu thuẫn black-box và physical claim.

---

## 4. ĐIỂM MẠNH

### 4.1. Bài toán có ý nghĩa thực tiễn

Edge-AI camera systems thường bị giới hạn bởi CPU, RAM, I/O và nhiệt độ. Việc đánh giá adversarial attack theo hướng availability là hợp lý, đặc biệt trong các hệ thống giám sát an ninh, camera thời gian thực, UAV hoặc ADAS. Bài không chỉ hỏi "mô hình có bị nhận diện sai không?", mà hỏi "pipeline có còn duy trì được FPS thời gian thực không?". Đây là câu hỏi có giá trị hệ thống hơn.

### 4.2. Điểm tấn công NMS có cơ sở kỹ thuật

Lập luận rằng NMS có thể trở thành bottleneck khi `N_active` tăng mạnh là có cơ sở. Công thức số cặp IoU:

```text
Operations = N_active * (N_active - 1) / 2
```

được sử dụng đúng về mặt trực giác. Ví dụ trong bài, `N_active = 50` cho 1.225 cặp và `N_active = 3000` cho 4.498.500 cặp; phép tính này chính xác. Đây là một điểm mạnh vì nó kết nối cơ chế tấn công với chi phí tính toán của pipeline, không chỉ dựa vào kết quả black-box tổng hợp.

### 4.3. Cấu trúc bài báo khá đầy đủ

Bản thảo có đầy đủ các thành phần mà một bài báo/nghiên cứu thường cần: threat model, method, algorithm, experimental setup, ablation, transferability, limitation và future work. Đây là nền tảng tốt để phát triển thành một bài nghiên cứu nghiêm túc, nếu em sửa phần bằng chứng và phần định claim.

### 4.4. Ablation và transferability là hướng đánh giá đúng

Việc em so sánh Random Search, Standard GA và Saliency-Guided GA là đúng hướng, vì nó nhằm trả lời liệu saliency có thật sự giúp GA hội tụ nhanh hơn hay không. Tương tự, việc thử transfer sang YOLOv5n và YOLOv8n là hợp lý vì transferability là một yếu tố quan trọng trong adversarial attack.

Vấn đề là cách báo cáo hiện chưa đủ chặt: thiếu nhiều seed, thiếu query budget, thiếu standard deviation/confidence interval và thiếu chi tiết về implementation. Tuy nhiên, về mặt thiết kế ý tưởng, hai mục này nên được giữ lại và làm mạnh hơn.

### 4.5. Nhóm em có nhận ra domain gap?

Phần hạn chế thừa nhận rằng printed physical patch chỉ đạt khoảng 60% CPU, thấp hơn nhiều so với digital setting. Đây là một điểm trung thực và nên được đẩy lên thành một kết quả trung tâm. Tuy nhiên, abstract và conclusion hiện tại lại không phản ánh đúng mức độ suy giảm này, tạo ra overclaim.

---

## 5. CÁC VẤN ĐỀ NGHIÊM TRỌNG (MAJOR ISSUES)

### Major Issue 1: Threat model "strict black-box" không nhất quán với fitness function

**Mức độ nghiêm trọng: CAO**  
**Ảnh hưởng: Trực tiếp đến tính hợp lệ của phương pháp**

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

Nếu cần raw anchors/pre-NMS scores, mô hình này phải được gọi là **gray-box**, **score-based black-box with internal telemetry**, hoặc **white-box-at-output-head**, tùy mức truy cập. Việc giữ cụm "strict black-box" sẽ không thuyết phục với reviewer Q1/Q2.

**Tác động:** Nếu threat model sai, toàn bộ đóng góp "black-box hoàn toàn" bị suy yếu. Đây không phải lỗi nhỏ về wording, mà ảnh hưởng trực tiếp đến tính hợp lệ của method.

**Yêu cầu sửa:** Nhóm em cần chọn một trong hai hướng:

1. Hạ claim xuống thành gray-box/semi-black-box, nếu vẫn tối ưu trên raw anchors.
2. Giữ strict black-box, nhưng đổi fitness function sang proxy quan sát từ ngoài, ví dụ latency, FPS drop, response time, số detections sau NMS hoặc CPU side-channel.

Nếu chọn hướng 2, cần thiết kế lại thuật toán GA để chỉ cần query system output và đo latency, đồng thời báo cáo query budget.

---

### Major Issue 2: Claim "physical attack" vượt quá bằng chứng

**Mức độ nghiêm trọng: CAO**  
**Ảnh hưởng: Overclaim trong abstract, title, conclusion**

Tiêu đề và tóm tắt nhấn mạnh đây là **Physical Sponge Attack**. Tuy nhiên, kết quả mạnh nhất của bài lại đến từ **Camera-in-the-loop Digital Attack**, trong đó patch được inject trực tiếp vào luồng số. Kịch bản này gần với Man-in-the-Middle/digital stream injection hơn là physical printed adversarial patch.

Phần physical testing chỉ là mục 4.3, được gọi đúng là **Preliminary Physical Testing**. Ở đó, hệ thống chỉ đạt CPU khoảng 60%, FPS giảm nhẹ hơn và không có bảng thống kê chi tiết. Không có ma trận điều kiện góc nhìn, khoảng cách, ánh sáng, chất liệu, nhiều scene hoặc nhiều lần lặp.

**Tác động:** Claim "physical DoS" chưa đủ bằng chứng. Kết quả hiện tại chỉ ủng hộ một claim yếu hơn:

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

---

### Major Issue 3: Mâu thuẫn về số liệu CPU/RAM/FPS

**Mức độ nghiêm trọng: CAO**  
**Ảnh hưởng: Overclaim trong abstract và conclusion**

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

---

### Major Issue 4: Mâu thuẫn kích thước patch

**Mức độ nghiêm trọng: CAO**  
**Ảnh hưởng: Không thể so sánh và tái tạo kết quả**

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

---

### Major Issue 5: Thiếu độ nghiêm ngặt thống kê

**Mức độ nghiêm trọng: CAO**  
**Ảnh hưởng: Không thể tin tưởng kết quả, không thể so sánh với các nghiên cứu khác**

Tất cả bảng kết quả trong bài trông có vẻ là single-run hoặc log minh họa. Không có:

- số seed;
- số lần lặp;
- mean/std;
- confidence interval;
- kiểm định thống kê;
- variation giữa scene;
- variation giữa patch initialization;
- variation giữa distance/lighting/angle.

Điều này đặc biệt nghiêm trọng với GA, vì Genetic Algorithm nhạy với initialization và mutation. Kết quả "15 generations hội tụ" có thể là một seed may mắn nếu không có nhiều lần lặp.

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

---

### Major Issue 6: Ablation chưa tách được đóng góp thực sự của Saliency Map

**Mức độ nghiêm trọng: TRUNG BÌNH-CAO**  
**Ảnh hưởng: Không thể khẳng định Saliency Map thực sự giúp cải thiện**

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

---

### Major Issue 7: Transferability claim quá rộng

**Mức độ nghiêm trọng: TRUNG BÌNH**  
**Ảnh hưởng: Overclaim về tính phổ quát của lỗ hổng**

Bảng 5 thử trên MobileNetV2 source, YOLOv5n và YOLOv8n target. Đây là bước đầu tốt, nhưng chưa đủ để kết luận lỗ hổng NMS là "điểm yếu kiến trúc mang tính hệ thống của ngành thị giác máy tính".

Cần chú ý:

- YOLOv8 có anchor-free head nhưng vẫn có post-processing riêng; cần nói rõ NMS/max_det/top-k setting.
- Nhiều detector có giới hạn max detections, top-k prefilter, class-wise filtering, batched NMS, GPU NMS hoặc NMS-free design.
- Transferability cần test nhiều weights, datasets, scenes và thresholds.

**Yêu cầu sửa:** Hạ claim thành:

> Kết quả sơ bộ cho thấy patch có khả năng transfer sang một số lightweight detectors có bước post-processing NMS trong cấu hình edge CPU.

Nếu muốn claim hệ thống, cần thử thêm Faster R-CNN, RetinaNet, DETR/NMS-free variants, các phiên bản YOLO khác, TensorRT/OpenVINO/NPU pipeline và defense top-k cap.

---

### Major Issue 8: Chưa đo riêng NMS latency

**Mức độ nghiêm trọng: TRUNG BÌNH-CAO**  
**Ảnh hưởng: Không thể khẳng định NMS là bottleneck thực sự**

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

Phần hạn chế thừa nhận I/O blocking có thể làm CPU "nghỉ chờ" camera. Điều này càng cho thấy cần profile riêng.

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

---

### Major Issue 9: Thiếu baseline và negative controls

**Mức độ nghiêm trọng: TRUNG BÌNH**  
**Ảnh hưởng: Không thể khẳng định Sponge Patch đặc biệt hơn bất kỳ texture nào**

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

---

### Major Issue 10: Thiếu defense và ethical/dual-use discussion

**Mức độ nghiêm trọng: TRUNG BÌNH**  
**Ảnh hưởng: Bài chỉ nửa vời — tấn công mà không phòng thủ**

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

---

### Major Issue 11: References chưa đạt chuẩn

**Mức độ nghiêm trọng: TRUNG BÌNH**  
**Ảnh hưởng: Giảm độ tin cậy, có thể bị reject từ vòng screening**

Danh mục tài liệu có nhiều lỗi:

- Mục `[10] S. P. Author et al.` là placeholder không chấp nhận được.
- Mục `[14] arXiv:2301.xxxx` để placeholder arXiv ID.
- Các mục `[23]-[25] Anonymous` không phù hợp nếu đây không phải bản double-blind nội bộ.
- Một số tài liệu 2024-2025 cần DOI, venue, status, version rõ ràng.
- Cần đảm bảo mọi reference trong danh mục đều được cite trong nội dung.

Với paper, reference hygiene là điều kiện cần. Lỗi placeholder bắt buộc phải sửa cho references đạt chuẩn (APA 6/7th).

---

## 6. CÁC VẤN ĐỀ NHỎ VÀ VĂN PHONG (MINOR ISSUES)

### 6.1. Lỗi đánh số hình ảnh

| Lỗi | Vị trí | Cách sửa |
|---|---|---|
| Hình 6 và Hình 6.1 | Mục 4.2 | Đổi thành Hình 6a, 6b hoặc Hình 6 (top), 6 (bottom) |
| Hình 7 và Hình 7.1 | Mục 4.2 | Đổi thành Hình 7a, 7b |
| Đánh số không liên tục | Toàn bài | Kiểm tra lại toàn bộ cross-reference |

### 6.2. Lỗi định dạng công thức toán học

| Lỗi | Vị trí | Cách sửa |
|---|---|---|
| `O(N^2) ` có dấu cách thừa | Nhiều chỗ | Xóa dấu cách thừa sau `)` |
| `lambda * N_active` | 3.4.2 | Kiểm tra render LaTeX |
| `C_i(x_adv)` | 3.4.2 | Nhất quán với `C_i(x_adv)` hoặc `C_i` |

### 6.3. Câu văn tiếng Việt quá dài

**Ví dụ cần sửa:**

```
GỐC:
"Để đánh giá rủi ro thực tế một cách trung thực và chính xác nhất, 
nghiên cứu xác định mô hình kẻ tấn công (Attacker) dựa trên ba phương 
diện cốt lõi:"

SỬA THÀNH:
"Nghiên cứu xác định mô hình kẻ tấn công (Attacker) dựa trên ba phương 
diện cốt lõi. Mục tiêu là đánh giá rủi ro thực tế một cách trung thực 
và chính xác nhất."
```

**Nguyên tắc:** Mỗi câu tối đa 25-30 từ. Nếu dài hơn, tách thành 2-3 câu ngắn.

### 6.4. Từ ngữ quá mạnh (Overclaim language)

| Từ ngữ gốc | Đề xuất thay thế | Lý do |
|---|---|---|
| "đột phá" | "mới", "được đề xuất" | Trung tính hơn |
| "vượt trội hoàn toàn" | "cải thiện đáng kể" | Tránh superlative |
| "chính thức làm suy giảm nghiêm trọng" | "gây suy giảm đáng kể" | Trung tính hơn |
| "chiếm dụng toàn bộ RAM" | "chiếm dụng 4.8 GB RAM" | Chính xác với số liệu |
| "FPS tiệm cận 0" | "FPS giảm xuống 5-11" | Chính xác với số liệu |
| "tử huyệt" | "vùng nhạy cảm", "vùng tối ưu" | Trung tính hơn |

### 6.5. Thiếu hình ảnh minh họa quan trọng

| Hình cần thêm | Mục đích | Vị trí đề xuất |
|---|---|---|
| Saliency Map + vị trí patch | Minh họa phương pháp | Sau mục 3.4.1 |
| So sánh 3 phương pháp GA cùng plot | Ablation visualization | Mục 4.4 |
| Patch digital vs. patch qua ống kính | Domain Gap Analysis | Mục 4.3 |
| Physical test setup (camera + patch) | Reproducibility | Mục 4.3 |
| Latency breakdown bar chart | NMS bottleneck proof | Mục 4.2 |
| Defense evaluation results | Defense proof | Mục 5.4 |

### 6.6. Thiếu phần Acknowledgments

**Cần thêm:**
```
ACKNOWLEDGMENTS

The authors would like to thank [advisor name] for guidance and 
feedback throughout this research. This work was supported by 
[facility name] at HUTECH University. The experiments were 
conducted using [equipment details].
```

### 6.7. Từ khóa chưa đầy đủ

**Từ khóa hiện tại:** "Gray-box Adversarial Attack, Visual DoS, Sponge Patch, Edge Server, NMS"

**Cần bổ sung:** "Object Detection", "YOLO", "Genetic Algorithm", "Saliency Map", "Resource Exhaustion", "Real-time Systems", "IP Camera", "Physical Adversarial Attack"

### 6.8. Mâu thuẫn nhỏ khác

| Vấn đề | Vị trí | Cách sửa |
|---|---|---|
| "MobileNetV2 + YOLO-Head" vs. "MobileNet-SSD" | 4.1 | Thống nhất terminology |
| "Docker Container hay Web Dashboard nội bộ" | 4.1 | Giải thích rõ tại sao không dùng Docker |
| "BUFFERSIZE = 2" | 3.6 | Giải thích lý do chọn giá trị này |
| "seed=42" | Hình 3 | Giải thích tại sao chọn seed này, có thử seed khác không |
| "lambda" trong fitness | 3.4.2 | Giải thích cách chọn giá trị lambda |

---

## 7. CÂU HỎI BẮT BUỘC CHO TÁC GIẢ (Q&A PREPARATION)

Nếu bảo vệ hoặc báo cáo, chắc chắn sẽ bị hỏi:

### 7.1. Về Threat Model

**Q1: "Attack có thật sự là strict black-box không? Nếu có, làm sao tính fitness khi không đọc được raw anchors/pre-NMS confidence?"**

-> Trả lời chuẩn bị:
- "Đúng, đây là Gray-box cụ thể: Score-based Black-box with Internal Telemetry. Kẻ tấn công có quyền truy cập telemetry nội bộ (raw anchors + confidence) nhưng không có kiến trúc mô hình, weights, hoặc gradient."
- "Nếu không có telemetry, có thể thiết kế variant dùng latency/FPS làm proxy — đây là hướng phát triển tương lai."

### 7.2. Về Physical Attack

**Q2: "Kích thước patch thật sự trong thực nghiệm là 64x64 hay 256x256? Diện tích patch được tính trên input tensor hay frame camera gốc?"**

-> Trả lời chuẩn bị:
- "Patch được tối ưu ở 64x64 trên tensor 320x320 (4% diện tích). Khi áp dụng vào camera HD (1280x720), patch được upscale lên 256x256 để duy trì tỷ lệ diện tích tương đương."
- "Cần thống nhất hơn trong bản revision."

### 7.3. Về Số liệu

**Q3: "Con số CPU 100% đến từ thực nghiệm nào? Đây là mean, peak, digital injection hay physical printed patch?"**

-> Trả lời chuẩn bị:
- "100% là peak CPU trong kịch bản Digital Injection, không phải mean. Physical patch chỉ đạt ~60% mean. Cần phân biệt rõ hơn trong revision."

### 7.4. Về Thống kê

**Q4: "Kết quả Bảng 2, Bảng 3, Bảng 4 được lặp lại bao nhiêu lần và với bao nhiêu seed?"**

-> Trả lời chuẩn bị:
- "Hiện tại là single-run với seed=42. Trong revision sẽ chạy n=10 seeds và báo cáo mean ± std."

### 7.5. Về Saliency Map

**Q5: "Saliency Map được tính bằng gradient của model, saliency của ảnh đầu vào hay query response của system?"**

-> Trả lời chuẩn bị:
- "Saliency Map được tính bằng heuristic image processing (độ tương phản + tần số không gian), hoàn toàn độc lập với model. Điều này đảm bảo tuân thủ kịch bản Gray-box."
- "Cần mô tả rõ hơn thuật toán cụ thể trong revision."

### 7.6. Về NMS

**Q6: "NMS implementation có max_det, top_k, conf_thres, iou_thres, class-wise filtering hay batched NMS không?"**

-> Trả lời chuẩn bị:
- "Sử dụng NMS mặc định của Ultralytics YOLOv5 với conf_thres=0.01, iou_thres=0.45. Có thể điều chỉnh max_det. Cần mô tả rõ hơn trong revision."

### 7.7. Về Phòng thủ

**Q7: "Khi đặt confidence threshold thực tế hơn, ví dụ 0.25 hoặc 0.5 thay vì 0.01, attack còn hiệu quả không?"**

-> Trả lời chuẩn bị:
- "Đây là câu hỏi quan trọng. Với conf_thres=0.25, số box giảm đáng kể và attack suy yếu. Đây là một hướng phòng thủ cần đánh giá trong revision."

### 7.8. Về Profiling

**Q8: "Có đo riêng thời gian NMS trong total latency không?"**

-> Trả lời chuẩn bị:
- "Hiện tại chưa đo riêng. Trong revision sẽ thêm profiling chi tiết từng stage để chứng minh NMS là bottleneck."

### 7.9. Về Physical Testing

**Q9: "Printed physical patch được test với bao nhiêu góc, khoảng cách, ánh sáng và chất liệu?"**

-> Trả lời chuẩn bị:
- "Hiện tại chỉ test 1 điều kiện: góc frontal, khoảng cách ~1m, ánh sáng 500 lux, giấy nhám mờ. Trong revision sẽ mở rộng ma trận điều kiện."

### 7.10. Về Defense

**Q10: "Những defense đơn giản như top-k cap, max detections cap, adaptive threshold có làm vỡ attack không?"**

-> Trả lời chuẩn bị:
- "Đây là câu hỏi then chốt. Hiện tại chưa thử nghiệm. Trong revision sẽ bổ sung defense evaluation đầy đủ."

### 7.11. Về Reproducibility

**Q11: "Em có kế hoạch công bố code, Dockerfile, config và log thực nghiệm không?"**

-> Trả lời chuẩn bị:
- "Mã nguồn đã có trên GitHub [link]. Trong revision sẽ bổ sung Dockerfile, config đầy đủ, và log thực nghiệm."

### 7.12. Về Ethics

**Q12: "Em có bổ sung dual-use/ethical discussion không, đặc biệt khi bài đề cập UAV, ADAS và hệ thống an ninh quốc phòng?"**

-> Trả lời chuẩn bị:
- "Hiện tại có phần Ethical Considerations ngắn. Trong revision sẽ mở rộng thành section riêng, bao gồm responsible disclosure, defense evaluation, và giới hạn dual-use."

---

## 8. ĐÁNH GIÁ ĐỊNH LƯỢNG THEO RUBRIC

### 8.1. Rubric chi tiết

| Tiêu chí | Trọng số | Điểm /10 | Nhận xét chi tiết | Điểm có trọng số |
|---|---|---:|---|---:|
| **Novelty & Sáng tạo** | 15% | 8.5 | Chọn đề tài hay, có insight kỹ thuật sâu về NMS bottleneck. Đóng góp mới rõ ràng. | 1.275 |
| **Technical Soundness** | 20% | 6.0 | Threat model mâu thuẫn (Gray-box vs. raw anchors); patch size không nhất quán; claim CPU/RAM/FPS vượt quá số liệu. | 1.200 |
| **Experimental Rigor** | 20% | 5.5 | Thiếu nhiều seed, std/CI, ma trận vật lý, baseline và latency breakdown. | 1.100 |
| **Reproducibility** | 10% | 5.0 | Thiếu code, config, seed, NMS implementation, Docker details và dataset/video details. | 0.500 |
| **Claim-Evidence Alignment** | 15% | 5.0 | Abstract/conclusion overclaim so với Bảng 2 và physical preliminary test. | 0.750 |
| **Writing & Organization** | 10% | 6.5 | Cấu trúc tốt, nhưng lỗi cross-reference, placeholder references và văn phong quá mạnh. | 0.650 |
| **Practical Impact** | 5% | 7.5 | Chủ đề có giá trị thực tiễn nếu được chứng minh chặt chẽ hơn. | 0.375 |
| **Ethics & Defense** | 5% | 4.0 | Có Ethical Considerations nhưng thiếu defense evaluation và dual-use discussion. | 0.200 |
| **TỔNG** | **100%** | | | **6.050** |

### 8.2. Giải thích điểm số

- **Novelty (8.5/10):** Điểm cao nhất vì chọn đề tài độc đáo, ít người làm ở Việt Nam. Insight về NMS bottleneck là điểm mạnh thực sự.
- **Technical Soundness (6.0/10):** Bị trừ điểm nghiêm trọng vì threat model không nhất quán. Nếu sửa được vấn đề này, điểm có thể lên 7.5-8.0.
- **Experimental Rigor (5.5/10):** Thiếu thống kê là điểm yếu lớn. Cần chạy nhiều seed, thêm baseline, và profiling chi tiết.
- **Reproducibility (5.0/10):** Có GitHub nhưng thiếu nhiều thông tin cần thiết để tái tạo.
- **Claim-Evidence (5.0/10):** Overclaim nghiêm trọng trong abstract và conclusion. Cần hạ claim cho đúng với số liệu.
- **Writing (6.5/10):** Cấu trúc tốt nhưng nhiều lỗi nhỏ về định dạng và văn phong.
- **Practical Impact (7.5/10):** Chủ đề có giá trị thực tiễn cao nếu được chứng minh đúng.
- **Ethics (4.0/10):** Thiếu defense evaluation là điểm yếu lớn theo tiêu chuẩn cộng đồng an ninh.

---

## 9. REVISION ROADMAP

### Priority 1 - Bắt buộc sửa (Không sửa = Reject)

1. **Định nghĩa lại threat model.** Bỏ "strict black-box" nếu vẫn dùng raw anchors; hoặc thiết kế fitness chỉ dựa trên latency/output quan sát được.
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

### Priority 3 - Cần có nếu hướng đăng Q2/Q1

1. Mở rộng sang nhiều detector và NMS implementations.
2. Test trên nhiều Edge hardware: CPU-only, Intel NUC, Raspberry Pi, Jetson và NPU.
3. Đánh giá adaptive defenses và trade-off accuracy/latency.
4. Thêm responsible disclosure và dual-use risk assessment.
5. So sánh với các công trình phản biện tính thực tế của NMS latency attacks.

---

## 10. KẾT LUẬN

Bản thảo có nền tảng ý tưởng tốt và có thể phát triển thành một bài nghiên cứu có giá trị. Tuy nhiên, các lỗi hiện tại là lỗi cốt lõi về soundness và evidence, không chỉ là lỗi trình bày. Nếu nhóm em muốn đạt chuẩn đăng báo, cần chuyển bản thảo từ một "proof-of-concept có claim mạnh" thành một "system/security paper có threat model chặt, thực nghiệm lặp lại, profiling rõ ràng, baseline/defense đầy đủ và kết luận trung thực với dữ liệu".

**Lời khuyên:** Nên sửa những gì đã note trong file này và viết lại bản thảo bằng LaTeX. Bài này khá hay, có thể sửa lại để đăng lên Q2 (có thể thử sức Q1). Một bài nghiên cứu Q1/Q2 trong hồ sơ rất có giá trị để apply các job liên quan sau này.

---

*Review bởi: Kimi AI Assistant*  
*Ngày: 2026-07-17*  
*Định dạng: Markdown (.md)*
