# KIỂM TOÁN KHOA HỌC VÀ KHOẢNG CÁCH Q1

> Báo cáo lịch sử: các mục 1–14 phản ánh bằng chứng trước các lượt kiểm chứng mới; mục 15 chỉ bổ sung tóm tắt. Không dùng các ô chưa hoàn tất ở báo cáo này làm trạng thái hiện tại. Báo cáo sau hợp nhất, hoàn toàn bằng tiếng Việt và đối chiếu số liệu mới, nằm tại `KIEM_TOAN_Q1_SAU_HOP_NHAT_VI.md` trong cùng thư mục. Cả hai là tài liệu phản biện cục bộ, không thuộc gói công bố Drive/Git.

## Bản thảo: *Adversarial Sponge Attack on Edge Servers*

**Ngày kiểm toán:** 02/10/2026  
**Phạm vi bằng chứng:** `BaoCaoKhoaHocAdversarialAttacks.tex`, mã nguồn trong `core/`, `attack/`, `experiments/`, log trong `logs/`, và kết quả trong `outputs/`.  
**Nguyên tắc:** số liệu không có log/config truy vết được không được dùng như kết quả thực nghiệm đã xác nhận.

## 1. Kết luận điều hành

**Khuyến nghị hiện tại: Reject and resubmit after major experimental reconstruction.** Ý tưởng đánh vào tính sẵn sàng của detector biên thông qua khuếch đại số ứng viên đi vào NMS có giá trị, nhưng bản thảo hiện chưa đủ bằng chứng cho các claim “Edge-server DoS”, “physical attack”, “cross-architecture transferability” và “Saliency-Guided GA vượt trội”.

Những phát hiện quyết định:

1. Fitness trong implementation được **tối đa hóa**; phương trình dấu “+” là phù hợp với code. Lỗi nằm ở số liệu bị hỏng trong LaTeX và diễn giải quá mức.
2. Tỷ lệ số cặp lý thuyết từ 50 lên 3.000 ứng viên là **3.672,24 lần**, không phải 5,94 lần. Đây không phải là tỷ lệ latency đo được.
3. `3000+` ứng viên không được log hiện có hỗ trợ. Kết quả mô phỏng có truy vết báo 47,3 ứng viên sạch và 55,7 ứng viên digital; benchmark 3.000 hộp là microbenchmark tổng hợp, không phải detector tạo ra 3.000 hộp.
4. Source model trong code hiện tại là **YOLOv8n anchor-free**, không phải MobileNetV2-SSD hay MobileNetV2 + YOLO head.
5. Kết quả 10 seed không chứng minh saliency vượt trội: 65,07 ± 4,67 so với 64,02 ± 5,17; số thế hệ dừng trung bình là 15,4 so với 14,7.
6. Baseline texture cho thấy Sponge Patch hiện tại yếu hơn solid color và checkerboard ở số ứng viên/NMS latency. Đây là negative result phải giữ lại.
7. Transfer script không đo NMS nhất quán giữa các kiến trúc; chính wrapper ghi rõ YOLO backbone path không bao gồm NMS. Kết quả không đủ để tuyên bố transferability.
8. “Physical” trong `proper_dos_sim.py` là **EOT/degradation simulation**, không phải patch in được camera quan sát.
9. Log camera không có manifest phần cứng/phần mềm; các cặp clean/attack hiện không cho thấy mức tăng CPU ổn định. Không thể dùng chúng để xác nhận các dải 52–64% → 67–78% hay RAM 4,8 GB.
10. Confidence threshold 0,01 là ngưỡng fitness/pre-NMS trong tối ưu, không nên trình bày như ngưỡng vận hành detector mặc định.

---

## 2. Major-review audit table

| ID | Mục / dòng | Phát biểu hiện tại | Loại vấn đề | Mức độ | Vì sao reviewer có thể bác | Sửa đề xuất | Sửa không cần thí nghiệm? |
|---|---|---|---|---|---|---|---|
| A01 | Abstract; 3.2; Results | Patch tạo “hàng nghìn” box và gây DoS | Không có bằng chứng truy vết | **Critical** | Log detector hiện có không đạt 3.000; benchmark 3.000 là synthetic NMS input | Xóa claim hoặc thay bằng kết quả ứng viên có log; tách microbenchmark | Có, để hạ claim; **Không** để chứng minh claim cũ |
| A02 | NMS complexity | 50→3.000 tương ứng khoảng 5,94× | Sai toán học | **Critical** | (4.498.500/1.225=3.672,24) | Sửa thành tỷ lệ số cặp lý thuyết; không gọi là latency ratio | Có |
| A03 | Threat model | Black-box/gray-box dùng lẫn | Construct validity | **Critical** | Fitness cần pre-NMS scores, không phải strict black-box | Gọi là score-based gray-box with internal telemetry | Có |
| A04 | Model description | MobileNetV2+YOLO head, SSD, YOLOv5, YOLOv8 dùng lẫn | Reproducibility | **Critical** | Không xác định được source/target model | Chốt code hiện tại: YOLOv8n source; các model khác chỉ là exploratory transfer targets | Có |
| A05 | Fitness/Ablation | Proposed tốt hơn vì số nhỏ hơn / số bị hỏng | Logic + typesetting | **Critical** | Code dùng `argmax` và `topk`: số lớn hơn mới tốt | Báo 65,07±4,67 vs 64,02±5,17; bỏ “vượt trội hoàn toàn” | Có |
| A06 | Physical evaluation | CPU ~60% chứng minh physical DoS | Overclaim | **Critical** | CPU load không tự chứng minh DoS; log vật lý đầy đủ không có | Gọi là preliminary physical feasibility nếu có protocol/log; hiện tại đánh dấu chưa xác minh | Có để hạ claim; Không để giữ claim mạnh |
| A07 | Transferability | MobileNet source → YOLOv5n/YOLOv8n với 2.150+/1.800+ box | Unsupported result | **Critical** | Không có output tương ứng; script hiện test model khác và đo không đồng nhất | Xóa bảng cũ; trình bày transfer experiment là inconclusive | Có |
| A08 | Hardware | Intel i5 cũ, 8 GB, Ubuntu, IP camera | Thiếu manifest | **High** | Log cho thấy RAM dùng ~13,5 GB, không phù hợp máy 8 GB; không có CPU/OS/version trong log | `[AUTHOR MUST PROVIDE HARDWARE MANIFEST]` | Không |
| A09 | Box terms | “Raw Boxes” dùng cho 12–45/3.000+ | Terminology | **High** | Chạy trực tiếp ngày 02/10/2026 xác nhận YOLOv8n có 2.100 vị trí dự đoán ở 320×320; 8.400 chỉ ứng với 640×640. Số sau threshold là active candidates | Đổi header thành Active/Post-threshold candidates và sửa số vị trí theo kích thước đầu vào thực đo | Có |
| A10 | NMS cost | (O(N^2)) được coi là runtime thực tế tất định | Theoretical overreach | **High** | Vectorization, top-k, class filtering và implementation ảnh hưởng runtime | Gọi là worst-case pairwise proxy; báo microbenchmark riêng | Có |
| A11 | Patch size | 64×64, 256×256, 4%, 15–20% | Inconsistency | **High** | Khác threat model và mức hiển thị | Dùng area ratio: 64²/320²=4%; mọi upscaling phải giữ ratio và được mô tả riêng | Có nếu xác nhận pipeline; còn camera footprint cần tác giả |
| A12 | Threshold | τ=0,01 trình bày như vận hành thường | Fairness | **High** | Ngưỡng thấp làm tăng candidates; baseline deployment có thể dùng 0,25 | Gọi là optimization threshold; đánh giá sensitivity 0,01–0,5 trên dữ liệu thật | Có để làm rõ; Không cho sensitivity hợp lệ |
| A13 | Saliency | “top 5%” và “thu hẹp không gian lai ghép” | Method/code mismatch | **High** | Code chọn một vị trí bằng Laplacian variance; không tối ưu top-5% mask | Viết đúng implementation: fixed location selected from image-based Laplacian response | Có |
| A14 | GA representation | Gaussian mutation/block crossover | Method/code mismatch | **Medium** | Code mặc định dùng horizontal split; mutation dùng uniform noise | Sửa mô tả thuật toán theo code | Có |
| A15 | EOT | ±30°, scale/perspective | Method/code mismatch | **High** | Code dùng rotation ±15°, brightness/contrast ±0,2, blur và noise 0,02; không scale/perspective | Sửa thông số; nêu một transform/sample/evaluation | Có |
| A16 | Multi-seed | “n=5” / “10 seed” lẫn nhau | Statistics | **High** | Không rõ đơn vị độc lập; base image là random tensor | Báo đúng 10 seed nhưng gọi exploratory synthetic-image optimization | Có |
| A17 | Random Search | So sánh công bằng query budget | Confounded baseline | **High** | Output random search dùng 500 eval; GA tối đa 600 và có early stopping/EOT khác | Không kết luận superiority; chạy lại cùng image/EOT/query budget | Không |
| A18 | Candidate cap defense | `max_det` được gọi là cap trước NMS | Implementation validity | **High** | Code cắt `nms_idx` sau khi NMS đã chạy; không giảm cost NMS | Cài pre-NMS top-k/candidate cap và đo accuracy-latency trade-off | Không |
| A19 | Baseline controls | Sponge Patch vượt texture thường | Contradicted by output | **Critical** | Solid color/checkerboard tạo nhiều candidates và NMS latency hơn trong output hiện tại | Báo negative result; kiểm tra lại training/data/protocol | Có để báo trung thực; Không để cải thiện phương pháp |
| A20 | Transfer script | Cross-architecture NMS latency | Measurement mismatch | **Critical** | YOLO path chỉ đo raw forward; torchvision path gộp post-processing | Chuẩn hóa end-to-end pipeline và stage timing cho từng model | Không |
| A21 | RAM claim | “chiếm dụng 4,8 GB”/“toàn bộ RAM” | Unsupported/ambiguous | **High** | Không rõ process RSS hay system used; log hiện ~13,5 GB system used | Xóa cho tới khi có process-level RSS/USS và baseline delta | Có để xóa; Không để giữ claim |
| A22 | Figures | Caption “CPU 100%, proving attack” | Causal overclaim | **High** | Peak không đại diện mean; không có paired trials | Caption trung tính, nêu scenario và statistic | Có |
| A23 | References | Một số metadata sai/không tìm thấy | Citation integrity | **High** | Ref 8 sai venue/year; ref 9 sai authors/venue; ref 13 sai năm/venue; ref 16 không xác minh được | Thay bằng metadata từ publisher/proceedings; đánh dấu ref 16 | Có, cần tra cứu |
| A24 | Organization | Cấu trúc luận văn, nhiều subsection mô tả | Presentation | **Medium** | Loãng research questions, validity và evidence | Tái cấu trúc 14 mục journal | Có |
| A25 | Reproducibility | Không pin dependency/model hash/run manifest | Artifact quality | **High** | Không tái tạo được bảng/hình | Ghi exact versions, checksum, config, seed, run ID | Một phần; cần chạy lại |

---

## 3. Ba nhóm hành động

### A. SAFE TO FIX NOW

- Sửa phép tính 3.672,24× và tách khỏi latency đo được.
- Chuẩn hóa thuật ngữ box.
- Đổi threat model thành gray-box có internal telemetry.
- Chốt source implementation hiện tại là YOLOv8n, input 320×320.
- Sửa fitness thành bài toán maximize và đồng bộ algorithm/code.
- Sửa EOT thành rotation ±15°, brightness/contrast 0,2, Gaussian blur, additive noise 0,02.
- Định nghĩa patch 64×64 trên 320×320 là 4% diện tích.
- Hạ toàn bộ claim physical/transfer/DoS không đủ bằng chứng.
- Tách Digital Injection, Camera-in-the-loop, EOT simulation và Printed Physical Patch.
- Sửa cấu trúc bài, ngôn ngữ và caption trung tính.

### B. NEEDS AUTHOR CONFIRMATION

- CPU chính xác (i5-2400 hay model khác), core/thread, RAM và OS của máy đo.
- Log nào là bare-metal edge, log nào là laptop/VM/simulation.
- Camera model, codec, kết nối, độ phân giải/FPS và capture settings.
- Patch in thật: kích thước cm, vật liệu, máy in, khoảng cách, góc, ánh sáng, số trial.
- CPU 100% là peak của run nào; 4,8 GB là process delta hay system total.
- Model/weights dùng trong bản thảo cũ có từng là MobileNetV2-SSD/YOLOv5 hay không.
- Dataset/scene thật dùng để tối ưu patch; output hiện dùng random/synthetic inputs.

### C. REQUIRES NEW EXPERIMENT

- Tất cả thí nghiệm Q1 được liệt kê tại Mục 10.

---

## 4. Critical numerical and logical errors

### 4.1. NMS pair-count amplification

\[
C(50,2)=\frac{50\times49}{2}=1{,}225,
\qquad
C(3000,2)=\frac{3000\times2999}{2}=4{,}498{,}500.
\]

\[
\frac{4{,}498{,}500}{1{,}225}=3{,}672.2449.
\]

Đây chỉ là **theoretical pairwise-comparison amplification** dưới mô hình so sánh mọi cặp. Microbenchmark repository đo `torchvision.ops.nms` ở 50 và 3.000 hộp lần lượt khoảng 0,415 ms và 23,647 ms, tức khoảng 57,0× trong lần chạy lưu lại — không phải 3.672×. Hai con số không mâu thuẫn: implementation tối ưu/vectorized không nhất thiết tỷ lệ trực tiếp với số cặp lý thuyết.

### 4.2. Fitness direction

Code xác nhận:

- `fit_tensor.max()`;
- `fit_tensor.argmax()`;
- `torch.topk(scores_t, elite_k)`;
- `global_best_fit = -inf` và cập nhật khi fitness tăng.

Do đó:

\[
\max_p F(p),\qquad
F(p)=\sum_i C_i(A(x,p,l))\mathbf{1}[C_i>\tau]+\lambda N_{\mathrm{active}}.
\]

Số fitness lớn hơn là tốt hơn. Các chuỗi “6.26/8.08/9.30” trong prompt không khớp output; LaTeX hiện còn bị lỗi chuyển đổi `\section{26 ± 5.45}`. Output có truy vết là 65,07±4,67 (saliency), 64,02±5,17 (no-saliency), 15,18±0,51 (random search, nhưng protocol không ngang bằng).

### 4.3. Evidence conflict

| Claim cũ | Bằng chứng repository | Kết luận audit |
|---|---|---|
| 3.000+ active boxes | `proper_sim`: 55,7 mean, 118 max; defense output: 10 at τ=0,01 | Không được hỗ trợ |
| Edge CPU 52–64→67–78% | Hai cặp log cho kết quả không ổn định; một cặp gần như bằng nhau, một cặp attack thấp hơn clean | Không được xác nhận |
| 4,8 GB RAM | Log ghi system used ~13,5 GB; không có process delta | Không truy vết được |
| Physical patch ~60% CPU | Không có manifest/protocol/log vật lý phân biệt | Cần tác giả xác nhận và chạy lại |
| MobileNet→YOLO transfer | Output hiện là YOLOv8/Faster R-CNN/RetinaNet/DETR, trên random tensor | Bảng cũ không được hỗ trợ |

---

## 5. Corrected terminology dictionary

| Thuật ngữ chuẩn | Ký hiệu | Định nghĩa | Không dùng thay thế bằng |
|---|---|---|---|
| Raw predictions | \(B_{\mathrm{raw}}\), \(K\) | Tất cả output của detector trước confidence filtering. Với YOLOv8n 320×320: 2.100 vị trí dự đoán, đã xác nhận bằng tensor đầu ra; 640×640 mới có 8.400 | “Raw boxes” cho số đã threshold |
| Active / post-threshold candidates | \(B_\tau\), \(N_{\mathrm{active}}\) | Predictions có \(C_i>\tau\), được đưa vào NMS | Raw anchors, post-NMS boxes |
| Post-NMS detections | \(D=\mathrm{NMS}(B_\tau)\) | Detections cuối cùng sau suppression và cap | Active boxes |
| Optimization threshold | \(\tau_{\mathrm{opt}}=0.01\) | Ngưỡng dùng trong fitness GA | Deployment threshold |
| Deployment/evaluation threshold | \(\tau_{\mathrm{dep}}\) | Ngưỡng pipeline nạn nhân, phải báo theo experiment | 0,01 mặc định nếu chưa chứng minh |
| Candidate amplification | \(N_{\mathrm{active}}^{adv}/N_{\mathrm{active}}^{clean}\) | Proxy cho workload NMS | CPU amplification/DoS |
| Measured NMS latency | \(t_{\mathrm{NMS}}\) | Wall-clock time của implementation cụ thể | Theoretical pair count |

---

## 6. Model-role table

| Vai trò | Kiến trúc | Input | NMS | Tối ưu? | Transfer test? | Trạng thái bằng chứng |
|---|---|---:|---|---|---|---|
| Source hiện tại | YOLOv8n (`yolov8n.pt`) | 320×320 trong scripts | Custom pure-PyTorch NMS ở `VictimModel` | Có | Tự kiểm tra | Xác nhận bằng code |
| Target exploratory | Faster R-CNN MobileNetV3-Large-FPN | 320×320 tensor trong script | Torchvision bundled | Không | Có | Timing không stage-aligned |
| Target exploratory | RetinaNet ResNet50-FPN | 320×320 tensor trong script | Torchvision bundled | Không | Có | Timing không stage-aligned |
| NMS-free control exploratory | DETR-ResNet50 | 320×320 tensor trong script | Không | Không | Có | Chỉ total forward; random input |
| MobileNetV2-SSD | Không có implementation tương ứng trong current source path | Chưa rõ | Chưa rõ | Chưa xác nhận | Claim cũ | **AUTHOR MUST VERIFY** |
| YOLOv5n | Không có weights/config/output trong repository hiện tại | Chưa rõ | NMS | Không | Claim cũ | **AUTHOR MUST VERIFY** |

---

## 7. Revised journal outline

1. Introduction
2. Related Work
3. Threat Model and Problem Formulation
4. Proposed Method
5. Experimental Methodology
6. Results
7. Ablation and Sensitivity Analysis
8. Transferability Evaluation
9. Physical-World Evaluation
10. Discussion
11. Defense Implications
12. Limitations and Threats to Validity
13. Ethical Considerations
14. Conclusion

Bản viết lại tương ứng nằm trong `BaoCaoKhoaHocAdversarialAttacks_Q1_revised.tex`.

---

## 8. Before → After: major claims

| Trước | Sau |
|---|---|
| “Patch ép mô hình sinh hàng nghìn bounding boxes.” | “Phương pháp tối ưu proxy là số ứng viên sau threshold; output hiện có chưa chứng minh hàng nghìn ứng viên từ detector.” |
| “NMS tăng 3.672×/5,94×.” | “Số cặp lý thuyết tăng 3.672,24×; latency phải đo riêng cho implementation.” |
| “Physical attack chứng minh DoS.” | “Physical transfer chưa được xác nhận bằng protocol/log tái lập; chỉ có EOT-simulated degradation trong code.” |
| “Saliency-Guided GA vượt trội hoàn toàn.” | “Fitness trung bình nhỉnh hơn 1,05 điểm, với độ phân tán chồng lấn; chưa có kiểm định/paired design.” |
| “Transferability sang kiến trúc khác.” | “Exploratory timings không cho thấy transfer nhất quán và có mismatch pipeline; kết luận inconclusive.” |
| “CPU 100% chứng minh attack.” | “CPU peak là telemetry hệ thống và không tự chứng minh causal NMS bottleneck hay QoS failure.” |
| “Giới hạn max detections giảm NMS workload.” | “Cap hiện tại được áp dụng sau NMS; cần pre-NMS top-k/candidate cap.” |

### Claims đã hạ cấp hoặc loại bỏ

- universal / architecture-independent vulnerability;
- industry-wide systemic flaw;
- definitive physical DoS;
- 3.000+ detector candidates;
- complete RAM exhaustion;
- CPU saturation as the primary success definition;
- statistically superior saliency guidance;
- validated defense effectiveness;
- YOLOv5n/YOLOv8n transfer percentages trong bảng cũ.

---

## 9. Reference audit

### Đã xác minh và nên giữ/cập nhật

- Brown et al., *Adversarial Patch*, arXiv:1712.09665 (arXiv-only).
- Athalye et al., *Synthesizing Robust Adversarial Examples*, ICML 2018.
- Shumailov et al., *Sponge Examples*, IEEE EuroS&P 2021.
- Shapira et al., *Phantom Sponges*, WACV 2023, pp. 4571–4580.
- Wang et al., *Daedalus*, IEEE Transactions on Cybernetics, 52(8), 7427–7440.
- Chen et al., *Overload*, CVPR 2024, pp. 24716–24725.
- Ma et al., *SlowTrack*, AAAI 2024, 38(5), 4062–4070, DOI 10.1609/aaai.v38i5.28200.
- Wang et al., *Can't Slow Me Down*, CVPR 2025, pp. 19230–19240.
- Monteuuis et al., *Latency NMS Attacks: Is It Real Life or Is It Just Fantasy?*, NeurIPS 2025, DOI 10.52202/085713-1289. Đây là công trình đặc biệt quan trọng vì trực tiếp chất vấn practicality của NMS latency attacks.

### Cần sửa metadata

- Ref 7: tác giả không phải “Y. Hong et al.”; DOI được tìm thấy là 10.1109/ACCESS.2024.3370849, pp. 33843–33851.
- Ref 8: không phải TPAMI 2024 theo dữ liệu tìm được; preprint arXiv:2203.08147 và publication metadata cần cập nhật theo bản cuối.
- Ref 9: tác giả đúng được tìm thấy là Andreas Müller và Erwin Quiring; venue IEEE Security and Privacy Workshops 2024, DOI 10.1109/SPW63631.2024.00016.
- Ref 13: không phải ICLR 2024; phiên bản publisher hiện là NeurIPS 2025.
- Ref 15: arXiv ID trong bản thảo cũ không khớp kết quả hiện tại; repository AdvReal trỏ tới arXiv:2505.16402. **[REFERENCE REQUIRES VERIFICATION]**
- Ref 21: metadata đầy đủ là Lin et al., IEEE Access 12, 32772–32779, DOI 10.1109/ACCESS.2024.3371507.
- Ref 22: tác giả là Jakob Shack, Katarina Petrovic, Olga Saukh; arXiv:2410.19863 / EMERGE 2024, không nên ghi chung “IEEE 2024”.

### Không xác minh được bằng exact-title search

- “Physical-World Adversarial Attacks against Object Detectors via Saliency-Guided Patch Optimization,” AAAI 2024. **[REFERENCE REQUIRES VERIFICATION]**

---

## 10. Experiments still required

### E01 — Strong prior-attack baseline (**Critical**)

**Marker:** `[HIGH PRIORITY Q1 EXPERIMENT]`  
**Why:** Không có so sánh ngang bằng với Phantom Sponges, Overload hoặc một NMS/resource-depletion attack mạnh.  
**Protocol:** cùng detector weights, image set, patch/perturbation budget, confidence/NMS settings, hardware, end-to-end timing, 10 independent seeds.  
**Metrics:** candidate amplification, NMS latency, total latency, deadline-miss rate, FPS, energy nếu có, query budget.  
**Table:** rows=attacks; columns=success proxy, latency/FPS, queries, wall-clock, visibility constraint.

### E02 — Reproducible edge-server benchmark (**Critical**)

**Why:** Không có run manifest cho claim phần cứng.  
**Protocol:** freeze git commit/model checksum; ≥10 independent run blocks/scenario; randomized clean/attack order; warm-up; background processes controlled.  
**Variables:** CPU model/core/thread/frequency, RAM, OS/kernel, Python/PyTorch/OpenCV/Ultralytics versions, backend, thread counts.  
**Metrics:** stage latency distribution, process CPU/RSS, system CPU/RAM, FPS, deadline misses.  
**Figure:** paired ECDF/box plots per stage; table mean±SD, median, IQR, 95% bootstrap CI.

### E03 — Optimizer ablation with equal query budget (**High**)

**Why:** GA/random/saliency hiện không cùng protocol.  
**Protocol:** same fixed natural image set, EOT samples, patch location search space, 500/1.000/2.500-query budgets; 10 seeds; paired seeds.  
**Baselines:** random search, standard GA fixed center, standard GA random location, image-saliency GA.  
**Metrics:** best-so-far fitness vs queries, candidates, NMS latency, total optimization time.

### E04 — Threshold sensitivity and fairness (**High**)

**Why:** τ=0,01 có thể tạo workload không đại diện deployment.  
**Protocol:** τ ∈ {0,01; 0,05; 0,10; 0,25; 0,50}, cùng clean/attack images; detector accuracy trên labeled dataset.  
**Metrics:** candidates, NMS/total latency, mAP/precision/recall.  
**Figure:** latency–accuracy Pareto curve.

### E05 — Defense evaluation (**High**)

**Why:** Các biện pháp hiện chỉ là đề xuất; `max_det` trong code là post-NMS.  
**Protocol:** implement pre-NMS top-k/candidate cap, adaptive threshold, frame backpressure; clean and attack; multiple caps.  
**Metrics:** attack latency/FPS and clean mAP/recall trade-off; 10 runs.  
**Table:** defense configuration × clean utility × attack resilience.

### E06 — Transferability including NMS-free models (**High**)

**Why:** Pipeline timing hiện không đồng nhất.  
**Protocol:** source YOLOv8n; targets YOLO family, SSD/Faster R-CNN/RetinaNet, DETR/RT-DETR; standardized natural images; each model’s documented preprocessing and synchronized timing; 10 seeds/patches.  
**Metrics:** candidate amplification where applicable, post-processing latency, total latency, transfer success.  
**Negative control:** NMS-free model.

### E07 — Printed physical patch matrix (**Critical nếu giữ “physical” trong title**)

**Why:** EOT simulation không thay cho physical capture.  
**Protocol:** ghi material/printer/dimensions; ≥3 distances × ≥3 yaw angles × ≥3 illumination levels × ≥2 camera/codecs; ≥10 trial blocks/condition; clean/random/checkerboard/sponge controls.  
**Metrics:** candidates, NMS/total latency, FPS/deadline misses, CPU/RAM; patch visibility/area ratio.  
**Figure:** robustness heatmaps; table mixed condition summary.

### E08 — Natural-scene and dataset validation (**Critical**)

**Why:** Nhiều output hiện dùng random tensors hoặc synthetic scenes.  
**Protocol:** labeled validation subset with diverse scenes; image-level splits; no reusing frames as independent seeds.  
**Metrics:** clean utility plus attack workload.

---

## 11. Q1-readiness checklist

- [x] Fitness direction verified from source.
- [x] NMS arithmetic corrected.
- [x] Terminology scheme defined.
- [x] Threat model downgraded to gray-box telemetry.
- [x] Negative results retained.
- [x] Manuscript structure rewritten.
- [ ] Hardware/software manifest captured per run.
- [ ] Natural dataset and exact scene set frozen.
- [ ] Strong prior-attack baseline under equal budget.
- [ ] Equal-query optimizer ablation with ≥10 seeds.
- [ ] Stage-level NMS profiling on ≥2 edge platforms.
- [ ] Accuracy–latency defense trade-off.
- [ ] Cross-architecture evaluation with aligned pipelines.
- [ ] NMS-free control.
- [ ] Printed physical robustness matrix.
- [ ] Trial-level statistics, CI and preregistered analysis.
- [ ] Every table/figure generated from immutable run IDs.
- [ ] Dependency lockfile, model hashes and reproduction command.
- [ ] Reference metadata fully verified.
- [ ] Open-source artifact stripped of simulated-as-real wording.

---

## 12. Remaining Gap to Q1-Level Submission

| Mức | Vấn đề | Reviewer concern | Hành động | Chỉ viết? | Cần experiment? | Tác động khoa học |
|---|---|---|---|---|---|---|
| **CRITICAL** | Claim chính không khớp log | Core conclusion unsupported | Rebuild evidence pipeline and rerun | Không | Có | Rất cao |
| **CRITICAL** | Không có SOTA baseline | Novelty/effectiveness unknown | Compare Phantom/Overload-like baseline | Không | Có | Rất cao |
| **CRITICAL** | Physical claim chưa có protocol | External/physical validity absent | Physical condition matrix | Không | Có | Rất cao |
| **HIGH** | Một hardware/platform | Generalization poor | ≥2 constrained platforms + one modern edge device | Không | Có | Cao |
| **HIGH** | Transfer pipeline sai lệch | Architecture claim invalid | Standardized end-to-end timing | Không | Có | Cao |
| **HIGH** | Statistics/units không độc lập | Significance unreliable | 10 paired seeds + run-block statistics | Không | Có | Cao |
| **HIGH** | Query efficiency chưa công bằng | Optimizer claim confounded | Equal budgets and anytime curves | Không | Có | Cao |
| **HIGH** | Defense chưa validated | Incomplete security paper | Pre-NMS cap and threshold trade-off | Không | Có | Cao |
| **MEDIUM** | Reproducibility artifact thiếu manifest | Cannot reproduce | Add configs, lockfile, hashes, run IDs | Một phần | Có để tái tạo số | Trung bình–cao |
| **MEDIUM** | Related work metadata lỗi | Trust/citation hygiene | Verify publisher metadata | Có | Không | Trung bình |
| **LOW** | Caption/format/language | Presentation quality | Neutral captions, consistent units | Có | Không | Thấp |

---

## 13. Reviewer #2 simulation

### Major comments

1. The central claim of an edge-server denial of service is not supported by traceable, controlled experiments. The repository mixes synthetic NMS microbenchmarks, random-image optimization, camera logs without manifests, and EOT-degraded digital patches.
2. The threat model is gray-box because optimization requires pre-NMS scores. The manuscript must not describe the method as strict black-box.
3. The source architecture is inconsistent. The released code optimizes YOLOv8n, whereas the manuscript repeatedly identifies MobileNetV2-SSD/MobileNetV2+YOLO as the source.
4. The claimed 3.000+ candidate explosion is not present in the supplied detector outputs. The paper appears to conflate a synthetic NMS benchmark at N=3.000 with detector-produced candidates.
5. The saliency ablation does not demonstrate a clear benefit. Confidence intervals or paired statistical analysis are absent, and the query budgets/protocols are not aligned.
6. The transferability experiment is methodologically invalid because different model wrappers include different pipeline stages. No general transfer claim is justified.
7. The physical result is not reproducible and may actually be an EOT/degradation simulation. A printed-patch evaluation matrix is required.
8. The defense discussion is not an evaluation. Moreover, the implemented `max_det` truncation occurs after NMS and therefore cannot mitigate the claimed NMS workload.
9. Recent work questioning the practical impact of NMS latency attacks must be discussed and experimentally addressed.
10. All results need run-level provenance, hardware/software manifests, model hashes and immutable raw outputs.

### Minor comments

- Use “active/post-threshold candidates,” not “raw boxes,” for threshold survivors.
- State whether thresholds are optimization, deployment or evaluation parameters.
- Replace “anchors” for YOLOv8n with “prediction locations/raw predictions.”
- Report patch area ratio and physical angular/normalized size, not only pixels.
- Correct EOT ranges to match implementation.
- Remove causal wording from CPU plots and captions.
- Fix corrupted LaTeX numbers and table cross-references.
- Pin exact dependency versions instead of minimum versions.
- Correct incomplete/incorrect bibliography entries.

### Likely rejection reasons

- unsupported main claim;
- inconsistent threat model and architecture;
- simulated evidence presented as physical/edge evidence;
- no strong baseline;
- invalid transferability methodology;
- insufficient statistics and reproducibility;
- negative baseline result contradicting the proposed method.

### Must resolve before submission

At minimum: E01, E02, E03, E04, E05 and E08. If “physical” remains in the title/abstract, E07 is mandatory. If “cross-architecture” remains a contribution, E06 is mandatory.

---

## 14. Năm tiêu đề journal-style đề xuất

1. **Gray-Box Candidate Amplification for Stress-Testing NMS-Based Edge Object Detectors**
2. **Saliency-Guided Genetic Optimization for Adversarial Workload Amplification in Edge Object Detection**
3. **Evaluating NMS Workload Amplification under Gray-Box Telemetry in Resource-Constrained Object Detectors**
4. **From Candidate Proliferation to Latency: A Gray-Box Study of Availability Risks in Edge Object Detection**
5. **Adversarial Patch Optimization for NMS Stress Testing on Resource-Constrained Vision Pipelines**

**Khuyến nghị:** dùng tiêu đề 1 hoặc 3 ở trạng thái bằng chứng hiện tại; chỉ dùng “Physical” sau khi E07 hoàn tất.

## 15. Cập nhật sau kiểm chứng cục bộ ngày 02/10/2026

Các mục phía trên kiểm toán bằng chứng cũ. Bản thảo tiếng Việt đã bổ sung hai lượt mới, không gộp chúng với số liệu thăm dò cũ:

- `outputs/local_q1_20261002_v2`: i5-1335U, 10 nhân/12 bộ xử lý logic, hai luồng PyTorch, YOLOv8n 320×320, miếng dán 64×64, 30 lượt tối ưu (ba phương pháp × 10 seed), 400 truy vấn/seed, 420 khối đo hai ảnh và 70 khối đo riêng NMS. Có phiên bản mô-đun đang nạp, mã băm mã nguồn, trọng số và ảnh.
- Fitness EOT tốt nhất đã quan sát: SG-GA 312,61 ± 10,65; GA trung tâm 367,39 ± 18,40; tìm kiếm ngẫu nhiên 309,97 ± 5,65. SG-GA không tốt hơn GA trung tâm trong cấu hình này. Điểm cực đại một mẫu EOT không phải kỳ vọng EOT được kiểm chứng độc lập.
- Không EOT tại ngưỡng 0,01: SG-GA có 115,4 ứng viên trên bus (sạch: 118) và 119,1 trên zidane (sạch: 122). Không có khuếch đại ứng viên hay tăng thời gian toàn vòng xử lý nhất quán. Khoảng tin cậy bootstrap của chênh lệch tổng thời gian đều bao gồm không.
- `outputs/local_coco128_20261002`: 128 ảnh, tám điều kiện × ba cấu hình = 3.072 bản ghi; miếng dán seed 0 quy định trước, 929 hộp nhãn/71 lớp. Tập train2017 có thể trùng tiền huấn luyện; AP là chẩn đoán 101 mức, không phải COCO chính thức. Hai ảnh thiếu tệp nhãn ngay trong ZIP nguồn được giữ với nhãn rỗng theo quy ước nạp dữ liệu.
- SG-GA có trung bình 114,78 ứng viên so với 116,06 sạch ở ngưỡng 0,01; AP50 chẩn đoán giảm 46,00% → 41,70%. Đây là suy giảm độ chính xác, không phải bằng chứng DoS. Top-100 giảm NMS nhưng làm độ bao phủ SG-GA giảm 52,85% → 47,58%.
- `paired_bootstrap.json`: 10.000 lần lấy mẫu lại ghép cặp, 35 tương phản; phân tích hậu nghiệm, không điều chỉnh đa so sánh, không có giá trị p xác nhận hay khoảng tin cậy AP.
- Họa tiết mang khóa `checkerboard` trong lượt hai ảnh thực chất là sọc chéo. Hình công bố sửa nhãn; COCO128 bổ sung ô bàn cờ 4×4 pixel đúng nghĩa. Dữ liệu gốc không bị sửa để thay kết quả.
- Bốn kiểm tra hồi quy đã qua: EOT không liên tục/gradient, NMS, bộ kiểm tra AP và bootstrap hằng số.

Các khoảng trống còn lại: tấn công đối chứng mạnh, tối ưu đa ảnh, tập kiểm tra độc lập, nhiều mô hình/phần cứng, tiêu chí chất lượng dịch vụ thiết kế trước, AP COCO chính thức và miếng dán in/camera. Khuyến nghị đánh giá Q1 chưa thay đổi. Bản LaTeX giữ mẫu CAS hai cột nhưng PDF chưa được xác nhận vì trình biên dịch trong ứng dụng báo lỗi thư mục chuẩn của nền tảng.
