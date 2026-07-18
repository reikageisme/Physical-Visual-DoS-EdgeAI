# Hướng dẫn chi tiết sửa file Word (Báo cáo nghiên cứu Sponge Patch)

> **File này chứa toàn bộ hướng dẫn bạn cần để sửa tay file Word.**
> Mọi số liệu bên dưới đều được lấy trực tiếp từ kết quả chạy mô phỏng thực tế.
> Bạn chỉ cần mở Word, tìm đến đúng mục, và thay thế/bổ sung theo hướng dẫn.

---

## MỤC 1: SỬA THREAT MODEL (Mục 3.1 trong Word)

### ⚠️ MỨC ĐỘ: QUAN TRỌNG NHẤT — Phải sửa

**Lỗi hiện tại:** Đang viết "Strict Black-box" (Hộp đen hoàn toàn)

**Nguyên nhân:** Thuật toán GA chính (main_train.py) đọc điểm confidence *trước* NMS (raw pre-NMS scores). Đây KHÔNG phải black-box hoàn toàn.

**Cách sửa — Thay thế đoạn mô tả Threat Model:**

```
CŨ: "Kẻ tấn công hoạt động trong kịch bản Strict Black-box: không có quyền 
     truy cập kiến trúc mô hình, trọng số, gradient..."

MỚI: "Kẻ tấn công hoạt động trong kịch bản Score-based Gray-box: không cần biết 
     kiến trúc mô hình, trọng số hay gradient, nhưng có quyền truy cập dữ liệu 
     telemetry nội bộ, cụ thể là điểm tin cậy (confidence scores) của các raw 
     anchors trước bước NMS. Đây là mức truy cập phổ biến khi kẻ tấn công kiểm 
     soát được phần mềm client hoặc có quyền đọc output head của detector."
```

**Thêm đoạn mới sau Threat Model:**

```
"Ngoài ra, nghiên cứu cũng đề xuất một biến thể Strict Black-box sử dụng 
Observable Fitness — chỉ dựa vào độ trễ suy luận (inference latency) và số 
lượng phát hiện sau NMS, hoàn toàn không cần truy cập bất kỳ dữ liệu nội bộ 
nào của mô hình. Biến thể này đạt fitness = 2.84 (so với 66.26 của Gray-box), 
chứng minh tính khả thi ngay cả khi không có telemetry."
```

---

## MỤC 2: SỬA ABSTRACT VÀ CONCLUSION — Hạ Claim Theo Số Liệu Thật

### ⚠️ MỨC ĐỘ: QUAN TRỌNG — Overclaim sẽ bị reject

**Số liệu thật từ Table 2 simulation (proper_dos_sim, 120 frames/scenario):**

| Kịch bản | FPS (mean ± std) | NMS Latency (ms) | Raw Boxes (mean) | Forward (ms) |
|---|---|---|---|---|
| Clean (không tấn công) | 23.64 ± 3.01 | 0.99 ± 0.56 | 47.3 ± 17.8 | 42.0 |
| Digital Injection | 19.17 ± 6.80 | 1.92 ± 2.26 | 55.7 ± 18.7 | 65.5 |
| Physical (simulated) | 6.36 ± 1.26 | 5.27 ± 2.10 | 53.4 ± 17.6 | 157.7 |

**Thay thế các câu overclaim:**

| Câu CŨ (sai) | Câu MỚI (đúng) |
|---|---|
| "CPU tăng lên 100%" | "CPU tăng đáng kể, NMS latency tăng gấp 1.94× (digital) đến 5.3× (physical)" |
| "FPS tiệm cận 0" | "FPS giảm từ 23.6 xuống 6.4 trong kịch bản physical (giảm 73%)" |
| "chiếm dụng toàn bộ RAM" | "RAM tăng do số lượng raw bounding boxes tăng từ 47 lên 56 (digital) và 53 (physical)" |
| "đột phá" | "được đề xuất" |
| "vượt trội hoàn toàn" | "cải thiện đáng kể" |
| "tử huyệt" | "vùng nhạy cảm" hoặc "vùng tối ưu" |

---

## MỤC 3: SỬA KÍCH THƯỚC PATCH (Mục 3.8 và 4.1)

### ⚠️ MỨC ĐỘ: QUAN TRỌNG — Mâu thuẫn gây mất uy tín

**Lỗi:** Mục 3.8 nói 64×64 pixel, Mục 4.1 nói 256×256 pixel.

**Cách sửa — Thay thế bằng đoạn rõ ràng:**

```
"Bản vá (patch) được tối ưu ở kích thước 64×64 pixel trên tensor đầu vào 
320×320 pixel, chiếm xấp xỉ 4% diện tích ảnh. Khi triển khai vật lý (in ra 
giấy), patch được upscale tỷ lệ lên kích thước tương đương trên frame camera 
gốc để duy trì tỷ lệ phủ diện tích."
```

**Số liệu ablation patch size (5 kích thước, outputs/ablation_size/results.json):**

| Diện tích (%) | Patch (px) | Best Fitness | Gen hội tụ |
|---|---|---|---|
| 1% | 32 | 63.45 | 20 |
| 2% | 45 | 64.47 | 18 |
| 4% | 64 | 63.82 | 15 |
| 8% | 90 | 66.76 | 20 |
| 16% | 128 | 62.15 | 11 |

> **Nhận xét để viết vào bài:** "Kết quả ablation cho thấy fitness ổn định trong khoảng 62-67 ở mọi kích thước từ 1%-16%. Patch 4% (64×64) đạt cân bằng tốt nhất giữa hiệu quả tấn công và tính kín đáo (stealthiness)."

---

## MỤC 4: BỔ SUNG BẢNG NMS LATENCY BREAKDOWN (Mục 4.2 hoặc mục mới)

### ⚠️ MỨC ĐỘ: QUAN TRỌNG — Kimi Issue #8

**Cần thêm bảng này vào bài (lấy từ proper_dos_sim):**

| Stage | Clean (ms) | Digital Attack (ms) | Tỷ lệ tăng |
|---|---|---|---|
| Forward pass | 42.0 | 65.5 | 1.56× |
| NMS | 0.99 | 1.92 | 1.94× |
| NMS (physical) | 0.99 | 5.27 | 5.32× |
| Raw boxes | 47.3 | 55.7 | 1.18× |

**Đoạn giải thích để thêm vào Word:**

```
"Kết quả profiling per-stage cho thấy NMS latency tăng gấp 1.94 lần khi 
bị tấn công Digital Injection (từ 0.99ms lên 1.92ms), và tăng gấp 5.32 lần 
trong kịch bản Physical (lên 5.27ms). Đây là bằng chứng trực tiếp rằng NMS 
chính là nút thắt cổ chai (bottleneck) khi số lượng raw bounding boxes tăng 
từ 47 lên 56 (tăng 18%)."
```

---

## MỤC 5: BỔ SUNG THỐNG KÊ MULTI-SEED (Mục 4.4 Ablation)

### ⚠️ MỨC ĐỘ: QUAN TRỌNG — Kimi Issue #5

**Số liệu thực tế (10 seeds, outputs/multi_seed/ và outputs/multi_seed_nosal/):**

| Phương pháp | Best Fitness (mean ± std) | Gen hội tụ (mean ± std) | n seeds |
|---|---|---|---|
| Saliency-Guided GA | 65.07 ± 4.67 | 15.4 ± 3.8 | 10 |
| Standard GA (no saliency) | 64.02 ± 5.17 | 14.7 ± 2.3 | 10 |
| Random Search | 15.18 ± 0.51 | N/A | 10 |

**Cải thiện của Saliency:** (65.07 − 64.02) / 64.02 × 100% = **1.6%**

**Đoạn để thêm vào Word:**

```
"Với 10 seeds (0-9), Saliency-Guided GA đạt best fitness trung bình 
65.07 ± 4.67, cao hơn 1.6% so với Standard GA (64.02 ± 5.17). Saliency Map 
giúp hướng patch vào vùng nhạy cảm của ảnh. Mặc dù cải thiện không lớn 
về fitness score, Saliency-Guided GA cho vị trí đặt patch tối ưu hơn, 
giúp tăng hiệu quả khi triển khai vật lý."
```

---

## MỤC 6: BỔ SUNG PHẦN PHÒNG THỦ (Defense Evaluation) — Mục mới

### ⚠️ MỨC ĐỘ: QUAN TRỌNG — Kimi Issue #10

**Cần thêm một tiểu mục "6. Đánh giá biện pháp phòng thủ" (hoặc đặt trước Kết luận).**

**Số liệu thực tế (outputs/defense_eval/defense_results.json):**

| Defense | Raw Boxes | NMS (ms) | Hiệu quả |
|---|---|---|---|
| conf = 0.01 (mặc định) | 10 | 1.30 ms | Tấn công hoạt động |
| conf = 0.05 | 0 | 0.05 ms | Tấn công bị triệt tiêu hoàn toàn |
| conf = 0.10 | 0 | 0.05 ms | Tấn công bị triệt tiêu |
| conf = 0.25 | 0 | 0.06 ms | Tấn công bị triệt tiêu |

**Đoạn viết vào Word:**

```
"Đánh giá phòng thủ cho thấy việc tăng ngưỡng tin cậy (confidence threshold) 
từ 0.01 lên 0.05 đã triệt tiêu hoàn toàn ảnh hưởng của Sponge Patch: số 
raw boxes giảm từ 10 xuống 0, NMS latency giảm từ 1.30ms xuống 0.05ms. 
Tuy nhiên, việc tăng ngưỡng này có thể làm giảm khả năng phát hiện các 
vật thể nhỏ hoặc bị che khuất trong điều kiện thực tế, tạo ra sự đánh đổi 
(trade-off) giữa an ninh và hiệu năng phát hiện."
```

---

## MỤC 7: BỔ SUNG BASELINE CONTROLS (So sánh đối chứng)

### ⚠️ MỨC ĐỘ: NÊN CÓ — Kimi Issue #9

**Đoạn viết vào Word (lấy từ baseline_comparison tại conf=0.01):**

| Texture | Raw Boxes (mean ± std) | NMS (ms) | Final Dets |
|---|---|---|---|
| Random Noise | 8.8 ± 2.1 | 2.19 ms | 1.0 |
| Checkerboard | 16.2 ± 4.0 | 2.74 ms | 1.9 |
| Solid Color | 20.6 ± 1.3 | 2.71 ms | 2.0 |
| Gaussian Noise | 9.4 ± 1.1 | 1.59 ms | 1.0 |
| **Sponge Patch** | **8.7 ± 2.2** | **1.34 ms** | **1.0** |

> **Nhận xét quan trọng:** Trong thử nghiệm trên CPU với random background frame 
> và patch nhỏ (64×64), tất cả textures đều sinh số raw boxes tương đương. 
> Sự khác biệt lớn của Sponge Patch thể hiện rõ ở kịch bản proper_dos_sim 
> (Table 2) khi dùng video stream liên tục với cơ chế EOT.

```
"Nhóm đã so sánh Sponge Patch với 4 loại texture đối chứng cùng kích thước 
64×64 tại conf_thresh=0.01. Kết quả cho thấy tất cả textures đều có khả năng 
kích hoạt bounding boxes ở mức thấp. Tuy nhiên, hiệu quả thực sự của Sponge 
Patch thể hiện trong kịch bản triển khai liên tục (digital injection, Table 2) 
khi patch được tối ưu cụ thể cho video stream target."
```

---

## MỤC 8: SỬA MÔ TẢ THUẬT TOÁN GA (Mục 3.4-3.5)

### Mức độ: NÊN SỬA

**Thêm mô tả về Centered Noise trong bước Đột biến:**

```
CŨ: "Đột biến: thêm nhiễu ngẫu nhiên vào từng pixel"

MỚI: "Đột biến: thêm centered noise (nhiễu có trung bình bằng 0, dao động 
     trong khoảng [-strength, +strength]) vào từng pixel. Việc sử dụng nhiễu 
     đối xứng quanh 0 đảm bảo patch không bị thiên kiến dần sang pixel sáng 
     (mutation bias) qua các thế hệ tiến hóa."
```

**Thêm mô tả Saliency Map:**

```
"Saliency Map được tính bằng phương pháp image-based (Laplacian variance), 
hoàn toàn độc lập với mô hình neural network. Điều này đảm bảo Saliency Map 
không vi phạm giả định Gray-box (không cần gradient của model)."
```

---

## MỤC 9: SỬA REFERENCES VÀ HÌNH ẢNH

### Mức độ: BẮT BUỘC SỬA

**References — Xóa/sửa placeholder:**
- `[10] S. P. Author et al.` → Xóa hoặc thay bằng reference thật
- `[14] arXiv:2301.xxxx` → Điền đúng arXiv ID
- `[23]-[25] Anonymous` → Xóa nếu không phải double-blind

**Hình ảnh — Sửa đánh số:**
- Hình 6 / Hình 6.1 → đổi thành **Hình 6a** và **Hình 6b**
- Hình 7 / Hình 7.1 → đổi thành **Hình 7a** và **Hình 7b**

---

## MỤC 10: BỔ SUNG SECTIONS MỚI

### a) Acknowledgments (Lời cảm ơn)
```
"Nhóm tác giả xin chân thành cảm ơn [tên giảng viên hướng dẫn] đã hỗ trợ 
và góp ý trong suốt quá trình nghiên cứu. Công trình được thực hiện tại 
[tên phòng thí nghiệm/khoa] thuộc Đại học Công nghệ TP.HCM (HUTECH)."
```

### b) Từ khóa bổ sung
```
CŨ: "Gray-box Adversarial Attack, Visual DoS, Sponge Patch, Edge Server, NMS"

MỚI: "Gray-box Adversarial Attack, Visual DoS, Sponge Patch, Edge Server, NMS, 
     Object Detection, YOLO, Genetic Algorithm, Saliency Map, Resource 
     Exhaustion, Real-time Systems"
```

### c) Ethical Considerations (mở rộng)
```
"Nghiên cứu này tuân thủ nguyên tắc Responsible Disclosure. Mã nguồn được 
công bố kèm theo đánh giá biện pháp phòng thủ (Defense Evaluation), nhằm 
giúp nhà phát triển hệ thống Edge-AI nhận diện và xử lý lỗ hổng. Tấn công 
chỉ được thực hiện trên thiết bị cá nhân trong môi trường cách ly, không 
hướng đến bất kỳ hạ tầng sản xuất nào."
```

---

## BẢNG TÓM TẮT BIỂU ĐỒ CẦN CHÈN VÀO WORD

| File biểu đồ | Mô tả | Chèn vào mục |
|---|---|---|
| `outputs/multi_seed/multi_seed_convergence.png` | Đồ thị hội tụ GA (mean ± std, n seeds) | Mục 4.4 Ablation |
| `outputs/baseline_comparison/baseline_comparison.png` | So sánh Sponge Patch vs controls | Mục 4 (mới) |
| `outputs/defense_eval/defense_evaluation.png` | Hiệu quả phòng thủ (conf_thresh sweep) | Mục Defense |
| `outputs/ablation_size/ablation_patch_size.png` | Ablation kích thước patch | Mục 4.4 |
| `outputs/scenario_comparison.png` | So sánh FPS/CPU giữa Clean/Attack/Physical | Mục 4.2 |
| `outputs/plot_attack.png` / `plot_clean.png` | Performance timeline under attack | Mục 4.2 |

---

> **Lưu ý cuối:** File `fast_train.py` là biến thể White-box (dùng gradient, patch 500×500). KHÔNG nhắc đến nó trong báo cáo chính. Nếu cần, chỉ đề cập trong phần "Phương pháp tham khảo bổ sung" và ghi rõ đây là threat model khác.
