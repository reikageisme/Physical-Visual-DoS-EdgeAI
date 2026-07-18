# Hướng dẫn chi tiết sửa file Word (Báo cáo nghiên cứu)

Dựa trên file đánh giá `REVIEW_SPONGE_PATCH_DETAILED.md` của Kimi và các đoạn code mình vừa chạy mô phỏng, dưới đây là **danh sách chi tiết từng mục bạn cần sửa bằng tay** trong file Word.

Hãy mở file Word của bạn lên, tìm đến các mục tương ứng và sửa lại theo hướng dẫn này để bài báo của bạn đạt điểm tối đa nhé:

## 1. Sửa Mô hình mối đe dọa (Threat Model) - [Major Issue 1]
- **Chỗ cần sửa:** Các câu nhắc đến "Tấn công Hộp đen hoàn toàn" (Strict Black-box) trong mô tả thuật toán GA.
- **Sửa thành:** Đổi thành **"Hộp xám" (Gray-box)** hoặc **"Score-based Black-box"**. Kẻ tấn công không biết kiến trúc mô hình nhưng đọc được điểm confidence trước NMS.
- **Thêm vào báo cáo:** Hãy thêm 1 đoạn nói rằng nhóm đã đề xuất thêm biến thể **Hộp đen thực sự (Strict Black-box)** dựa hoàn toàn vào độ trễ (Latency). Bằng chứng là file `train_blackbox.py` và kết quả `outputs/blackbox/` vừa được đẩy lên Github.

## 2. Sửa Cụm từ "Physical Attack" - [Major Issue 2]
- **Chỗ cần sửa:** Tiêu đề, Tóm tắt (Abstract) và Kết luận (Conclusion).
- **Sửa thành:** Giảm nhẹ độ "nguy hiểm" của tấn công vật lý. Hãy khẳng định: *"Tấn công Digital Injection (chèn ảnh số) có thể gây suy giảm tài nguyên mạnh mẽ, còn tấn công Physical (in ra giấy) đã cho thấy tính khả thi sơ bộ nhưng chưa làm treo máy hoàn toàn (CPU chỉ lên ~60%)"*.

## 3. Sửa số liệu CPU/RAM/FPS - [Major Issue 3]
- **Chỗ cần sửa:** Tóm tắt (Abstract) và Kết luận (Conclusion).
- **Lỗi hiện tại:** Đang ghi "CPU 100%, chiếm dụng toàn bộ RAM, FPS tiệm cận 0". Con số này là overclaim (nói quá) so với thực tế ở Bảng 2.
- **Sửa thành:** *"Tấn công làm FPS giảm xuống mức 5-11 FPS (so với 15-21 FPS ban đầu), đẩy mức tiêu thụ CPU lên 67-78% và chiếm dụng 4.8 GB RAM."* Tránh dùng từ "tiệm cận 0" hay "100%" nếu đó chỉ là mức Peak (đỉnh) trong khoảnh khắc.

## 4. Đồng nhất Kích thước Patch - [Major Issue 4]
- **Chỗ cần sửa:** Phần mô tả thực nghiệm.
- **Sửa thành:** Ghi rõ thuật toán chạy tối ưu trên kích thước **64x64 pixel** (chiếm ~4% ảnh đầu vào). Kích thước **256x256 pixel** chỉ là bản vá được upscale (phóng to) để in ra giấy A4 cho thực nghiệm vật lý ngoài đời thực.

## 5. Thêm phần Phân tích Độ trễ NMS - [Major Issue 8]
- **Chỗ cần sửa:** Phần mô tả NMS Bottleneck.
- **Thêm vào:** Thêm câu: *"Sponge Patch không làm chậm quá trình quét ảnh của Neural Network (Forward pass), mà tạo ra hàng ngàn bounding boxes ảo (Raw boxes), làm quá tải khâu NMS."* 
- **Bằng chứng:** Bạn có thể chèn ảnh biểu đồ phân tách độ trễ ở file `outputs/baseline_comparison/baseline_comparison.png` vào Word.

## 6. Sửa lỗi thiên kiến của Thuật toán (Mutation Bias) - [Issue Code]
- **Chỗ cần sửa:** Mô tả thuật toán Genetic Algorithm (GA), phần Đột biến (Mutation).
- **Thêm vào:** Ghi chú rằng nhiễu đột biến là **"Centered noise" (Nhiễu trung tâm, dao động từ -strength đến +strength, trung bình = 0)**. Điều này giúp độ sáng của patch không bị tăng ảo qua các thế hệ.

## 7. Bổ sung phần Đánh giá Phòng thủ (Defense Evaluation) - [Major Issue 10]
- **Chỗ cần sửa:** Phần cuối bài, trước kết luận hoặc trong "Future Work".
- **Thêm vào:** Thêm một tiểu mục "Biện pháp phòng thủ (Mitigations)".
  - Bạn giải thích: *Có thể chống lại Sponge Patch bằng cách tăng Confidence Threshold (ví dụ 0.1) hoặc giới hạn Max Detections (ví dụ 300).* 
  - Đưa số liệu từ `outputs/defense_eval/defense_results.json` vào: *"Khi tăng conf_thresh lên 0.05 hoặc giới hạn max_det, ảnh hưởng của Sponge Patch bị triệt tiêu hoàn toàn."*

## 8. Bổ sung Baseline Controls (So sánh đối chứng) - [Major Issue 9]
- **Chỗ cần sửa:** Phần đánh giá kết quả (Evaluation).
- **Thêm vào:** *"Để chứng minh Sponge Patch đặc biệt, chúng tôi đã so sánh nó với các họa tiết ngẫu nhiên (Random Noise, Checkerboard, Gaussian Noise). Kết quả cho thấy các họa tiết này sinh ra 0.0 hộp (Raw boxes), trong khi Sponge Patch sinh ra số lượng hộp khổng lồ."*
- **Bằng chứng:** Chèn kết quả từ Bảng `comparison_results.json` ở folder `outputs/baseline_comparison/`.

## 9. Sửa lỗi văn phong và References - [Minor Issues]
- Xoá các câu quá "đao to búa lớn" (vd: "tử huyệt", "vượt trội hoàn toàn", "chính thức làm suy giảm nghiêm trọng"). Thay bằng các từ trung tính mang tính học thuật ("cải thiện đáng kể", "vùng nhạy cảm", "gây suy giảm").
- Sửa các tài liệu tham khảo (References) bị ghi là `Anonymous` hoặc `arXiv:2301.xxxx` thành link bài báo hoặc DOI chuẩn.
- Đánh số lại các Hình 6.1, 7.1 thành Hình 6a, Hình 7a.

---

> **Lưu ý:** Tất cả file code, file biểu đồ `png`, và file kết quả `json` phục vụ cho việc điền số liệu vào các mục này đều đã được mình tạo và push lên Github của bạn. Bạn chỉ việc mở folder `outputs/` trên Github, xem hình và copy số điền vào Word là xong!
