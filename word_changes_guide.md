# Hướng dẫn sửa file Word (Báo cáo nghiên cứu)

Chào bạn, dựa trên các lỗi đã fix trong code và các phân tích từ file `REVIEW_SPONGE_PATCH_DETAILED.md` của Kimi, bạn cần **sửa tay** lại file Word báo cáo của mình ở các phần sau để bài viết hoàn toàn khớp với code và giải quyết triệt để các câu hỏi phản biện:

## 1. Sửa Mô hình mối đe dọa (Threat Model) - RẤT QUAN TRỌNG
- **Lỗi hiện tại trong Word:** Đang viết là "Tấn công Hộp đen hoàn toàn" (Strict Black-box) nhưng thực tế thuật toán GA (ở `main_train.py`) sử dụng điểm confidence *trước* khi qua NMS, tức là hộp xám (Gray-box).
- **Cách sửa trong Word:** 
  - Đổi các từ "Black-box" / "Hộp đen" mô tả thuật toán GA chính thành **"Gray-box" (Hộp xám)** hoặc **"Score-based Black-box"**.
  - **Thêm đoạn mới:** Đề cập rằng nhóm nghiên cứu đã bổ sung thêm một kịch bản **Hộp đen thực sự (Strict Black-box)** dựa hoàn toàn vào độ trễ (Latency) thay vì điểm confidence, chứng minh được tính khả thi trên thực tế. (Code cho phần này đã được mình viết ở file `train_blackbox.py`).

## 2. Sửa Kích thước Patch (Patch Size) - Đồng nhất số liệu
- **Lỗi hiện tại trong Word:** Có chỗ ghi 64x64, có chỗ lại ghi 256x256 khiến người đọc bị rối.
- **Cách sửa trong Word:** 
  - Khẳng định lại kích thước của bản vá (Sponge Patch) trong thuật toán tối ưu trên máy tính là **64x64 pixel**.
  - Kích thước **256x256 pixel** hoặc lớn hơn chỉ là kích thước **được phóng to (upscale)** để in ra giấy A4 dùng cho thử nghiệm vật lý (Physical Attack). Cần ghi rõ: *"Bản vá 64x64 được upscale lên để in ra giấy"*.

## 3. Thêm phần Phân tích Độ trễ NMS (NMS Latency Profiling)
- **Lỗi hiện tại trong Word:** Chỉ nói "làm tăng độ trễ" nhưng không tách biệt thời gian chạy model (Forward pass) và thời gian chạy NMS. NMS chạy trên CPU nên dễ bị thắt cổ chai.
- **Cách sửa trong Word:** 
  - Bổ sung một câu giải thích rõ: *"Sponge Patch không làm chậm quá trình quét ảnh (Forward pass) của mạng nơ-ron, mà tạo ra hàng ngàn bounding boxes ảo (Raw boxes), từ đó làm quá tải thuật toán Non-Maximum Suppression (NMS) ở khâu hậu xử lý (Post-processing), khiến FPS giảm mạnh."*

## 4. Chỉnh sửa mô tả Thuật toán Di truyền (Genetic Algorithm)
- **Cần thêm vào Word:** Khi mô tả bước Đột biến (Mutation), hãy nhấn mạnh rằng nhóm đã sử dụng **"Centered Noise"** (Nhiễu có trung bình bằng 0, dao động từ `-strength` đến `+strength`). Điều này giúp tránh hiện tượng "Mutation Bias" (thiên kiến làm sáng ảnh dần theo từng thế hệ). 

## 5. Bổ sung phần Đánh giá Phòng thủ (Defense Evaluation)
- **Cần thêm vào Word:** Phản biện có thể hỏi: *"Làm sao để chống lại tấn công này?"*. Trong Word bạn nên thêm một tiểu mục "Biện pháp phòng thủ (Mitigations)".
  - Nêu ra 2 cách đã được code mô phỏng kiểm chứng (`experiments/defense_evaluation.py`):
    1. **Tăng Confidence Threshold** (ví dụ từ 0.01 lên 0.1): Sẽ lọc bớt các hộp rác do Patch sinh ra.
    2. **Giới hạn số lượng Max Detections** (max_det cap = 300): Chặn không cho NMS xử lý quá 300 hộp.
  - **Kết luận:** Mặc dù các biện pháp này chống lại được Sponge Patch, nhưng nó có thể làm giảm khả năng nhận diện các vật thể nhỏ hoặc bị che khuất ở điều kiện bình thường (Trade-off).

---

> **Lưu ý:** Code mới cập nhật đã sửa toàn bộ các lỗi liên quan đến API khiến Web và Raspberry Pi bị crash. Các experiments cũng đã được đóng gói chuẩn chỉnh và xuất ra biểu đồ trong folder `outputs/`. Bạn hãy xem các biểu đồ trong đó để copy vào Word nếu cần!
