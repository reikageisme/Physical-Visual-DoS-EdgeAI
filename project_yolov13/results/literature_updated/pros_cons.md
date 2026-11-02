# Ma trận ưu/nhược điểm và mức chứng cứ

| Thành phần / hướng đi | Ưu điểm có cơ sở | Hạn chế / đánh đổi | Mức chứng cứ và kiểm chứng cần có |
|---|---|---|---|
| HyperACE | Học nhóm tương quan nhiều vị trí và thang đo [1] | Lợi ích ngữ cảnh chưa chứng minh miễn nhiễm patch; chi phí phụ thuộc số hyperedge | Cơ chế và ablation của tác giả gốc; dự án cần ablation riêng |
| FullPAD | Đưa đặc trưng đã tổng hợp tới nhiều tầng [1] | Thêm nhánh/tunnel và phụ thuộc backend | Kiến trúc gốc; không coi là phát minh của dự án |
| DS-C3k2 | Depthwise-separable convolution giảm số phép tính/tham số trong ablation gốc [1] | FLOPs không đo băng thông bộ nhớ hay hiệu quả kernel; ít FLOPs vẫn có thể chậm | Lý thuyết và Table V gốc; cần đo pipeline trên thiết bị đích |
| YOLOv12 | Area Attention/R-ELAN hỗ trợ biểu diễn ngữ cảnh [2] | Backend attention và phiên bản Turbo/v1 ảnh hưởng tái lập | Bài gốc và repo; so sánh phải khóa phiên bản |
| YOLOv10 | Dual assignment hỗ trợ suy luận không NMS [3] | Loại NMS không tự chứng minh AP cao nhất cho person/patch | Bài gốc; cần cùng dataset và giao thức nếu bổ sung baseline |
| RT-DETR | Encoder đa thang và lựa chọn query; điều chỉnh số lớp decoder [4] | Dung lượng lớn hơn các nano được chạy; không cùng thiết lập đã đo | Nguồn gốc; chưa có benchmark địa phương |
| Tối ưu scheduler/loss/augmentation | Có thể thích ứng bài toán person và giảm bất ổn fine-tuning | Tìm nhiều cấu hình làm tăng nguy cơ overfit validation; augmentation quá mạnh có thể phá dữ liệu vật lý | Phải lưu mọi trial, chỉ dùng val chọn, khóa trước test |
| Sampling theo nguồn train | Có thể tăng tần suất nhìn thấy nguồn Adv ít ảnh | Oversampling có thể lặp cảnh/người và tăng lệch phân phối | Giả thuyết của dự án; cần ablation và chỉ resample train |
| Thực nghiệm nhiều seed | Cho biết biến thiên do tối ưu ngẫu nhiên | CI theo seed không đo bất định do split/đối tượng/dataset | Kết quả địa phương; dùng Student-t CI và paired difference khi hợp lý |
| Dữ liệu gộp COCO + Adv | Có cả cảnh phổ thông và áo patch vật lý [5] | Pretrain exposure và split ảnh; pooled AP chưa đo causal defense | Cần báo cáo riêng nguồn, subject/sequence split và thử nghiệm ngoài phân phối |

Nguồn:

[1] [Lei et al., YOLOv13, Sec. III và Table V](https://arxiv.org/html/2506.17733v1).

[2] [Tian et al., YOLOv12 v1](https://arxiv.org/html/2502.12524v1).

[3] [Wang et al., YOLOv10](https://arxiv.org/html/2405.14458v1).

[4] [Zhao et al., RT-DETR](https://arxiv.org/abs/2304.08069).

[5] [Wang et al., BadPatch v5, Sec. 4.5](https://arxiv.org/html/2412.01440v5).

Phần suy luận về sampling, chọn cấu hình và giới hạn CI là phân tích phương pháp của dự án, không phải kết quả đo đã có. Không chuyển giá trị công bố thành giá trị thực nghiệm địa phương.
