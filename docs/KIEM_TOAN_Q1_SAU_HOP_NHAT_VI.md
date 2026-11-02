# Kiểm toán khoa học sau hợp nhất bản thảo tiếng Việt

Ngày đối chiếu: 02/10/2026. Bản khoa học được đối chiếu tại commit `60b2c5d8f8e0aaeab417e5c529d56c895fb0afdd`.

Báo cáo này là tài liệu phản biện cục bộ, không đưa lên Drive/Git. Nó thay trạng thái hiện tại trong báo cáo kiểm toán cũ, không xóa bằng chứng hoặc sửa số đo. Tên tiếng Anh của bài báo, mô hình và hàm mã nguồn được giữ nguyên khi cần nhận diện; phần diễn giải dùng tiếng Việt.

## 1. Phạm vi và kết luận điều hành

Bản thảo hiện tại gồm `BaoCaoKhoaHocAdversarialAttacks_Q1_VI.tex` và bản Word hợp nhất `BaoCaoKhoaHocAdversarialAttacks_HopNhat_VI.docx`. Word có 23 trang A4, 9 hình, 8 bảng, 9 khối công thức trình bày và 157 đối tượng công thức gốc chỉnh sửa được. Bản Word gốc được bảo toàn; không ghép các bảng CPU/FPS/chuyển giao thiếu căn cứ vào kết quả đã kiểm chứng. TeX giữ mẫu CAS hai cột; PDF TeX chưa được xác nhận vì trình biên dịch của ứng dụng báo `Unable to find standard directories for platform`.

Bằng chứng mới gồm 30 lượt tối ưu (3 phương pháp × 10 seed), mỗi lượt 400 truy vấn, 420 khối đánh giá trên hai ảnh tự nhiên và 70 khối đo riêng NMS; đánh giá COCO128 có 3.072 bản ghi, 128 ảnh và 929 hộp nhãn thuộc 71 lớp. Seed là giá trị khởi tạo bộ sinh ngẫu nhiên; không xem lượt đo lặp hoặc khung liên tiếp là seed độc lập.

Kết luận phù hợp là một nghiên cứu kiểm chứng quy trình và giới hạn của mục tiêu đại diện. Chưa đủ căn cứ để gọi là DoS thành công, ưu thế của vị trí nổi bật, chuyển giao giữa kiến trúc hay DoS vật lý. Theo định vị bài tấn công mới, khuyến nghị hiện tại là chưa nộp với các tuyên bố hiệu lực đó. Có thể phát triển định vị kiểm chứng/kết quả âm, nhưng vẫn cần đối chứng mạnh, tập kiểm tra độc lập và đánh giá triển khai thích hợp; không bảo đảm tạp chí Q1 sẽ chấp nhận.

Nguồn kiểm tra trực tiếp:

- `outputs/local_q1_20261002_v2/manifest.json`, `aggregate.json`, `evaluation_seed_*.json`, `nms_samples.json`, `paired_bootstrap.json`.
- `outputs/local_coco128_20261002/manifest.json`, `aggregate.json`, `image_*.json`, `summary.csv`.
- `attack/genetic_algo.py`, `core/victim_model.py`, `core/sponge_fitness.py`, `core/eot_transforms.py` và các chương trình kiểm chứng trong `experiments/`.
- Nguồn SVG và thông tin nguồn ảnh trong `figures/`; bản kê biểu đồ trong các thư mục `figures_corrected`.
- 4 kiểm tra hồi quy: EOT với tensor không liên tục/gradient, NMS trên cấu hình hộp kiểm tra, ghép hộp/nội suy AP, bootstrap chênh lệch hằng số. Các kiểm tra này không chứng minh toàn bộ chất lượng mô hình hoặc tính hợp lệ vật lý.

## 2. Bảng kiểm toán phản biện chính

Mức độ đánh giá: nghiêm trọng, cao, vừa, thấp. Cột cuối phân biệt sửa diễn giải với việc tạo bằng chứng cho tuyên bố mạnh hơn. Số mục là của bản hợp nhất, không phải bản Word gốc.

| Mã | Mục/trang Word | Phát biểu hoặc trạng thái được kiểm tra | Loại vấn đề | Mức | Vì sao phản biện có thể bác | Hiệu chỉnh/trạng thái hiện tại | Sửa không cần thí nghiệm? |
|---|---|---|---|---|---|---|---|
| A01 | 1, 6, 14; trang 1, 12–17, 22 | Miếng dán gây DoS | Chưa chứng minh mục tiêu hệ thống | Nghiêm trọng | Số ứng viên và tổng thời gian không tăng nhất quán; chưa định nghĩa tiêu chí trễ hạn | Đã hạ thành kiểm chứng mục tiêu đại diện, giữ kết quả âm | Có để hạ tuyên bố; không để chứng minh DoS |
| A02 | 4.3, 6.2; trang 5, 12 | 50 → 3.000 hộp tương ứng 5,94 lần | Sai toán học trong bản cũ | Nghiêm trọng | Tỷ số đúng của số cặp là 3.672,24 | Đã sửa; độ trễ trung bình đo riêng có tỷ số khoảng 141,1, không gộp hai đại lượng | Có; đã sửa |
| A03 | 4.2, 5.2; trang 4–5, 8 | Gọi số sau ngưỡng là dự đoán thô | Thuật ngữ/đại lượng | Cao | Dự đoán thô, sau ngưỡng, vào NMS và sau NMS khác nhau | Đã định nghĩa, nêu top-100 trước NMS và 300 sau NMS | Có; đã sửa |
| A04 | 5.2; trang 8 | MobileNetV2-SSD là mô hình nguồn | Không khớp triển khai | Nghiêm trọng | Mã và đầu ra thực chạy là YOLOv8n | Đã dùng YOLOv8n, đầu vào 320 × 320, 2.100 vị trí dự đoán | Có; đã sửa |
| A05 | 4.5, 6.1, 6.3; trang 5, 11, 13 | Fitness nhỏ hơn tốt hơn | Sai chiều tối ưu trong diễn giải cũ | Nghiêm trọng | Mã dùng cực đại, argmax và topk | Đã thống nhất tối đa hóa; GA trung tâm có điểm cao hơn SG-GA trong lượt mới | Có; đã sửa |
| A06 | 4.4, 4.6; trang 5–7 | Độ nổi bật dựa trên gradient/top 5% | Không khớp phương pháp | Cao | Mã chọn một vị trí bằng phương sai Laplacian của ảnh | Đã mô tả đúng và giữ vị trí cố định, không gọi là độ nhạy của mô hình | Có; đã sửa |
| A07 | 4.7; trang 7 | EOT có thay đổi tỷ lệ/phối cảnh | Không khớp biến đổi | Cao | Chưa có các biến đổi đó trong mã | Đã báo xoay ±15°, sáng/tương phản 0,2, làm mờ, nhiễu; một mẫu/truy vấn | Có; đã sửa |
| A08 | 3; trang 3–4 | Tấn công hộp đen nghiêm ngặt | Giả định quyền truy cập | Cao | Cần điểm trước NMS không phải đầu ra công khai thông thường | Đã gọi hộp xám và nêu tính thực tế của kênh thông số nội bộ | Có để làm rõ; thực tế sản phẩm cần xác nhận |
| A09 | 5.3; trang 8 | Máy chủ biên đời cũ/Raspberry Pi đã được đo | Khái quát phần cứng | Cao | Lượt mới chỉ chạy trên i5-1335U giới hạn luồng | Đã báo máy thực, RAM/OS/phiên bản; không nhận là phép mô phỏng Pi | Có để sửa; nhiều nền tảng cần thí nghiệm |
| A10 | 5.7, 6.3; trang 9–10, 13 | So sánh GA và tìm kiếm ngẫu nhiên không cùng ngân sách | Nhiễu phương pháp cũ | Cao | Không thể quy khác biệt cũ cho thuật toán | Lượt mới đã dùng cùng 400 truy vấn, EOT, diện tích, 10 seed; ảnh tối ưu vẫn chỉ một | Một phần đã xử lý; cần tối ưu đa ảnh |
| A11 | 6.3; trang 13 | SG-GA vượt trội | Tuyên bố không được dữ liệu hỗ trợ | Nghiêm trọng | 312,61 ± 10,65 thấp hơn 367,39 ± 18,40; khoảng so với ngẫu nhiên gồm không | Đã bỏ tuyên bố ưu thế; không chọn lại số liệu để cứu kết luận | Có để báo đúng; không để tạo ưu thế |
| A12 | 5.6, 12.4; trang 9, 21–22 | Khoảng tin cậy xác nhận giả thuyết | Suy luận thống kê quá mức | Cao | Bootstrap hậu nghiệm, 35 tương phản, chưa điều chỉnh đa so sánh | Đã gắn nhãn thăm dò; không có giá trị p xác nhận hay khoảng tin cậy AP | Có để hạn chế; kiểm định xác nhận cần thiết kế trước |
| A13 | 5.8, 6.5; trang 11, 15–17 | COCO128 là tập kiểm tra độc lập | Rò rỉ/đại diện dữ liệu | Cao | Tập con train2017 có thể trùng tiền huấn luyện; chỉ một miếng dán/thuật toán | Đã giới hạn thành chẩn đoán hậu xử lý, không khẳng định khái quát | Có để làm rõ; cần tập độc lập |
| A14 | 5.8, 6.5; trang 11, 16 | AP báo là COCO chính thức | Không khớp trình đánh giá | Cao | Mã tùy chỉnh, không xử lý đám đông/bỏ qua như COCO; NMS không phân biệt lớp | Đã gọi AP/mAP chẩn đoán, giữ ngưỡng rõ trong bảng | Có để đổi tên; COCO chính thức cần chạy riêng |
| A15 | 5.8; trang 11 | Nhãn COCO128 hoàn chỉnh | Thiếu nhãn nguồn | Cao | Hai ảnh thiếu tệp nhãn; nhãn rỗng ảnh hưởng chỉ số | Đã công bố ngoại lệ trong ZIP, không tự bịa nhãn | Có để công bố; xác nhận độ chính xác cần bộ nhãn chuẩn |
| A16 | 5.7; trang 10 | Sọc chéo được gọi ô bàn cờ | Nhãn đối chứng sai | Vừa | Công thức hai họa tiết khác nhau | Hình mới gọi đúng sọc; COCO128 thêm ô bàn cờ 4 × 4 thực sự; dữ liệu thô giữ nguyên | Có; đã sửa |
| A17 | 5.7; trang 10 | Họa tiết và miếng dán trên zidane có cùng vị trí | Yếu tố gây nhiễu | Cao | Đối chứng chọn lại vị trí, miếng dán giữ vị trí bus | Đã công bố; phép COCO128 giữ vị trí cố định | Có để nêu giới hạn; kiểm tra riêng cần chạy lại |
| A18 | 11; trang 21 | max_det sau NMS giảm công việc NMS | Sai ranh giới phòng thủ | Cao | Cắt đầu ra không giảm đầu vào NMS | Lượt mới đã có top-100 trước NMS; không gọi max_det là giới hạn trước | Có để sửa; đối chiếu mới đã có |
| A19 | 6.5, 11; trang 16, 21 | Phòng thủ giữ nguyên chất lượng | Đánh đổi bị bỏ qua | Cao | SG-GA mất 5,27 điểm phần trăm độ bao phủ với top-100 | Đã báo giảm chi phí kèm mất độ bao phủ trên cả sạch và miếng dán | Có để sửa; quét nhiều giới hạn cần thí nghiệm |
| A20 | 8; trang 19 | Chuyển giao kiến trúc đã được xác lập | Đo không đồng nhất | Nghiêm trọng | Bộ bao YOLO và torchvision đo khác công đoạn | Đã gọi chưa xác lập; giữ kết quả thăm dò tách biệt | Có để hạ tuyên bố; cần chuẩn hóa phép đo |
| A21 | 9; trang 19–20 | Ảnh giao diện chứng minh miếng dán in gây DoS | Thiếu giao thức vật lý | Nghiêm trọng | Không có nhật ký ghép cặp và siêu dữ liệu đầy đủ | Giữ ảnh gốc như minh họa; không dùng các con số trên ảnh làm hiệu ứng | Có để sửa; vật lý cần người bố trí thiết bị |
| A22 | 5.5; trang 9 | Bộ đệm 2 bảo đảm luôn xử lý khung mới nhất | Giả định không được kiểm tra | Cao | Mã không xác nhận hiệu lực bộ đệm/tuổi khung | Đã đề xuất đo hàng đợi, khung bỏ, tuổi khung; không chạy camera tự động | Có để sửa; hành vi thật cần thử có kiểm soát |
| A23 | 5.5, 6.8; trang 9, 18 | CPU/FPS cũ đại diện chi phí toàn trình | Sai phạm vi đo | Cao | FPS chỉ đo hàm dự đoán; CPU/RAM toàn hệ thống; nhiệt độ lỗi trả 0 | Đã phân biệt CPU tiến trình, thông lượng ảnh tĩnh, camera và đo công đoạn | Có; đã sửa diễn giải |
| A24 | Tài liệu tham khảo; trang 22–23 | Tài liệu cũ sai/không xác minh | Độ tin cậy trích dẫn | Cao | Trích sai tiêu đề/tác giả/năm làm hỏng lập luận | Bản mới giữ 10 mục đã rà; không chuyển nguyên 22 mục từ Word gốc | Có bằng xác minh nguồn, không bịa siêu dữ liệu |
| A25 | Đầu bài; trang 1 | Word gốc 2 tác giả, TeX 3 tác giả | Thông tin tác giả | Cao | Người biên tập không tự xác lập tư cách tác giả | Giữ 3 tên đang có trong TeX, đã hỏi tác giả; chưa nhận xác nhận | Cần tác giả xác nhận |
| A26 | Toàn tài liệu | Word/TeX cùng nội dung và đúng hình thức | Kiểm tra tạo tác | Vừa | Biên dịch TeX chưa thành công; kiểm tra Word không xác nhận bố cục CAS PDF | Word đã kiểm tra đủ 23 trang; TeX đã sửa tại chỗ nhưng PDF còn chưa xác minh | Word đã kiểm tra; TeX cần trình biên dịch hoạt động |

## 3. Ba nhóm hành động

### A. Có thể sửa ngay, đã thực hiện

Đã sửa số cặp NMS; chiều fitness; tên mô hình, dạng tensor, kích thước/diện tích; vị trí Laplacian; thông số GA/EOT; ngưỡng tối ưu và ngưỡng đánh giá; ranh giới lọc/top-100/NMS/max_det; tên họa tiết; định nghĩa đo CPU/RSS/thông lượng; chú thích trung tính và cấu trúc 14 mục. Đã thêm câu hỏi nghiên cứu và câu trả lời tương ứng. Đã giữ kết quả âm và cập nhật hình từ số liệu có bản kê nguồn. Các chỉnh sửa không làm thay đổi số đo thô.

### B. Cần tác giả xác nhận, không tự điền

- Hai hay ba tác giả; người liên hệ và địa chỉ thư điện tử cuối cùng. Bản hiện giữ danh sách của TeX, không suy ra tư cách tác giả từ lịch sử tệp.
- Camera được phép dùng, độ phân giải/tốc độ khung hình, bộ mã hóa/kết nối, cấu hình thu hình và mốc thời gian có thể đồng bộ.
- Miếng dán gốc có được in/hiển thị thật trong lần nào; vật liệu, máy in, kích thước cm, khoảng cách/góc/độ sáng và nhật ký gốc. Ảnh giao diện không đủ xác nhận các thông số.
- Mục tiêu bài: trình bày tấn công mới với bằng chứng hiệu lực mới, hay nghiên cứu kiểm chứng/kết quả âm. Không đổi mục tiêu chỉ vì dữ liệu hiện tại dễ đáp ứng một định vị khác.
- Phê duyệt bản tiếng Việt trước khi dịch tiếng Anh, theo yêu cầu mới nhất của tác giả.

### C. Cần thí nghiệm/phân tích mới

Các giao thức E01–E08 ở mục 8 là đề xuất chưa thực hiện. Không ghi chúng như kết quả hoặc hứa sẽ thu được hiệu ứng mong muốn. Cỡ mẫu là mức kế hoạch ban đầu, cần kiểm tra công suất và khả năng thực hiện; không bảo đảm ý nghĩa thống kê.

## 4. Lỗi số học và logic quan trọng

1. Số cặp: 50 × 49 / 2 = 1.225; 3.000 × 2.999 / 2 = 4.498.500; tỷ số = 3.672,2449. Chỉ là tỷ số số cặp lý thuyết. NMS torchvision mới: 0,07053 và 9,95332 ms trung bình giữa seed; tỷ số hai trung bình khoảng 141,1. Không phải độ trễ tăng bởi miếng dán.
2. Fitness được tối đa hóa. Lượt mới: SG-GA 312,61 ± 10,65; GA trung tâm 367,39 ± 18,40; ngẫu nhiên 309,97 ± 5,65. Chênh lệch SG-GA trừ trung tâm −54,78, khoảng thăm dò [−66,81; −42,45]; trừ ngẫu nhiên 2,64, khoảng [−1,92; 7,64]. Không gộp với số liệu cũ 65,07/64,02/15,18 trên tensor ngẫu nhiên.
3. Miếng dán 64 × 64 trên 320 × 320 chiếm 4%. Miếng dán 256 × 256 trên 1.280 × 720 chiếm khoảng 7,11%, không mặc định 15–20%. Cần phân biệt diện tích pixel với kích thước cm/góc quan sát.
4. YOLOv8n ở 320 × 320 có 2.100 vị trí dự đoán trong tensor thực đo; không thể nhận kết quả 3.000 ứng viên sau ngưỡng ở cấu hình này. Mức 3.000/8.400 trong phép đo NMS là hộp tạo trực tiếp.
5. Top-100 là giới hạn trước NMS. max_det = 300 là giới hạn phát hiện sau NMS. Hai thao tác không tương đương.
6. COCO128 tại ngưỡng 0,01: ứng viên SG-GA 114,78 so với sạch 116,06; AP50 chẩn đoán 41,70% so với 46,00%. Đây là giảm chất lượng phát hiện trong phép đo, không phải DoS.
7. Với SG-GA, top-100 làm độ bao phủ 52,85% → 47,58%, mất 5,27 điểm phần trăm, không phải giảm 5,27% tương đối. Trung vị NMS 5,08 → 4,65 ms; không dùng tỷ số hai trung vị này làm hiệu ứng toàn trình camera.

## 5. Từ điển thuật ngữ thống nhất

| Thuật ngữ dùng | Ký hiệu/tên dữ liệu | Định nghĩa | Không đồng nhất với |
|---|---|---|---|
| Dự đoán thô | B_raw, K | Đầu ra trước lọc điểm tin cậy | Số hộp sau ngưỡng hoặc sau NMS |
| Ứng viên sau ngưỡng | B_tau, N_active, active_candidates | Các dự đoán có điểm lớn hơn ngưỡng, đếm trước top-k | Số đầu vào NMS khi có giới hạn |
| Đầu vào NMS | nms_input_candidates | Ứng viên thực sự chuyển tới NMS sau giới hạn | K hoặc phát hiện cuối |
| Phát hiện cuối | D, final_detections | Hộp giữ lại sau NMS và giới hạn đầu ra | Ứng viên hoạt động |
| Ngưỡng tối ưu | tau_opt = 0,01 | Ngưỡng trong hàm fitness | Mặc định vận hành sản phẩm |
| Ngưỡng đánh giá | 0,01 hoặc 0,25 | Ngưỡng được quy định cho từng nhánh kiểm chứng | Một cấu hình camera đã xác nhận |
| Fitness EOT tốt nhất quan sát | best_eot_fitness | Cực đại điểm truy vấn có một mẫu EOT | Kỳ vọng EOT độc lập hay độ trễ |
| Khuếch đại ứng viên | Tỷ số ứng viên miếng dán/sạch | Đại lượng đại diện khối lượng hậu xử lý | Tỷ số CPU hoặc tiêu chí DoS |
| Độ trễ NMS | nms_ms | Thời gian của mã NMS/công đoạn được nêu | Tổng thời gian/camera FPS |
| Thông lượng ảnh tĩnh | fps trong phép thử hai ảnh | Số lượt vòng xử lý chia thời gian vòng đo | Tốc độ camera/sensor-to-output |
| CPU tiến trình chuẩn hóa toàn máy | cpu_percent_machine_normalized | Thời gian CPU tiến trình/chạy thực/chia số bộ xử lý logic | psutil.cpu_percent toàn hệ thống |
| RSS tiến trình | rss_mib | Bộ nhớ thường trú theo MiB tại thời điểm đo | Bộ nhớ tăng do miếng dán hoặc rò rỉ |
| AP/mAP chẩn đoán | Tên chỉ số 101point_diagnostic | Chỉ số từ bộ ghép hộp và nhãn đã mô tả | COCO chính thức |
| Chèn số | Miếng dán trên tensor ảnh | Can thiệp sau giải mã | Miếng dán vật lý hoặc tấn công mạng đã thực hiện |
| Mô phỏng EOT | Biến đổi toàn ảnh đã chèn | Biến đổi phần mềm trong tối ưu | Thu ảnh miếng dán đã in |
| Miếng dán vật lý | In/hiển thị ngoài cảnh | Camera tự thu qua quang học/cảm biến | Ảnh tensor gắn nhãn “physical” |

## 6. Cấu trúc bài đã sửa và định vị đóng góp

Mười bốn mục hiện có: Giới thiệu; Nghiên cứu liên quan; Mô hình đe dọa và phát biểu bài toán; Phương pháp đề xuất; Phương pháp thực nghiệm; Kết quả; Thực nghiệm loại bỏ thành phần và phân tích độ nhạy; Đánh giá tính chuyển giao; Đánh giá thế giới vật lý; Thảo luận; Hàm ý phòng thủ; Hạn chế và các mối đe dọa tới tính hợp lệ; Cân nhắc đạo đức; Kết luận.

Đóng góp được mô tả ở mức quy trình có thể kiểm toán, không tự nhận mới vì ghép GA với Laplacian:

| Đóng góp hiện tại | Bằng chứng | Điều chưa chứng minh |
|---|---|---|
| Đặc tả giao diện hộp xám trước NMS | Mục 3 và mã đọc vector điểm | Kênh này tồn tại trên mọi camera thương mại |
| Tách fitness, ứng viên và chi phí thực thi | Phương trình/mã đo và kết quả âm | Đại lượng đại diện bảo đảm DoS |
| So sánh tối ưu cùng ngân sách có bản kê nguồn | 3 × 10 lượt, 400 truy vấn, hình tiến trình | Ưu thế SG-GA trên phân bố ảnh hoặc đối chứng tiên tiến |
| Chẩn đoán đánh đổi lọc ứng viên/chất lượng | 128 ảnh có nhãn, 8 điều kiện, 3 cấu hình | Phòng thủ phổ quát hoặc độ chính xác COCO chính thức |

Năm tiêu đề thay thế thận trọng, bằng tiếng Việt ở giai đoạn hiện tại:

1. Khuếch đại ứng viên hộp xám để kiểm thử sức chịu tải NMS trong các hệ thống phát hiện vật thể biên.
2. Kiểm chứng mối liên hệ giữa ứng viên dự đoán và độ trễ hậu xử lý trong phát hiện vật thể tại biên.
3. Tối ưu miếng dán hộp xám dựa trên thông số trước NMS: phương pháp, chi phí và giới hạn thực nghiệm.
4. Từ fitness đến chi phí hệ thống: kiểm chứng miếng dán đối kháng trên đường ống phát hiện vật thể dùng NMS.
5. Đánh đổi giữa giới hạn ứng viên, độ trễ NMS và chất lượng phát hiện dưới nhiễu miếng dán.

Tiêu đề 1 đang được dùng. Các tiêu đề 2–5 là lựa chọn định vị, không thay đổi tự động và không khẳng định tính mới so với toàn bộ văn liệu.

## 7. Trước → sau và tuyên bố đã hạ mức

| Tuyên bố trước | Diễn giải sau hợp nhất | Lý do |
|---|---|---|
| Miếng dán tạo hàng nghìn hộp và làm hệ thống tê liệt | Chưa có khuếch đại ứng viên/độ trễ toàn vòng nhất quán; 3.000 hộp là tải NMS tổng hợp | Không nối được chuỗi miếng dán → tải → QoS |
| Phương pháp nổi bật vượt trội hoàn toàn | GA trung tâm có fitness cao hơn; SG-GA so với ngẫu nhiên chưa có ưu thế rõ | So sánh mới cùng ngân sách, khoảng thăm dò |
| CPU chạm 100% chứng minh DoS | CPU riêng lẻ không phải tiêu chí DoS; cần độ trễ và trễ hạn có đối chứng | Phân biệt tài nguyên và tính sẵn sàng |
| Đã xác lập hiệu lực vật lý | Giữ ảnh giao diện gốc như minh họa, tách khỏi số liệu mới | Thiếu giao thức/nhật ký và siêu dữ liệu |
| Có thể chuyển giao phổ quát giữa kiến trúc | Chuyển giao chưa được xác lập, phép đo cũ khác công đoạn | Không so các đại lượng khác nhau |
| Giới hạn phát hiện giữ chất lượng và giảm tải NMS | Top-100 trước NMS giảm công việc nhưng mất độ bao phủ; max_det sau NMS khác chức năng | Có dữ liệu độ chính xác/độ trễ mới |
| Phép biến đổi mô phỏng là thử vật lý | EOT là biến đổi phần mềm toàn ảnh trong tối ưu | Không có camera/vật liệu trong phép biến đổi |
| Đã phát hiện điểm yếu của toàn ngành | Kết quả điều kiện trên một máy, mô hình, tiền xử lý và mã NMS | Phạm vi khái quát hạn chế |

Đã loại hoặc hạ mức các tuyên bố: DoS vật lý dứt khoát; cạn kiệt toàn RAM; 3.000+ ứng viên do detector 320 tạo; ưu thế thống kê của Laplacian; tỷ lệ chuyển giao CPU/FPS cũ; độc lập kiến trúc; phòng thủ giữ nguyên chất lượng; tỷ số cặp lý thuyết là tỷ số độ trễ.

## 8. Thí nghiệm còn cần và giao thức đề xuất

Các mức seed/lượt dưới đây là kế hoạch, không phải kết quả. Khi tái sử dụng một phép đo đã làm, phải giữ định danh lượt và ranh giới công đoạn; không đổi tên mô phỏng thành phần cứng thật.

### E01 — Đối chứng tấn công NMS mạnh

[CẦN THÍ NGHIỆM MỚI — ƯU TIÊN CAO]

- Lý do: GA/ngẫu nhiên kiểm tra thành phần tối ưu, chưa đặt hiệu lực trong tương quan với tấn công đã công bố.
- Giao thức: chọn ít nhất một triển khai gốc phù hợp, kiểm tra mục tiêu/ràng buộc trước khi chạy; giữ cùng trọng số, bộ ảnh, tiền xử lý, ngưỡng/NMS, phần cứng và ranh giới đo. Nếu tấn công gốc dùng nhiễu toàn ảnh, không giả vờ cùng ràng buộc miếng dán; báo riêng hoặc chuyển thể có mô tả và đối chứng.
- Biến: loại tấn công, giới hạn diện tích/biên độ, quyền truy cập, số truy vấn. Dùng ít nhất 10 seed/miếng dán khi thuật toán ngẫu nhiên và 10 khối đo ghép cặp; phép quyết định phải được quy định trước.
- Đối chứng: ảnh sạch, SG-GA, GA trung tâm, ngẫu nhiên, nhiễu/sọc/ô bàn cờ/màu đặc; tấn công NMS mạnh.
- Đo: ứng viên, đầu vào NMS, phát hiện cuối, thời gian NMS/toàn vòng, tỷ lệ trễ hạn, chất lượng phát hiện, số truy vấn và thời gian tối ưu.
- Bảng/hình: hàng là phương pháp, cột là quyền truy cập/ràng buộc/truy vấn/chất lượng/độ trễ; đường hiệu năng theo truy vấn và phân bố chênh lệch thời gian.

### E02 — Tối ưu đa ảnh và tập kiểm tra độc lập

[CẦN THÍ NGHIỆM MỚI]

- Lý do: đã có so sánh cùng 400 truy vấn nhưng chỉ tối ưu trên bus; COCO128 không phải tập kiểm tra độc lập với tiền huấn luyện.
- Giao thức: chốt trước bộ tối ưu/kiểm tra nhiều nội dung và tiêu chí tách; chọn miếng dán không nhìn kết quả kiểm tra. Giữ cùng vị trí và EOT giữa đối chứng phù hợp; nếu tối ưu cả vị trí, tính đầy đủ truy vấn tìm vị trí.
- Biến: vị trí Laplacian/trung tâm/ngẫu nhiên, phương pháp, nhiều mức ngân sách định trước (ví dụ 400/1.000/2.500), 10 seed/phương pháp/ngân sách.
- Đối chứng: ngẫu nhiên, GA trung tâm, GA vị trí ngẫu nhiên và GA Laplacian.
- Đo: cực đại tích lũy theo truy vấn, fitness đánh giá lại trên nhiều mẫu EOT độc lập, ứng viên, độ trễ và chất lượng trên tập kiểm tra.
- Bảng/hình: đường theo truy vấn, bảng kết quả trên tập tối ưu và kiểm tra tách biệt; không chọn điểm dừng theo lợi thế quan sát hậu nghiệm.

### E03 — Độ nhạy diện tích và EOT

[CẦN THÍ NGHIỆM MỚI]

- Lý do: diện tích 1–16% cũ chỉ một lượt/mức; EOT hiện một mẫu và không có tỷ lệ/phối cảnh.
- Giao thức: diện tích mục tiêu 1/2/4/8/16%, công bố diện tích thực sau làm tròn; 10 seed/mức, cùng ngân sách truy vấn, ảnh và đối chứng. Loại bỏ từng biến đổi EOT có đối chứng riêng, giữ các yếu tố khác; đánh giá lại với biến đổi độc lập.
- Biến: diện tích, loại EOT, số mẫu EOT/truy vấn, vị trí. Báo cả số truy vấn ảnh khi tăng mẫu EOT, không chỉ số thế hệ.
- Đối chứng: cấu hình đầy đủ, không EOT, từng cấu hình loại bỏ thành phần và họa tiết cùng diện tích.
- Đo: fitness/ứng viên/độ trễ/chất lượng, thời gian tối ưu; khả năng nhìn thấy chỉ dùng chỉ số đã định nghĩa hoặc đánh giá có giao thức, không gọi độ nhỏ là “tàng hình”.
- Bảng/hình: diện tích × phương pháp với sai số giữa seed; biểu đồ hiệu lực/chi phí theo diện tích; bảng thông số EOT và ngân sách.

### E04 — Chuyển giao và nhiều triển khai NMS

[CẦN THÍ NGHIỆM MỚI]

- Lý do: phép chuyển giao cũ đo các công đoạn khác nhau; NMS torchvision và NMS thuần PyTorch không đồng nhất.
- Giao thức: đóng băng miếng dán nguồn từ YOLOv8n; chốt mô hình đích, trọng số, tiền xử lý đúng từng mô hình và ranh giới đo tương ứng. Không thay miếng dán theo kết quả đích. Ít nhất 10 miếng dán nguồn/seed, cùng tập ảnh và 10 khối đo ghép cặp. Đồng bộ khi dùng bộ tăng tốc.
- Biến: mô hình YOLO khác, Faster R-CNN/RetinaNet, mô hình không NMS như DETR/RT-DETR; mã NMS và hậu xử lý.
- Đối chứng: sạch và họa tiết cùng ràng buộc; mô hình không NMS là đối chứng cơ chế, không gán số ứng viên NMS giả cho nó.
- Đo: công đoạn phù hợp, tổng thời gian, chất lượng; ứng viên/đầu vào NMS chỉ khi có định nghĩa tương ứng.
- Bảng/hình: mô hình × miếng dán, ranh giới đo mỗi hàng; phân rã thời gian và bảng tỷ số ghép cặp, giữ kết quả âm.

### E05 — Phần cứng và QoS triển khai

[CẦN THÍ NGHIỆM MỚI]

- Lý do: giới hạn luồng trên laptop không chứng minh hiệu lực trên máy biên khác, và CPU không định nghĩa DoS.
- Giao thức: thêm ít nhất một thiết bị biên thật khác máy hiện tại; chốt thời hạn theo ứng dụng và tải sạch, khởi động, nguồn điện/nhiệt/tần số, tác vụ nền; 10 khối lượt độc lập/thiết bị/điều kiện, xáo trộn thứ tự và khoảng nghỉ có kiểm soát.
- Biến: thiết bị/backend, số luồng, tải cảnh, ngưỡng/giới hạn, sạch/miếng dán. Không đổi đồng thời phần cứng và thuật toán mà không ghi.
- Đối chứng: sạch ghép cặp, họa tiết và tấn công mạnh. Mức sạch cũng phải đáp ứng thời hạn để tránh nhận hệ thống vốn chậm là DoS do miếng dán.
- Đo: độ trễ công đoạn/toàn trình, tuổi khung, tỷ lệ trễ hạn, khung bỏ/thông lượng, CPU tiến trình và RSS; nhiệt/điện năng chỉ nếu cảm biến đo được.
- Bảng/hình: thông số từng thiết bị; phân bố thời gian và tỷ lệ trễ hạn theo khối; không chỉ chọn giá trị CPU đỉnh.

### E06 — Miếng dán in và camera có kiểm soát

[CẦN TÁC GIẢ XÁC NHẬN THIẾT BỊ; CẦN THÍ NGHIỆM MỚI]

- Lý do: ảnh giao diện gốc thiếu nguồn số liệu/giao thức; mô phỏng EOT không thay thu hình thật.
- Giao thức: tác giả cung cấp camera được phép dùng, vật liệu/máy in hoặc màn hiển thị, kích thước cm, ảnh miếng dán đã in. Sau đó chốt ít nhất 3 khoảng cách × 3 góc × 3 mức sáng × 2 camera/bộ mã hóa, 10 khối thử/điều kiện nếu khả thi; chốt trước các mức vật lý thật, không tự bịa khoảng cách/lux. Giữ tỉ lệ miếng dán trong ảnh và báo phép đo diện tích/góc quan sát.
- Biến: khoảng cách, góc, sáng, diện tích, camera/codec; tách chèn số qua camera, miếng dán vật lý và không miếng dán.
- Đối chứng: sạch, giấy/màn trống, nhiễu, ô bàn cờ và miếng dán đã tối ưu; xáo trộn thứ tự, ghi mốc thời gian/tuổi khung/hàng đợi.
- Đo: ứng viên/độ trễ NMS/toàn trình/khung bỏ/trễ hạn/CPU/RSS; giữ video hoặc ảnh nguồn và bản kê theo chính sách riêng tư đã thống nhất.
- Bảng/hình: bảng bố trí vật lý và định danh lượt, ảnh bố trí có thước/siêu dữ liệu, bản đồ hiệu năng theo khoảng cách/góc/sáng; không dùng ảnh giao diện đơn lẻ thay nhật ký.

### E07 — Phòng thủ và độ chính xác chuẩn

[CẦN THÍ NGHIỆM MỚI; ĐÃ CÓ CHẨN ĐOÁN BAN ĐẦU]

- Lý do: top-100 và hai ngưỡng đã kiểm tra, nhưng chưa có quét nhiều cấu hình, bộ đánh giá COCO chính thức hoặc tấn công thích nghi.
- Giao thức: bộ kiểm tra có nhãn độc lập, trình COCO chuẩn; chốt ngưỡng 0,01/0,05/0,10/0,25/0,50 và giới hạn phù hợp số vị trí, cùng dự đoán gốc khi so hậu xử lý. 10 miếng dán/seed và 10 khối thời gian; tấn công thích nghi được báo riêng ngân sách.
- Biến: ngưỡng, top-k trước NMS, giám sát/điều tiết tải ngược; chính sách bỏ khung cần chạy trên luồng thực.
- Đối chứng: không phòng thủ, cấu hình sạch, tấn công hiện có và tấn công mạnh dưới cùng ràng buộc.
- Đo: AP/độ bao phủ sạch và nhiễu, thời gian NMS/toàn trình, trễ hạn/khung bỏ. Phải báo tác hại trên ảnh sạch.
- Bảng/hình: cấu hình × sạch × tấn công; đường đánh đổi độ trễ–chất lượng; phân biệt phòng thủ thật với việc hệ thống giảm chức năng.

### E08 — Suy luận thống kê xác nhận

[CẦN THIẾT KẾ/PHÂN TÍCH MỚI]

- Lý do: khoảng bootstrap hiện có là thăm dò hậu nghiệm, không phải xác nhận giả thuyết; một miếng dán seed 0 trên COCO128 không ước lượng biến thiên miếng dán.
- Giao thức: định trước giả thuyết, hiệu ứng có ý nghĩa thực tế, kết cục chính, đơn vị ngẫu nhiên hóa, xử lý nhiều so sánh và tiêu chí loại lượt; dùng thăm dò để chọn cỡ mẫu theo công suất, không gọi 10 seed tự động đủ.
- Biến: seed/miếng dán, ảnh, khối đo/thiết bị; giữ cấu trúc lồng nhau hoặc ghép cặp. Không gộp tất cả lượt thời gian thành mẫu độc lập.
- Đối chứng: ảnh sạch và phương pháp cạnh tranh cùng khối; kiểm định/ước lượng được chọn theo thiết kế đã chốt.
- Đo: hiệu ứng và khoảng tin cậy tương ứng; AP cần lấy mẫu lại phù hợp trên bộ ảnh và tính lại chỉ số gộp, không lấy trung bình AP từng ảnh.
- Bảng/hình: số đơn vị độc lập/nhóm, hiệu ứng thô và chuẩn hóa khi phù hợp, khoảng tin cậy/kết cục chính; giá trị p chỉ báo sau phép phân tích thực thi hợp lệ.

## 9. Danh mục sẵn sàng nộp Q1 và khoảng trống còn lại

### Những việc đã kiểm chứng

- [x] Sửa lỗi số cặp, chiều fitness, đại lượng hộp và kiến trúc nguồn.
- [x] Tách mục tiêu tối ưu khỏi chi phí thực thi và tiêu chí DoS.
- [x] Cấu trúc 14 mục, mô hình đe dọa, tính hợp lệ và đạo đức; kết quả âm giữ rõ.
- [x] So sánh ba phương pháp cùng 400 truy vấn × 10 seed trên một ảnh tối ưu.
- [x] Bản kê môi trường, nguồn/ảnh/trọng số và đo công đoạn cho lượt mới.
- [x] Chẩn đoán có nhãn 128 ảnh và đánh đổi top-100/ngưỡng với chất lượng.
- [x] Word hợp nhất có công thức chỉnh sửa được, đủ hình/bảng và được kiểm tra 23 trang.
- [x] TeX giữ nguồn và mẫu CAS hiện tại, các đường dẫn ảnh có trong gói công bố.
- [x] Gói 275 tệp khoa học được cập nhật lên Drive đúng thư mục; Git mới đã đẩy, không kèm báo cáo/prompt/tệp QA nội bộ.

### Những việc chưa được chứng minh hoặc cần người dùng

- [ ] Đối chứng tấn công tiên tiến dưới ràng buộc tương đương.
- [ ] Tối ưu đa ảnh, nhiều mức truy vấn và tập kiểm tra độc lập với tiền huấn luyện.
- [ ] Thực nghiệm diện tích/EOT nhiều seed và đánh giá lại EOT độc lập.
- [ ] Nhiều phần cứng/mô hình; chuyển giao cùng ranh giới đo và đối chứng không NMS.
- [ ] Tiêu chí QoS/trễ hạn được thiết kế trước và luồng camera đo toàn trình.
- [ ] Bộ dữ liệu miếng dán vật lý có cấu hình và phép thử kiểm soát.
- [ ] AP COCO chính thức, phòng thủ nhiều cấu hình và tấn công thích nghi.
- [ ] Suy luận thống kê xác nhận/cỡ mẫu phù hợp, khoảng tin cậy AP.
- [ ] Danh sách tác giả cuối cùng và sự phê duyệt của tác giả.
- [ ] Biên dịch thành công và kiểm tra bố cục PDF TeX.
- [ ] Bản tiếng Anh sau khi tác giả duyệt bản tiếng Việt.

| Mức | Vấn đề còn lại | Lo ngại phản biện | Hành động | Chỉ viết? | Cần thí nghiệm? | Tác động khoa học dự kiến |
|---|---|---|---|---|---|---|
| Nghiêm trọng | Tuyên bố DoS mới không được dữ liệu hiện tại hỗ trợ | Đóng góp hiệu lực không tồn tại trong phép thử | Giữ kết quả âm; tác giả chốt định vị, hoặc E01/E02/E05 với tiêu chí trước | Chỉ hạ tuyên bố được | Có nếu giữ đóng góp hiệu lực | Rất cao |
| Nghiêm trọng | Thiếu đối chứng mạnh | Không biết đóng góp so với phương pháp có sẵn | E01, công bố quyền truy cập/ràng buộc | Không | Có | Rất cao |
| Cao | Tập tối ưu một ảnh, COCO128 không độc lập | Khái quát yếu/tiền huấn luyện chồng lấn | E02 | Không | Có | Cao |
| Cao | Chuyển giao/phần cứng hạn chế | Không đại diện hệ thống biên | E04/E05 | Không | Có | Cao |
| Cao | Vật lý thiếu giao thức và số đo | Không thể giữ đóng góp vật lý | E06 sau xác nhận thiết bị | Có để bỏ tuyên bố | Có nếu giữ vật lý | Cao; nghiêm trọng nếu vật lý là đóng góp chính |
| Cao | Thống kê thăm dò | Không có xác nhận giả thuyết/biến thiên miếng dán | E08 | Không để xác nhận | Có/phân tích mới | Cao |
| Cao | Phòng thủ mới là chẩn đoán | Chưa kiểm tra tấn công thích nghi/chất lượng chuẩn | E07 | Không | Có | Cao |
| Vừa | Diện tích/EOT chưa kiểm soát nhiều seed | Không tách tác động từng thành phần | E03 | Không | Có | Vừa–cao |
| Vừa | Môi trường chưa có tệp khóa phụ thuộc tái tạo hoàn toàn | Phiên bản đang nạp và gói cài khác nhau | Dùng bản kê chính xác; kiểm tra tái tạo môi trường riêng | Một phần | Cần xác minh thực thi | Vừa |
| Vừa | PDF TeX chưa kiểm chứng | Chưa xác nhận hình thức tạo tác cuối | Trình biên dịch của ứng dụng hoạt động, rồi kiểm tra PDF | Không phải sửa khoa học | Không | Không tạo bằng chứng khoa học mới; cần cho bàn giao |
| Vừa | Hai danh sách tác giả | Thông tin nộp bài chưa chốt | Tác giả xác nhận, không tự suy đoán | Có sau xác nhận | Không | Tính chính xác xuất bản |
| Thấp | Tiếng Anh được hoãn | Chưa sẵn ngôn ngữ nộp quốc tế | Dịch/biên tập sau duyệt tiếng Việt | Có sau duyệt | Không | Hình thức, không tăng hiệu lực |

## 10. Kiểm toán hình, bảng và tài liệu tham khảo

### 10.1. Hình/bảng hiện tại

| Đối tượng | Câu hỏi/phạm vi | Kiểm tra diễn giải |
|---|---|---|
| Hình 1: kiến trúc | RQ1/RQ3, ranh giới tối ưu và đo | Nhánh vật lý nét đứt, không chứa CPU100/DoS giả |
| Hình 2: GA | RQ2 | 20 cá thể/20 thế hệ/400 truy vấn, 5 ưu tú, xác suất đột biến từng cá thể con |
| Hình 3: ảnh đầu vào | Nguồn ảnh và diện tích | Chèn số 4%, không nhận ảnh là vật lý |
| Hình 4: NMS tổng hợp | RQ1, chi phí thành phần | Trục hộp/thời gian ms; sai số giữa seed; không gán hộp tổng hợp cho detector |
| Hình 5: tiến trình tối ưu | RQ2 | Fitness cực đại EOT; vùng ± độ lệch chuẩn, không gọi khoảng tin cậy |
| Hình 6–7: hai ảnh | RQ1/RQ2 | Ngưỡng 0,01/0,25, ứng viên và NMS ms; chú thích nhiễu do vị trí trên zidane |
| Hình 8: COCO128 | Chất lượng/phòng thủ | Thang 0–1, AP gộp bộ ảnh, một miếng dán/phương pháp, không tạo thanh sai số giả |
| Hình 9: camera gốc | Minh họa lịch sử | Caption nêu thiếu định nghĩa/đối chứng/bản kê; không dùng số trên ảnh làm kết quả |
| Bảng 1: vai trò mô hình | Phạm vi kiến trúc | Mô hình mới và chuyển giao thăm dò tách rõ |
| Bảng 2: NMS | RQ1 | Hộp tạo trực tiếp, trung vị lặp rồi trung bình/độ lệch chuẩn seed |
| Bảng 3: tối ưu | RQ2 | Cùng ngân sách, số truy vấn/thời gian, không đảo chiều tốt/xấu |
| Bảng 4: vòng hai ảnh | RQ1 | Ứng viên/phát hiện/NMS/tổng/thông lượng, không gọi FPS camera |
| Bảng 5–6: COCO128 | Phòng thủ/chất lượng | Ứng viên trước cap/vào NMS; thời gian trung vị/IQR; chỉ số %, AP chẩn đoán |
| Bảng 7: tối ưu cũ | Thăm dò lịch sử | Tensor ngẫu nhiên, ngân sách/EOT cũ không ngang nhau |
| Bảng 8: diện tích cũ | Độ nhạy thăm dò | Diện tích thực, một lượt/mức, không suy xu hướng xác nhận |

Word đã được kiểm tra trực quan đủ 23 trang; sau lần sửa câu chuyển tiếp cuối, 22 trang không đổi pixel và trang 18 được kiểm tra lại. Không dùng kiểm tra Word để suy ra PDF TeX đúng. Nguồn sơ đồ SVG cho phép sửa nội dung, PNG dùng trong hai bản thảo; ảnh giao diện được trích từ tệp gốc, không chỉnh số hiển thị.

### 10.2. Trạng thái trích dẫn

Bản mới giữ 10 mục: Brown (Adversarial Patch, arXiv 2017); Athalye (ICML 2018); Shumailov (EuroS&P 2021); Wang (Daedalus, Transactions on Cybernetics 2022, 52(8), 7427–7440); Shapira (WACV 2023, 4571–4580); Chen (Overload, CVPR 2024, 24716–24725); Ma (SlowTrack, AAAI 2024, 38(5), 4062–4070); Wang (Can't Slow Me Down, CVPR 2025, 19230–19240); Monteuuis (Latency NMS Attacks, NeurIPS 2025); tài liệu dữ liệu COCO128 của Ultralytics.

Brown là bản arXiv; mục Ultralytics là nguồn dữ liệu/tài liệu, không được nhận là bài bình duyệt. Không tự thêm DOI chưa được xác minh. Các mục sai hoặc chưa xác minh trong 22 tài liệu của Word gốc không được nhập lại; bản gốc được bảo toàn để truy vết. Kiểm toán tài liệu cũ còn là lịch sử, không phải danh sách tài liệu đang trích trong bản mới.

## 11. Mô phỏng phản biện số 2

### Nhận xét chính

1. Bản sửa đã phân biệt DoS với sai dự đoán và sửa lỗi toán học/kiến trúc. Tuy nhiên, kết quả chính không hỗ trợ khuếch đại tải bởi SG-GA; tác giả cần giải thích giá trị nghiên cứu kiểm chứng và không bán lại kết quả như tấn công thành công.
2. So sánh mới cùng ngân sách giải quyết nhiễu lớn của lượt cũ, nhưng một ảnh tối ưu không đủ để xếp hạng quy tắc vị trí. GA trung tâm tốt hơn SG-GA về fitness trong cấu hình này.
3. Cực đại một mẫu EOT có thể được chọn vì biến đổi ảnh. Cần đánh giá lại fitness trên biến đổi độc lập và mục tiêu QoS thực, không chỉ vẽ đường cực đại tăng.
4. Chưa có đối chứng tấn công NMS mạnh. Phải làm rõ quyền truy cập và ràng buộc giữa các phương pháp trước khi kết luận mới/hiệu quả.
5. COCO128 mới cung cấp chẩn đoán chất lượng hữu ích, nhưng thuộc train2017, một miếng dán/thuật toán và nhãn nguồn có ngoại lệ. AP không phải COCO chính thức; cần kiểm tra độc lập cho kết luận khái quát.
6. Top-100 có phép đo mới nhưng làm mất độ bao phủ, cả trên ảnh sạch. Không thể gọi đây là phòng thủ giữ chất lượng; thiếu tấn công thích nghi và các cấu hình khác.
7. Chuyển giao cũ không cùng ranh giới công đoạn. Đã hạ tuyên bố là phù hợp, nhưng nếu chuyển giao là đóng góp thì phải thay bằng E04.
8. Ảnh camera cũ có giá trị minh họa, không đủ kiểm chứng vật lý. Nếu tiêu đề/đóng góp vật lý được giữ, E06 là bắt buộc.
9. Khoảng bootstrap hiện là thăm dò hậu nghiệm. Cần phân biệt seed/ảnh/khối, thiết kế phân tích trước và báo hiệu ứng thực tế bên cạnh bất định; không cần bịa ý nghĩa thống kê.
10. Bản kê nguồn và dữ liệu đã cải thiện khả năng tái lập. Còn cần xác minh lại trên môi trường khác, nhiều triển khai/phần cứng, tiêu chí thời hạn và tạo tác TeX biên dịch được.

### Nhận xét nhỏ

- Giữ nhất quán dấu thập phân, đơn vị ms/s/% và “điểm phần trăm”.
- Không gọi 2.100 vị trí YOLOv8n là hộp neo; thuật ngữ trước/sau ngưỡng, top-k và NMS phải có định nghĩa thống nhất.
- Chỉ rõ “tổng thời gian” bao gồm công đoạn nào trong từng phép thử; không cộng các trung vị công đoạn để suy FPS camera.
- Thống nhất danh sách tác giả theo xác nhận của nhóm, không theo tệp mới/cũ ngẫu nhiên.
- Bỏ các ghi chú công việc nội bộ khi nhóm đã thực hiện và xác nhận; hiện giữ các dấu yêu cầu thí nghiệm để không che phần thiếu bằng chứng.
- Kiểm tra PDF TeX cuối cùng khi trình biên dịch hoạt động, không chỉ nhìn nguồn hoặc dùng bản Word làm thay thế.
- Chỉ biên tập tiếng Anh sau khi tác giả duyệt bản tiếng Việt; không dùng dịch để làm mạnh hơn tuyên bố.

### Lý do có thể từ chối

Thiếu hiệu lực/đóng góp chính của một bài tấn công mới; thiếu đối chứng tiên tiến; tối ưu và kiểm tra quá hạn chế; chưa có kiểm chứng triển khai/vật lý; chưa có suy luận xác nhận. Lỗi số học, mô hình và so sánh ngân sách cũ đã được xử lý, nên không nên tiếp tục nêu chúng như lỗi chưa sửa trong bản hiện tại.

### Những việc cần chốt trước khi nộp

Tác giả chốt định vị và danh sách tác giả; hoàn thành E01/E02 và thiết kế E08; mở rộng E05/E07 theo phạm vi bài. E04 bắt buộc nếu nhận đóng góp chuyển giao; E06 bắt buộc nếu nhận đóng góp vật lý. E03 cần cho tuyên bố về thành phần/diện tích/EOT. Sau đó cập nhật cả Word/TeX, biên dịch/kiểm tra PDF và dịch tiếng Anh sau duyệt. Đây là yêu cầu tạo bằng chứng, không phải bảo đảm chấp nhận.

## 12. Đối chiếu 10 đầu ra được yêu cầu

| Đầu ra | Vị trí/bằng chứng | Trạng thái |
|---|---|---|
| Bảng kiểm toán phản biện chính | Mục 2, 26 dòng vấn đề | Đã cập nhật theo bản hợp nhất |
| Lỗi số học/logic quan trọng | Mục 4 | Đã đối chiếu số liệu mới |
| Từ điển thuật ngữ | Mục 5 | Đã thống nhất, có ranh giới top-k |
| Dàn ý sửa | Mục 6 và 14 mục của TeX/Word | Đã thực hiện tiếng Việt |
| Bản thảo viết lại | TeX hiện tại và Word HopNhat | Đã có tiếng Việt; tiếng Anh chờ duyệt; PDF TeX chưa xác nhận |
| Trước → sau các tuyên bố | Mục 7 | Đã cập nhật, không dùng kết quả cũ cứu tuyên bố |
| Danh sách tuyên bố đã hạ mức | Mục 7 | Đã chỉ rõ |
| Thí nghiệm còn cần | Mục 8, E01–E08 | Đề xuất có lý do, biến, đối chứng, số lượt, bảng/hình; chưa được coi đã làm |
| Danh mục sẵn sàng Q1/khoảng cách | Mục 9 | Phân biệt đã làm/chưa xác nhận |
| Mô phỏng phản biện số 2 | Mục 11 | Tiếng Việt, phản ánh cả tiến bộ và khoảng trống |

Chưa thể tuyên bố hoàn tất toàn bộ việc bàn giao cuối: PDF TeX chưa biên dịch/kiểm tra thành công và tiếng Anh chưa được tác giả cho phép thực hiện. Không chạy thêm camera khi chưa có thiết bị/bố trí đã xác nhận; không tự tạo kết quả cho E01–E08 để đánh dấu hết ô.
