# PREREG E5-phone: giao thức đo A/A trên điện thoại ARM (Galaxy A36, Snapdragon 6 Gen 3)

Trạng thái: **NHÁP, chờ duyệt** (chưa tag). Tag dự kiến sau khi duyệt: `prereg-e5phone-<ngày>`.
Biến thể của `PREREG_E5.md` (laptop). Thêm hai phần mới: **A. xác minh ghim core qua proot** và **B. kiểm soát xung nhiệt**.

## Mục đích và phạm vi
Định cỡ nhiễu đo (A/A) và MDE trên một thiết bị ARM thật, để các phép đo sau trên điện thoại (nếu có) biết hiệu ứng nhỏ nhất phát hiện được.
Đây là **một điện thoại ARM, một model (YOLOv8n), tensor chuẩn bị sẵn (resize trực tiếp 320×320)**; không phải thiết bị triển khai thực địa,
không có camera, không suy ra cho điện thoại hay chip khác. Gọi đúng tên: "điện thoại ARM (Galaxy A36) chạy trong Ubuntu proot trên Termux".
Không có patch, không có tấn công trong thí nghiệm này.

## Đã biết trước khi viết PREREG này (khai để không bị coi là đoán)
- SSH vào được; máy `aarch64`. Chưa đo thời gian gì trên điện thoại.
- Cài đặt xong: Ubuntu proot, Python 3.14.4, torch 2.15.0+cu132, torchvision 0.29.2+cu132, ultralytics 8.4.197, scipy 1.18.1, psutil 7.2.2
  (`~/sponge-sim/logs/setup_ubuntu.log`).
- Thăm dò tiến trình (`~/sponge-sim/logs/probe_process.py`, chỉ hỏi thông tin của chính tiến trình): thấy **8 CPU**,
  `sched_getaffinity` = {0..7}, torch mặc định **8 luồng** (OpenMP 4.5, oneDNN 3.13.3). Con số `1` mà log cài đặt in ra một lần
  (`IMPORT_OK … 1`) mâu thuẫn với 8 này; **không tin con số torch tự báo về số luồng/core**, kiểm bằng tải thật (phần A).
- Thông số công khai của chip (chưa kiểm trên máy): 4 lõi hiệu năng + 4 lõi tiết kiệm. **Không biết CPU nào thuộc cụm nào**; phần A phải xác định.
- Chưa đọc gì ngoài `~/sponge-sim` (sysfs, thermal, cpufreq): xem bước A0 xin phép riêng.

## Khác biệt so với E5 laptop (phải khai trong mọi trích dẫn)
| Mục | Laptop (E5) | Điện thoại (E5-phone) |
|---|---|---|
| Hệ điều hành | Windows 11 | Android → Termux → proot-distro Ubuntu (proot chạy bằng ptrace) |
| CPU | i5-1335U (P/E-core, đọc topology từ OS) | Snapdragon 6 Gen 3 (8 lõi, 2 cụm; ánh xạ phải đo) |
| Ghim core | `psutil.cpu_affinity` + `cpu_percent(percpu)` | `os.sched_setaffinity` + đọc CPU chạy gần nhất của từng luồng (phần A) |
| Phiên bản | torch 2.13.0+cpu, torchvision 0.28.0, ultralytics 8.4.90, scipy 1.18.0 | torch 2.15.0+cu132, torchvision 0.29.2+cu132, ultralytics 8.4.197, scipy 1.18.1 |
| Nhiệt | tham chiếu đầu khối, cờ >3% | đo theo đợt ngắn, nghỉ giữa đợt, đọc nhiệt nếu được (phần B) |
| Nguồn | cắm sạc, Balanced | cắm sạc, Không làm phiền, tắt tiết kiệm pin, mạng theo CLAUDE.md |

**Lệch phiên bản torch.** Điện thoại dùng bản `+cu132` (bản CUDA của PyPI cho aarch64, chạy CPU) mới hơn bản `+cpu` của laptop. Không cài lại vì cài xong là hết ngoại lệ mạng
(CLAUDE.md). Hậu quả đã khai trước: (1) **không so độ trễ tuyệt đối giữa laptop và điện thoại**; chỉ so tỷ số A/A, MDE và cấu trúc trong từng thiết bị;
(2) không coi hành vi NMS của ultralytics 8.4.197 tương đương 8.4.90; E5-phone chỉ dùng `torch_pure` và `tv_nms` (mã sao chép có SHA256), không dùng `ultra_native`;
(3) SHA256 của `yolov8n.pt` trên điện thoại phải bằng bản laptop (kiểm trước khi đo, sai thì dừng); (4) manifest ghi đủ phiên bản và `torch.__config__.parallel_info()`.

## A. Xác minh ghim core qua proot (làm TRƯỚC mọi phép đo)
Không giả định `psutil.cpu_affinity` hay `os.sched_setaffinity` hoạt động như trên Linux gốc: proot dịch lệnh hệ thống, Android có thể chặn.

**A0. Thăm dò chỉ-đọc (ĐÃ ĐƯỢC ĐỒNG Ý 01/11/2026; vẫn chỉ đọc, đặt `SPONGE_ALLOW_SYSFS=1`).** Chỉ đọc, không ghi, liệt kê trước các file: `/proc/cpuinfo`,
`/sys/devices/system/cpu/online`, `cpu*/cpu_capacity`, `cpu*/cpufreq/{cpuinfo_max_freq,scaling_cur_freq,scaling_governor}`,
`/sys/class/thermal/thermal_zone*/{type,temp}`, `/proc/self/task/*/stat`. File nào không đọc được (SELinux của Android) thì ghi "không đọc được", không tìm đường vòng.

**A1. Cơ chế ghim.** Trong tiến trình Python (trong proot): `os.sched_setaffinity(0, {cpu})`, đọc lại bằng `os.sched_getaffinity(0)`.
Nếu lỗi `EPERM`/bị bỏ qua (đọc lại không đổi) → không có ghim; chỉ còn profile "không ghim" gắn nhãn rõ, và **không được nói gì về số core**.

**A2. Kiểm bằng tải thật (không dựa vào `/proc/stat`, Android thường chặn).** Với mỗi tập CPU ứng viên S, đặt số luồng torch = 2·|S|, chạy matmul 3 giây và đo:
1. `effective_cores` = CPU-time tiến trình / thời gian thực; phải ≤ |S| + 0,15 (không vượt tập cho phép) **và** ≥ 0,5·|S| nếu |S| = 1, ≥ 0,7·|S| nếu |S| > 1 (song song thật; bài học E4 laptop);
2. lấy mẫu mỗi 50 ms trường "CPU chạy gần nhất" trong `/proc/self/task/<tid>/stat` của từng luồng; ≥ 99% mẫu phải thuộc S;
3. benchmark cố định (matmul 256×256 ×20) chạy riêng trên từng CPU 0..7: phân cụm theo tốc độ. Chấp nhận phân cụm khi trong cùng cụm chênh ≤ 15% và giữa hai cụm chênh ≥ 1,3 lần;
   nếu không thỏa → ghi "cụm không xác định", chỉ dùng CPU được xác định rõ.
Profile chỉ được dùng khi qua cả 3 điều kiện. Tên profile ban đầu: `big1` (1 CPU cụm nhanh), `little1` (1 CPU cụm chậm); `big4`/`little4` chỉ thêm nếu kiểm song song (điều kiện 1) đạt.
Chọn CPU cụ thể trong cụm = CPU có `benchmark` ổn định nhất (CV nhỏ nhất) trong 5 lần lặp; ghi lý do. Tránh gán nhầm CPU đang gánh phiên SSH: ghi tải nền (CPU-time của `sshd`/Termux nếu đọc được) vào manifest.
Kết quả A1/A2 ghi `results/phone_a36/<run>/pin_verify.json`. Verify FAIL → không đo profile đó, báo lại.

## B. Kiểm soát xung nhiệt (kỳ vọng trôi nhiều hơn laptop; không che giấu)
- **Đo theo đợt ngắn, nghỉ giữa đợt.** Một đợt = 10 ảnh × (1 warm-up + 20 lần A/B, ABBA/BAAB như E5), ước ~100–150 giây. Sau mỗi đợt nghỉ R giây, R = 60 s cố định;
  nếu đọc được nhiệt: nghỉ tiếp tới khi mọi vùng đọc được ≤ nhiệt đầu phiên + 1 °C, tối đa 180 s (quá thì ghi "nghỉ không đủ" và vẫn tiếp tục, gắn cờ).
- **Xen kẽ profile giữa các đợt** (thứ tự Latin square, seed 5031) thay vì đo hết một profile mới sang profile khác: nhiệt tích lũy chia đều cho các profile,
  không làm một profile (đo sau cùng) bất lợi hơn.
- **Tham chiếu xung nhịp**: benchmark cố định (như E5) đầu phiên, đầu và cuối mỗi đợt, cuối phiên; cờ nếu lệch > 3% so với đầu phiên.
  **Báo cáo đầu tiên của kết quả phải nêu số đợt bị cờ / tổng số đợt và độ lệch lớn nhất**, không để ở phụ lục.
- **Nhật ký nhiệt** (nếu A0 đọc được): đọc mọi vùng nhiệt đọc được, mỗi 5 giây, ghi `thermal.jsonl` cùng mốc thời gian đo. Nếu không đọc được thì chỉ có tham chiếu xung nhịp (nói rõ).
- **Điều kiện dừng an toàn cho điện thoại hàng ngày của bạn** (ĐÃ CHỐT 01/11/2026: giữ đúng các ngưỡng này): vùng nhiệt đọc được nào ≥ 70 °C, hoặc nhiệt pin ≥ 40 °C → dừng đo, nghỉ tới khi < 55 °C / < 38 °C;
  3 lần dừng liên tiếp → kết thúc phiên, báo lại. Pin phải 30–80% và đang sạc ổn định (sạc nóng pin: nếu pin nóng là lý do dừng, ghi rõ).
- **Phân tích độ trôi**: độ dốc của thời gian tham chiếu theo thứ tự đợt (hồi quy tuyến tính, CI bootstrap); endpoint chính vẫn tính trên tất cả đợt, thêm độ nhạy "bỏ đợt bị cờ" (nếu còn ≥ 30 ảnh).

## Thiết kế đo (giữ giống E5 để so MDE theo cấu trúc, không so tuyệt đối)
- 100 ảnh của E5 (`studies/e5_protocol/run.py:IMAGE_IDS`, seed chọn 20261021). Mỗi ảnh 1 warm-up bỏ + 20 lần đo (ABBA/BAAB); A và B là CÙNG điều kiện (ảnh sạch, τ = 0,01, không cap, IoU 0,45).
- Công đoạn đo riêng: `pre`, `forward`, `decode`, `tau`, `nms_torch_pure`, `nms_tv_nms`, và `total` đo trực tiếp một lần gọi (không cộng trung vị). `perf_counter_ns`, gc tắt trong vùng đo.
- Mỗi profile chạy trong tiến trình riêng ghim TRƯỚC khi torch tạo thread pool (như E5); số luồng intra-op = |S|, interop = 1.
- Dữ liệu trên điện thoại: chỉ trong `~/sponge-sim` (`yolov8n.pt` đã có; 100 ảnh + `images_manifest.json` gồm ID, tên file, SHA256 sẽ đẩy bằng scp). Không tải thêm gì từ mạng.
- Mã mới tách riêng (`studies/e5_phone/`, `src/sponge_sim/hw_linux.py`), vì `hw.py`/`manifest.py` hiện chỉ chạy trên Windows; mã sao chép từ repo cũ giữ nguyên header nguồn.
  Mã mới phải qua smoke trên điện thoại trước (xem dưới) và test nhanh trên laptop (phần không phụ thuộc thiết bị).

## Endpoint và phân tích
- **Chính** (theo profile, cho `forward` và `total`): d_i = ln(median_A,i / median_B,i) trên 100 ảnh; mean d, exp(mean d), CI95% bootstrap percentile theo ảnh
  (20.000 lần, seed 20261021), MDE = (t₀,₉₇₅,₉₉ + t₀,₈,₉₉)·SD(d)/√100 đổi sang %. "CI chứa 1" chỉ dùng để định cỡ nhiễu; **không** diễn giải thành "tương đương".
- **Phụ**: cùng thống kê cho `pre`, `decode`, `tau`, `nms_torch_pure`, `nms_tv_nms`; CV trong ảnh; số đợt bị cờ; độ dốc trôi; kết quả verify A2; tỷ phần NMS/toàn trình (mô tả).
- Không Holm (định cỡ nhiễu, không kiểm định hiệu ứng). Báo kèm hạn chế: một điện thoại, một model, 100 ảnh, proot, torch khác laptop.

## Tiêu chí loại và cổng
- Không loại điểm đo theo giá trị. Chỉ loại ảnh nếu lỗi đọc file (ghi lại).
- Cổng G1: verify ghim đạt (phần A) cho từng profile, không đạt → không đo profile đó. G2: SHA256 `yolov8n.pt` và 100 ảnh khớp laptop, không khớp → dừng.
  G3: smoke không lỗi và không chạm điều kiện dừng nhiệt. G4: bạn xác nhận máy rảnh bằng tin nhắn (xem dưới).

## Quy trình và điều kiện vận hành
1. Bạn duyệt PREREG này (và duyệt A0 nếu đồng ý đọc sysfs/thermal) → mình commit và gắn tag.
2. Smoke nhỏ trên điện thoại: A1/A2 cho từng CPU, 2 ảnh × 2 lần, một đợt ngắn; báo ước lượng thời gian thật (ước tính hiện tại: ~1,5–2 giờ cho 3 profile, sẽ thay bằng số smoke). **DỪNG, chờ bạn.**
3. Bạn xác nhận máy rảnh: đang sạc, không đụng tay, bật Không làm phiền, tắt tiết kiệm pin, **tắt mạng di động, Wi-Fi ở mức tối thiểu (hoặc tắt hẳn nếu SSH không cần liên tục)**;
   bật khóa thức (wake lock) của Termux để Android không treo tiến trình. Mình ghi trạng thái mạng bạn khai vào `manifest.json` (không tự kiểm được).
4. Chạy đầy đủ ở nền, log trong `~/sponge-sim`; kết thúc: kéo kết quả về `sponge-sim/results/phone_a36/`, cập nhật `PROGRESS.md`, tắt sshd (`pkill sshd`).
5. Mọi thao tác trên điện thoại chỉ đọc/ghi trong `~/sponge-sim`; không `rm` ngoài đó; không tắt/khởi động lại máy, không đổi cài đặt Android.

## Rủi ro đã thấy trước
- Android có thể treo/giảm tốc tiến trình nền (Doze, tối ưu pin) → độ trễ tăng đột ngột; giảm bằng wake lock, ghi nhận bằng cờ trôi.
- proot (ptrace) làm lệnh hệ thống chậm hơn; mã đo chủ yếu là tính toán thuần nên ảnh hưởng nhỏ nhưng **chưa định lượng**; không có bản đối chứng không-proot.
- Sạc làm nóng pin, ảnh hưởng ngưỡng nhiệt; bộ điều tốc tần số (governor) không điều khiển được, chỉ ghi lại nếu đọc được.
- Ánh xạ CPU → cụm có thể không xác định được nếu sysfs bị chặn và benchmark không tách rõ hai cụm; khi đó chỉ kết luận ở mức "CPU số k", không nói "lõi hiệu năng".
- SSH giữ Wi-Fi hoạt động khi đo → có thể thêm tải nền nhỏ; chọn tắt Wi-Fi nếu chạy nền (log + lấy kết quả sau).

## Đầu ra
`sponge-sim/results/phone_a36/<run_id>/`: `manifest.json` (git, SHA mã/trọng số/ảnh, phiên bản, `parallel_info`, trạng thái mạng bạn khai), `pin_verify.json`,
`raw_<profile>.jsonl`, `thermal.jsonl` (nếu đọc được), `summary.json`, `run.log`. Mọi số trích `summary.json:<selector>` hoặc raw JSONL.
