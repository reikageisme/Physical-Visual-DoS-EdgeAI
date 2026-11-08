# Profile CPU thứ hai trên Fedora (CHỈ CHUẨN BỊ — chưa chạy, chưa đưa vào bài)

Mục đích: lặp lại A/A và phép tiêm toàn pipeline (E9) trên một máy Linux với governor `performance` và tần số khoá,
cùng giao thức ghim/kiểm chứng (ghim một CPU logic, `verify_pin` dưới tải, A/A, điểm tham chiếu drift 3%).
Chỉ đưa vào bài khi đã có dữ liệu và tệp đăng ký trước riêng.

## 0. Trước khi đo (người dùng tự làm; harness không đổi cấu hình)
1. Đăng ký trước: soạn `prereg/PREREG_E12_fedora.md` (giả thuyết: N* ở Fedora cùng bậc với laptop; endpoint: MDE A/A toàn trình,
   N* theo cài đặt × hình học), commit và `git tag prereg-e12-<ngày>` TRƯỚC khi đo.
2. Cài môi trường giống laptop (python, torch, torchvision, ultralytics, psutil, numpy; ghi phiên bản vào manifest tự động).
   Chép `yolov8n.pt` (kiểm SHA-256 với manifest của laptop) và thư mục ảnh `outputs/datasets/coco_val2017_extended`.
3. Đọc cấu trúc CPU (chỉ đọc):
   ```bash
   python3 studies/linux_profile/topology_fedora.py > results/e12/fedora_topology.json
   ```
4. Tự đặt (cần quyền root, ngoài phạm vi harness), rồi ghi lại vào `PROGRESS.md`:
   ```bash
   sudo cpupower frequency-set -g performance
   sudo cpupower frequency-set -d <F> -u <F>        # khoá min = max = F (chọn F ≤ tần số cơ sở, không turbo)
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo   # máy Intel; AMD: /sys/devices/system/cpu/cpufreq/boost = 0
   ```
   Chạy lại bước 3 và kiểm `governor=performance`, `min_khz == max_khz` ở CPU sẽ ghim.
5. Máy rảnh: cắm sạc, đóng ứng dụng, không đăng nhập đồ hoạ nếu được, không chạy gì khác (quy tắc 9).

## 1. Profile
`hw_linux.profiles()` đọc profile từ biến môi trường. Dùng profile đề xuất ở bước 3 (CPU logic đầu tiên của lõi vật lý thứ hai):
```bash
export SPONGE_PROFILES='{"P1_cpu2": [<cpu>]}'
```
Lưu ý: `hw_linux.cpu_topology()` không đọc SMT; sibling của CPU được ghim nằm trong `fedora_topology.json` và phải ghi vào bài
(giống laptop: sibling không bị chặn trừ khi người dùng tự offline nó bằng `echo 0 | sudo tee /sys/devices/system/cpu/cpu<sib>/online`,
việc này phải ghi vào tệp đăng ký trước).

## 2. Lệnh đo (sau tag; mỗi lệnh chạy một mình, không chạy song song)
```bash
python3 -X utf8 studies/e5_protocol/run.py --profiles P1_cpu2 --run-dir results/e12/AA_fedora_<ngày>
python3 -X utf8 studies/e5_protocol/run.py --profiles P1_cpu2 --canvas 640 --run-dir results/e12/AA640_fedora_<ngày>
python3 -X utf8 studies/e9_pipeline/run.py --run-dir results/e12/E9_fedora_<ngày>
```
Ước tính thời gian: tương tự laptop (A/A 320 px khoảng 30–40 phút, 640 px khoảng 1 giờ, E9 vài giờ); chạy smoke (`--smoke`) trước để đo thực tế.

## 3. Phân tích
Dùng đúng script của laptop (`studies/e5_protocol/analyze.py`, `studies/e9_pipeline/analyze.py`, `studies/amdahl/analyze.py`),
với MDE của A/A Fedora. Bảng của bài chỉ thêm cột Fedora sau khi có dữ liệu, qua `paper_vi/tables/make_tables.py`.

## Giới hạn
`hw_linux.py` được viết cho điện thoại (proot); trên Fedora cần smoke để kiểm `verify_pin` và `pin` (đọc lại affinity) trước khi đo thật.
