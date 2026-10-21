# PROGRESS — Bước 1 (harness sponge-sim + E5, E1, E2)

Kế hoạch đã duyệt: `C:\Users\Tanh\.claude\plans\warm-tickling-quiche.md` (tóm tắt thứ tự: khung → src+tests → E5 → E1 → E2a/E2b).
Repo thật: `D:/Sponge-patch-study` (CLAUDE.md còn ghi đường dẫn cũ). Nhánh: `harness/step1` (tách từ `codex/repository-organization`).
Môi trường: Python 3.14 hệ thống (không .venv), torch 2.13.0+cpu, torchvision 0.28.0, ultralytics 8.4.90, psutil 7.2.2, scipy 1.18.0, pytest 9.1.1.

## Đã xong
- GĐ1: khung thư mục, `src/sponge_sim/hw.py` (topology Win32, profile, pin + verify_pin bằng tải torch, power snapshot read-only, CHECKLIST).
  - Topology đọc từ OS (21/10/2026): P-core = core 0 {0,1}, core 1 {2,3} (SMT); E-core = logical 4..11 (không SMT).
  - Profile: P1=[0], E1=[4], E4=[4,5,6,7]. verify_pin (threads = 2×|cpus|): effective_cores P1 0,90 / E1 0,95 / E4 3,48 (≤ |cpus|+0,15) → passed; không ghim (12 thread) 8,69.
  - Lưu ý: verify bằng thread Python thuần bị GIL làm sai → đã đổi sang torch matmul.

- GĐ2: `provenance.py` (copy có header nguồn + SHA), `backends.py` (torch_pure, tv_nms, tv_batched, ultra_native; prepare() ngoài vùng đo),
  `stats.py`, `cohorts.py`, `inject.py`, `timing.py`, `manifest.py`; tests 37/37 pass (`python -X utf8 -m pytest tests -q -p no:cacheprovider`, ~50 s).
  - test_regression_x1: OLD128 p=0,150390625; X1-500 diff −0,029335098…, CI [−0,038328; −0,021598], p=0,001953125 (khớp 1e-12).
  - test_regen_counts: forward mới (6 ảnh × clean, sg_ga q2500 seed0) khớp active/final_detections của record và raw_cache (atol 1e-4).
  - Phát hiện: pipeline resize 320×320 → head YOLOv8n có 2100 anchor (không phải 8400). Thang N E1 sẽ theo A=2100 (+ A=8400 nhãn "đầu vào 640").
  - Ultralytics 8.4.90 NMS: dùng torchvision.ops.nms khi torchvision đã import; time limit 2,0+0,05·bs s; max_nms 30000; max_det 300.

## Đang dở
- GĐ3: PREREG_E5 → tag → smoke → DỪNG chờ duyệt.

## Lệnh chạy tiếp
```
cd D:/Sponge-patch-study/sponge-sim
PYTHONPATH=src python -X utf8 -m sponge_sim.hw      # in topology + checklist
```

## Quyết định đang chờ tôi
- Remote: `origin` = sudo-baoz/Physical-Visual-DoS-EdgeAI; chưa push. Chờ bạn chọn remote (reikageisme hay sudo-baoz).
- Bạn tự chạy `powercfg /getactivescheme` và ghi kết quả vào đây (harness không chạy lệnh này). psutil báo cpu_freq max 1300 MHz (giá trị Windows báo, chưa phải xung thực).
