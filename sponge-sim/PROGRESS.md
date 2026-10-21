# PROGRESS — Bước 1 (harness sponge-sim + E5, E1, E2)

Kế hoạch đã duyệt: `C:\Users\Tanh\.claude\plans\warm-tickling-quiche.md` (tóm tắt thứ tự: khung → src+tests → E5 → E1 → E2a/E2b).
Repo thật: `D:/Sponge-patch-study` (CLAUDE.md còn ghi đường dẫn cũ). Nhánh: `harness/step1` (tách từ `codex/repository-organization`).
Môi trường: Python 3.14 hệ thống (không .venv), torch 2.13.0+cpu, torchvision 0.28.0, ultralytics 8.4.90, psutil 7.2.2, scipy 1.18.0, pytest 9.1.1.

## Đã xong
- GĐ1: khung thư mục, `src/sponge_sim/hw.py` (topology Win32, profile, pin + verify_pin bằng tải torch, power snapshot read-only, CHECKLIST).
  - Topology đọc từ OS (21/10/2026): P-core = core 0 {0,1}, core 1 {2,3} (SMT); E-core = logical 4..11 (không SMT).
  - Profile: P1=[0], E1=[4], E4=[4,5,6,7]. verify_pin (threads = 2×|cpus|): effective_cores P1 0,90 / E1 0,95 / E4 3,48 (≤ |cpus|+0,15) → passed; không ghim (12 thread) 8,69.
  - Lưu ý: verify bằng thread Python thuần bị GIL làm sai → đã đổi sang torch matmul.

## Đang dở
- GĐ2: provenance.py, backends.py, stats.py, inject.py, timing.py, manifest.py + 4 test.

## Lệnh chạy tiếp
```
cd D:/Sponge-patch-study/sponge-sim
PYTHONPATH=src python -X utf8 -m sponge_sim.hw      # in topology + checklist
```

## Quyết định đang chờ tôi
- Bạn tự chạy `powercfg /getactivescheme` và ghi kết quả vào đây (harness không chạy lệnh này). psutil báo cpu_freq max 1300 MHz (giá trị Windows báo, chưa phải xung thực).
