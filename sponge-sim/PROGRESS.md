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

- GĐ3 (E5): PREREG_E5 + run/analyze, commit 8afd3d5, tag `prereg-e5-20261021`. Smoke (scratchpad, không báo cáo) chạy được 3 profile,
  ghim verify pass; ước lượng full ≈ 25 phút. Đổi tên thư mục `experiments/` → `studies/` (permission chặn tên `experiments`).
- E2a (thăm dò, lịch sử, chỉ đọc): PREREG commit 9f2e355, tag `prereg-e2a-20261021`. Kết quả `results/e2a/20261021T145544/summary.json`:
  - between_image_clean.rho = 0,9379 [0,9215; 0,9499]; pooled_spearman.rho = 0,6542 [0,628; 0,6767];
  - within_image.*.rho_dN_dt = 0,11–0,13 (sg_ga 0,1214 [0,103; 0,1396]); không đảo dấu (simpson_check: sign_reversal False ×4);
  - historical_noise_clean: median CV trong ảnh 1,1903 nhưng median |ln lệch| 0,0633 → đuôi dày (cú vọt);
    noise_vs_effect.clean_clean_abs_diff_ms median 0,27225, p90 13,57325.
  - POST-HOC (DEVIATIONS.md) `posthoc_aa.json`: A/A share mean 0,0445 [−0,0151; 0,137]; tấn công 0,0533–0,1277 chồng lấn A/A.
  - Diễn giải được phép: N xếp hạng tốt độ trễ NMS GIỮA ảnh (backend custom); trong ảnh, ΔN do patch nhỏ và chìm trong nhiễu
    đo lịch sử → dữ liệu lịch sử không tách được độ trễ do tấn công khỏi nhiễu. KHÔNG nói "N không dự đoán độ trễ".

## Đang dở
- E5 full run: CHỜ DUYỆT + bạn làm checklist điện (xem `python -X utf8 -m sponge_sim.hw` hoặc PREREG_E5).

## Lệnh chạy tiếp
```
cd D:/Sponge-patch-study/sponge-sim
PYTHONPATH=src python -X utf8 -m sponge_sim.hw      # in topology + checklist
# E5 đầy đủ (~25 phút, chạy nền, KHÔNG dùng máy khi đo):
python -X utf8 studies/e5_protocol/run.py --run-dir results/e5/<YYYYMMDDTHHMMSS> > results/e5/<id>.log 2>&1
python -X utf8 studies/e5_protocol/analyze.py results/e5/<id>
```

## Quyết định đang chờ tôi
- Remote: `origin` = sudo-baoz/Physical-Visual-DoS-EdgeAI; chưa push. Chờ bạn chọn remote (reikageisme hay sudo-baoz).
- Bạn tự chạy `powercfg /getactivescheme` và ghi kết quả vào đây (harness không chạy lệnh này). psutil báo cpu_freq max 1300 MHz (giá trị Windows báo, chưa phải xung thực).
