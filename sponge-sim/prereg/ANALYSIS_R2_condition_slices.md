# ANALYSIS R2 — Lát cắt bản đồ điều kiện từ dữ liệu lịch sử OLD128 (THĂM DÒ, chỉ đọc)

Ngày: 21/10/2026. Nhãn bắt buộc khi trích: **thăm dò, dữ liệu lịch sử OLD128, không ghim core**.
REVIEW/04 §1 đã báo tương quan Pearson gộp (custom r = 0,62098; torchvision r = 0,12163) nên phân tích này KHÔNG mù dữ liệu.
Không p-value, không Holm; chỉ ước lượng + CI bootstrap cụm theo ảnh (16 ảnh → CI rộng, ghi rõ).
Script: `sponge-sim/studies/r2_slices/run.py`. Đầu ra: `results/r2/<run_id>/summary.json`.

## Dữ liệu
1. `outputs/extended_q1_20261002/backend_qos/backend.json:records` — 16 ảnh × 10 seed × {clean, sg_ga} × backend
   {custom, torchvision} × (τ, cap) ∈ {(0,01, None), (0,01, 100), (0,25, None)}; 5 mẫu/record; cùng decoded predictions.
2. `.../backend_qos/timing_seed{s}_threads{t}.json:records` — t ∈ {1,2,4}; 5 điều kiện × 16 ảnh × 3 lần.

## Phân tích
A. Theo ô (backend, τ, cap): t = trung vị 5 mẫu `nms_ms`; N = `nms_input_candidates`.
   (i) ρ_Spearman(N, t) gộp clean+sg_ga, CI bootstrap cụm theo ảnh (2.000 lần, seed 20261021);
   (ii) trung vị t, và tỷ số trung vị t của custom / torchvision cùng ô;
   (iii) trong ảnh: ΔN = N_sg − N_clean (cùng ảnh, seed), Δt tương ứng; ρ(ΔN, Δt), CI cụm theo ảnh;
   (iv) tỷ số t_sg / t_clean theo (ảnh, seed), trung bình, CI hai tầng (ảnh × seed), 20.000 lần.
B. Theo số thread (timing): tỷ phần (filter+NMS) / total_prepared_tensor theo record (trung vị, p95);
   ρ(N, nms_ms) và ρ(N, total) gộp clean+sg_ga, CI cụm theo ảnh; tỷ số total sg/clean (trung vị lần lặp, theo ảnh × seed),
   CI hai tầng.
C. Bảng ô: liệt kê các ô CÓ dữ liệu và các trục còn THIẾU của bản đồ C (backend × class-aware/agnostic × cap × τ × core × chu kỳ).

## Diễn giải được phép
- Nếu ρ khác nhau giữa backend trên cùng input: "mức N xếp hạng thời gian NMS phụ thuộc backend trong dữ liệu này".
- Không suy "N không dự đoán độ trễ". Không suy bậc phức tạp. Không dùng làm bằng chứng xác nhận (E1/E2b mới là xác nhận).
