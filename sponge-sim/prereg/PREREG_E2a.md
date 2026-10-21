# PREREG E2a — Proxy N_tau trên dữ liệu lịch sử X1-500 (THĂM DÒ, chỉ đọc)

Ngày: 21/10/2026. Tag: `prereg-e2a-20261021`. Script: `sponge-sim/studies/e2_proxy/e2a_historical.py`.
Nhãn bắt buộc khi trích: **thăm dò, dữ liệu lịch sử**: không ghim core (2 intra-op thread, `extended_evaluation.py:206`),
backend NMS `custom` (vòng lặp PyTorch), 1 mẫu NMS/record, forward = trung vị 3 lần. Không dùng làm bằng chứng xác nhận;
xác nhận thuộc E2b theo giao thức E5.

## Đã biết trước khi phân tích
Trước khi viết file này đã xem: (1) record schema; (2) clean seed0 vs seed1 có cùng active_candidates cho 500/500 ảnh,
nms_ms khác nhau (Pearson 0,74; trung vị |ln tỷ số| 1,61), tức clean được đo lại độc lập mỗi seed.
Chưa xem bất kỳ tương quan N–độ trễ nào của X1-500.

## Dữ liệu
`outputs/x1_500img_20261018/evaluation/{case}_seed{0..9}_q{q}.json`, record threshold == 0,01 và pre_nms_cap is None.
Case: `clean` (q0), `sg_ga`, `ga_center`, `ga_random_location`, `random_search` (q2500). 500 ảnh × 10 seed × 5 case.

## Phân tích (định trước)
1. **Nhiễu đo lịch sử (A/A tự nhiên)**: clean, mỗi ảnh có 10 lần đo NMS với N giống hệt nhau.
   CV trong-ảnh của nms_ms (trung vị theo ảnh), và trung vị |ln(nms_seed_s / median_ảnh)|.
2. **Tương quan gộp**: Spearman ρ(N, nms_ms) trên mọi record của 5 case; CI bootstrap cụm theo ảnh.
3. **Giữa ảnh**: trên clean, ρ giữa N của ảnh và trung vị nms_ms 10 seed của ảnh; CI bootstrap theo ảnh.
4. **Trong ảnh (Simpson)**: với mỗi case tấn công c và seed s: ΔN = N_c,s − N_clean,s ; Δt = nms_c,s − nms_clean,s
   (clean cùng seed, cùng ảnh). ρ(ΔN, Δt) gộp mọi (ảnh, seed) cho từng case; CI bootstrap cụm theo ảnh.
   Simpson được gọi là có nếu dấu của ρ trong-ảnh khác dấu ρ giữa-ảnh; nếu chỉ khác độ lớn thì báo là khác độ lớn.
5. **Headroom H2**: theo record, h = nms_ms / (forward_ms + postprocess_ms). Báo trung vị và phân vị 95 theo case.
   Phần độ trễ do tấn công: g = (nms_c − nms_clean) / (forward_clean + postprocess_clean) cùng ảnh, cùng seed;
   trung bình theo ảnh trong từng seed → ma trận [ảnh, seed] → CI bootstrap hai tầng (ảnh × seed).
6. **So nhiễu với hiệu ứng**: so |Δt| do tấn công với độ phân tán của nms_ms ở clean cùng ảnh (mục 1), mô tả.

Bootstrap: 20.000 lần, seed 20261021 (2.000 lần cho Spearman cụm vì chi phí; ghi rõ trong summary).
Không p-value, không Holm (thăm dò). Không kết luận "N không dự đoán độ trễ" từ dữ liệu này.

## Đầu ra
`sponge-sim/results/e2a/<run_id>/summary.json` + `manifest.json`.
