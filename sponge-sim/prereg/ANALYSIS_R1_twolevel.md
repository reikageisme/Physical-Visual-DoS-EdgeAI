# ANALYSIS R1 — Độ bất định hai tầng (ảnh × seed) cho so sánh chính SG-GA − GA tâm (POST-HOC, chỉ đọc)

Ngày: 21/10/2026. Trạng thái: **POST-HOC** (kết quả theo seed của OLD128 và X1-500 đã biết; CLAUDE.md "Kết quả đã xác minh").
Không phải PREREG xác nhận; không gắn tag. Kế hoạch này được commit TRƯỚC khi chạy script để cố định phép tính.
Script: `sponge-sim/studies/r1_twolevel/run.py`. Đầu ra: `sponge-sim/results/r1/<run_id>/summary.json` + `manifest.json`.

## Lý do
CI hiện có (bootstrap theo seed, 10 seed) điều kiện trên bộ ảnh cố định, không phủ độ bất định do chọn ảnh
(REVIEW/04 §2 "Statistics/power"; skill sponge-research §2). Phân tích này bổ sung CI phủ cả hai nguồn.

## Dữ liệu
`outputs/{extended_q1_20261002, x1_500img_20261018}/evaluation/{case}_seed{0..9}_q{q}.json`, record threshold == 0,01,
pre_nms_cap is None. case ∈ {clean (q0), sg_ga, ga_center, ga_random_location, random_search (q2500)}.
Ảnh: giao các image_id có mặt ở mọi case/seed của cohort (kỳ vọng 128 và 500).

## Đại lượng
Với ảnh i, seed s, phương pháp m: r_m[i,s] = N_m[i,s] / max(N_clean[i,s], 1). Ma trận chênh lệch
D[i,s] = r_SG[i,s] − r_ref[i,s]. Thống kê = trung bình toàn ma trận (bằng trung bình theo seed của trung bình theo ảnh,
vì ma trận cân bằng) → phải tái tạo đúng số theo seed đã xác minh (kiểm khớp 1e-12, fail-closed).

## Phép so (định trước trong file này)
- Chính (nhắc lại): ref = ga_center.
- Phụ: ref = ga_random_location; ref = random_search. Mô tả, không p-value mới, không Holm mới (Holm đã có ở analysis.json).

## CI (20.000 lần, seed RNG 20261021, percentile 95%)
1. `seed_only`: rút lại 10 seed (cột), giữ ảnh — tái tạo CI đã công bố (seed RNG khác → chỉ so gần đúng, ghi cả hai).
2. `image_only`: rút lại ảnh (hàng), giữ 10 seed.
3. `two_level`: rút lại hàng và cột độc lập (pigeonhole bootstrap, Owen 2007, Ann. Appl. Stat. 1(2):386–411).
Báo thêm: độ rộng CI mỗi loại, tỷ lệ độ rộng two_level / seed_only; tỷ phần ảnh có trung bình theo seed của D < 0.

## Quy tắc diễn giải (định trước)
- Nếu CI two_level của X1-500 không chứa 0: "chênh lệch âm vẫn giữ khi tính cả độ bất định chọn ảnh (trong quần thể
  500 ID nhỏ nhất COCO val2017)". Nếu chứa 0: báo nguyên trạng, kết luận X1-500 hạ xuống "chưa đủ bằng chứng khi tính
  độ bất định ảnh".
- 500 ID nhỏ nhất không phải mẫu ngẫu nhiên của COCO; CI two_level phủ biến thiên lấy mẫu lại trong tập này, không
  bảo đảm khái quát ra ngoài.
- OLD128 ⊂ X1-500: hai phân tích không độc lập.
