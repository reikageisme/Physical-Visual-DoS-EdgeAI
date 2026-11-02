# DEVIATIONS — sai lệch so với PREREG

Mỗi mục: ngày, PREREG nào, thay đổi gì, lý do, đã thấy dữ liệu đo chính chưa.

(chưa có)

## 2026-10-21 — E2a: phân tích A/A post-hoc
- PREREG: `prereg/PREREG_E2a.md` (tag `prereg-e2a-20261021`).
- Thêm `studies/e2_proxy/e2a_posthoc_aa.py` -> `results/e2a/20261021T145544/posthoc_aa.json`.
- Lý do: sau khi thấy `summary.json:attack_induced_share` (mean dương ở mọi case dù mean ΔN ≈ 0 hoặc âm) và
  `noise_vs_effect` (clean–clean p90 13,6 ms), nghi chỉ số bị chi phối bởi cú vọt thời gian và thứ tự đo; cần A/A cùng công thức.
- Đã thấy dữ liệu: CÓ. Gắn nhãn POST-HOC; không thay kết quả định trước.
