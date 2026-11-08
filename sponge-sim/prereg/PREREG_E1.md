# PREREG E1 — Đối chứng dương và dose-response tại ranh giới decoded → NMS

Trạng thái: **NHÁP, chờ duyệt** (chưa tag). Tag dự kiến sau khi duyệt: `prereg-e1-<ngày>`.
Script: `sponge-sim/studies/e1_dose/run.py` (đo; `--phone` cho điện thoại), `analyze.py` (phân tích). Cấu hình: `CONFIG`/`PHONE` trong `run.py`.
Yêu cầu thầy (`docs/REVIEW_THAY_NAM.md`): "tiêm N hộp tổng hợp từ 10² đến 10⁴ vào cùng pipeline, đo trên laptop và ít nhất một thiết bị ARM".

## Mục đích
Chứng minh testbed nhạy với tải NMS (đối chứng dương) và lập đường độ trễ theo N_tau cho từng backend × hình học × cap,
TÁCH khỏi forward. Đây là trục "backend × class-aware/agnostic × cap" của bản đồ điều kiện C và là nền cho luận điểm A:
cùng N_tau, độ trễ khác nhau theo backend/hình học thì N_tau một mình không đủ dự đoán độ trễ.
Không có patch, không quang học; ứng viên được tiêm tổng hợp (`src/sponge_sim/inject.py`).

## Phạm vi
Laptop: CPU ghim core (i5-1335U), tensor chuẩn bị sẵn, YOLOv8n đầu vào 320×320 (head 2100 anchor). Lưới A = 8400 (đầu vào 640) và lưới A = 10000
("đầu giả", vượt mọi head thật của YOLOv8n ở 320 và 640 px) chỉ là tham chiếu ngoài pipeline thật để phủ N đến 10⁴; gắn nhãn riêng, không trộn với A = 2100 trong một bảng.
Điện thoại ARM (Galaxy A36, Snapdragon 6 Gen 3) trong proot Ubuntu: xem mục "E1 trên điện thoại". Không có patch, không có camera.

## Đầu vào từ E5 (`results/e5/20261021T151412/summary.json`) và A/A bổ sung
- Profile chính: **P1_cpu2** = logical {2} (logical đầu của P-core thứ hai; tránh CPU 0 vì smoke E1 cho effective_cores 0,394 ở CPU 0).
  Profile phụ: **E1** = logical {4}. Quyết định của nhóm ngày 21/10/2026 (DEVIATIONS.md).
- **E4 không dùng** trong E1 và **không dùng để kết luận về số core**: trong E5, 4 thread của E4 không chạy song song
  (effective_cores 0,963; DEVIATIONS.md). Việc điều tra nguyên nhân nằm trong PROGRESS.md.
- MDE của P1_cpu2: lấy từ run A/A bổ sung chạy NGAY TRƯỚC E1 bằng đúng giao thức E5
  (`studies/e5_protocol/run.py --profiles P1_cpu2`, 100 ảnh × 20 lần), selector
  `summary.json:profiles.P1_cpu2.stages.<stage>.mde_ratio_pct`. Quy tắc định trước: ngưỡng Chính 1 = 1 + MDE(nms_torch_pure)/100 của run đó.
  Nếu verify ghim của run A/A FAIL (cận dưới 0,5) → không chạy E1, báo lại.
  Phần được đăng ký là QUY TẮC (ngưỡng = 1 + MDE của run A/A hợp lệ chạy ngay trước E1), không phải một con số. Con số MDE sẽ được ghi vào `PROGRESS.md` ngay sau run A/A và trước khi chạy E1; đây không phải sai lệch.
  `[CHƯA CÓ SỐ: chờ A/A P1_cpu2 hợp lệ]`
- MDE của E1 (từ E5): `nms_torch_pure` 2,97%, `nms_tv_nms` 2,05%, `tau` 1,67%, `total` 1,27%.
  Chênh lệch tương đối nhỏ hơn MDE của profile/công đoạn tương ứng KHÔNG được diễn giải (kể cả "không có khác biệt").
  Với backend chưa có MDE riêng (tv_batched, ultra_native), dùng MDE lớn nhất của profile đó trong các công đoạn NMS đã đo.
- Cờ trôi 3% ghi lại để mô tả, KHÔNG dùng để loại dữ liệu (E5 cho thấy ngưỡng này kích hoạt ở gần như mọi khối).
- Verify ghim: cận trên như E5 **và** cận dưới effective_cores ≥ 0,5 cho profile một core.
- Chế độ điện: Windows "Balanced" (`381b4222-f694-41f0-9685-ff5bb260df2e`), giữ nguyên cho mọi lần đo.

## Thiết kế
- Lưới A = 2100: N ∈ {0, 10, 30, 100, 300, 600, 1000, 1500, 2100}, nc ∈ {80, 1}. Lưới A = 8400: N ∈ {0, 100, 300, 1000, 3000, 8400}, nc = 80.
  Lưới A = 10000 (đầu giả): N ∈ {100, 300, 1000, 3000, 10000}, nc = 80 (phủ 10²–10⁴ theo yêu cầu thầy).
  Điểm 600/1500 đặt trước để dày quanh vùng knee dự kiến (không chọn bằng pilot).
- Hình học: disjoint (IoU = 0, NMS giữ hết), dense_cluster (8 cụm, IoU cao), mixed (nửa/nửa). N = 0 chỉ một ô.
- Cap ∈ {None, 300} (top-k theo score trước NMS); ô cap = 300 chỉ khi N > 300.
- Backend: torch_pure (agnostic, vòng lặp Python, bản copy pipeline cũ), tv_nms (agnostic), tv_batched (class-aware),
  ultra_native (class-aware, mặc định Ultralytics 8.4.90: max_det 300, max_nms 30000, time limit 2,0 + 0,05·bs s).
- Seed tiêm: `inject_seed_base + crc32("A|N|nc|geometry") mod 100000`; SHA256 tensor ghi theo từng phép đo.
- Lặp: 3 warm-up + 30 lần/ô, ở MỌI N (không giảm lặp ở N cao). Ước lượng từ hiệu chỉnh một lần gọi (21/10/2026,
  không ghim, không báo cáo): torch_pure disjoint ≈ 0,36 s ở N = 2100 và ≈ 2,1 s ở N = 8400 → ~6–15 phút cho P1_cpu2,
  ~20–40 phút cho E1; tổng ≈ 30–60 phút (+ 5 phút A/A). Sau khi thêm lưới A = 10000 (torch_pure disjoint ≈ 3 s/lần ở N = 10⁴):
  ước lượng lại **P1_cpu2 ≈ 50–90 phút**; profile E-core (chậm hơn ~3× theo E5) ≈ 2–3 giờ nên là **tùy chọn, chạy sau cùng nếu còn giờ**. Trong mỗi nhóm (A, N, nc, geometry, cap) đo xen kẽ 4 backend vòng tròn, thứ tự theo hình vuông
  Latin (seed 1023); thứ tự nhóm hoán vị (seed 1022); tham chiếu đầu mỗi 12 nhóm và cuối phiên; thứ tự profile seed 1021.
  Mỗi lần đo ghi riêng `tau_ns` (lọc tau trên đủ A hàng) và `nms_ns` (prepare() ngoài vùng đo).
- Phần B (toàn trình, thứ cấp): ảnh sạch ID 139, forward thật + ứng viên tiêm (A = 2100, disjoint, nc = 80, cap None)
  thay decoded → tau → NMS, đo trực tiếp một lần gọi; N ∈ {0, 100, 300, 1000, 2100}; torch_pure và ultra_native; 3 warm-up + 20 lần.
  Hạn chót 100 ms (chu kỳ 10 FPS): đây là **ngưỡng đặt trước của nhóm** để có một đối chứng dương cụ thể, KHÔNG phải chuẩn
  ngành hay yêu cầu của hệ thống thật; kết luận về "vượt hạn chót" chỉ có nghĩa với ngưỡng này.

## Endpoint
- **Chính 1 (đối chứng dương độ trễ NMS)**: P1_cpu2, torch_pure, disjoint, A = 2100, nc = 80, cap None:
  R = median(nms | N = 2100) / median(nms | N = 10). CI95% bootstrap percentile (rút lại lần lặp trong từng ô, 20.000, seed 20261021).
  Đạt nếu cận dưới CI > 1 + MDE(P1_cpu2, nms_torch_pure) của run A/A bổ sung.
- **Chính 2 (đối chứng dương vượt hạn chót)**: P1_cpu2, phần B, torch_pure, N = 2100: tỷ lệ lần lặp có total > 100 ms, kèm CI Wilson.
  Đạt nếu ≥ 50%. Không đạt thì báo nguyên trạng: testbed không tạo được deadline miss ở mức tải tối đa mà head 320 cho phép.
- **Phụ (mô tả, không p-value)**, theo profile × backend × hình học × nc × cap:
  (a) đường median nms theo N, CI bootstrap theo lần lặp; (b) độ dốc log–log trên N ≥ 100 kèm CI bootstrap: chỉ mô tả, KHÔNG suy bậc
  phức tạp; (c) tỷ số backend / tv_nms tại cùng N (cùng đầu vào); (d) tỷ số hình học: disjoint / dense_cluster tại cùng N, backend;
  (e) tỷ số cap300 / capNone tại N ≥ 600; (f) tỷ số nc=80 / nc=1 cho backend class-aware; (g) phần B tại mọi N và ultra_native;
  (h) số lần nms_ns > 2,05 s (chạm time limit của Ultralytics).
- **Chính 3 (phủ 10²–10⁴, theo yêu cầu thầy)**: P1_cpu2, A = 10000, nc = 80, cap None, cho `torch_pure` và `tv_nms`, hình học disjoint:
  R₁₀₄ = median(nms | N = 10⁴) / median(nms | N = 10²); CI95% bootstrap (cách như Chính 1). Báo cho cả ba hình học và cả 4 backend ở mức mô tả.
  **Quy tắc đọc kết quả (không phải pass/fail):** nếu cận dưới CI của R₁₀₄ > 1 + MDE(profile, công đoạn NMS tương ứng) thì ghi "phát hiện sự phụ thuộc vào N ở độ nhạy này"; ngược lại ghi "không phát hiện, MDE = x%". Không dùng "không phụ thuộc" hay "tương đương".
- **Chính 4 (thiết bị ARM)**: điện thoại, profile nhanh đạt verify (mục dưới), `torch_pure` và `tv_nms`, disjoint: R₁₀₄ như trên với CI95%.
  So với laptop **chỉ theo tỷ số và hình dạng đường cong**, không so độ trễ tuyệt đối (khác máy, khác bản torch).
  Cùng quy tắc đọc kết quả như Chính 3, với MDE của profile điện thoại lấy từ A/A E5-phone.
- Câu hỏi A định trước (mô tả): tại cùng N_tau, khoảng chênh độ trễ giữa backend và giữa hình học có vượt MDE hay không.
  Nếu có: bằng chứng N_tau một mình không xác định độ trễ NMS. Nếu không: báo nguyên trạng.

## Tiêu chí loại
Không loại theo giá trị. Profile verify FAIL → giữ raw, gắn nhãn, không dùng cho endpoint chính.

## Smoke (trước khi đo chính)
`run.py --smoke` (2 lặp, 6 nhóm/profile) vào scratchpad, chỉ kiểm script + ước lượng thời gian; dừng chờ duyệt trước khi chạy đầy đủ.

## Đầu ra
`sponge-sim/results/e1/<run_id>/`: manifest.json, raw_{P1_cpu2,E1}.jsonl, summary.json, run.log.

## E1 trên điện thoại (ARM) — dùng PREREG_E5_phone
- Điều kiện vận hành, ghim core, kiểm soát nhiệt, tắt mạng: đúng như `PREREG_E5_phone.md` (đã duyệt); profile chỉ lấy từ các profile ĐẠT `pin_verify.py`.
  Tên profile cụ thể (vd. `big1`, `little1`, hoặc `cpuK` nếu cụm không xác định) được chốt từ `pin_verify.json` trước khi đo và ghi vào PROGRESS.
- Lưới gọn (`run.py --phone`): A = 2100, N ∈ {100, 600, 2100}; A = 10000, N ∈ {100, 300, 1000, 3000, 10000}; nc = 80; hình học disjoint/dense_cluster/mixed;
  cap None; backend `torch_pure`, `tv_nms`; 2 warm-up + 10 lần/ô; không có phần B (toàn trình). Xen kẽ backend vòng tròn, thứ tự nhóm hoán vị, tham chiếu xung nhịp như laptop.
- Nghỉ giữa các phép đo: orchestrator của E1 chạy theo từng profile; vì nhiệt, **nghỉ ≥ 60 s giữa hai profile** và giữ ghi nhiệt nếu A0 cho phép. Cờ trôi báo ở đầu kết quả.
- Ước lượng: ≈ 15–30 phút mỗi profile (chưa đo; sẽ thay bằng số smoke). MDE của điện thoại lấy từ A/A E5-phone, không dùng MDE laptop.
- Diễn giải: ARM ở đây là **điện thoại có SoC Arm**, không phải board nhúng; mọi trích dẫn phải ghi đúng như vậy.
