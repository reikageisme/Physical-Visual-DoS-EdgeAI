# CPU laptop segmentation A/A smoke — prospective07/11/2026

Trạng thái: READY FOR TAG, chưa có timing smoke mới lúc viết. User yêu cầu
“code và thử phép đo”, cho phép xử lý giới hạn hiệu năng07/11. Biết trước
M1/M1b/M6 failures; không lặp campaign cũ tới khi PASS. Không benchmark tấn công
trong lượt này. Đây là smoke kiểm code/khả thi, không phải tái lập main study.

## Khóa thiết bị, phần mềm, đầu vào

- Laptop i5-1335U, Windows11; API topology10physical/12logical: P-cores[0,1]
  và[2,3], E logical4..11. Hai profile cố định tuần tự: P1_cpu2=[2],P2=[0,2].
  Không chọn profile bằng hiệu ứng; cả hai được báo dù FAIL. Một lần mỗi profile.
- Fresh process, process affinity trước nhậpTorch; intraop1/2 tương ứng,
  interop1; OMP/MKL/OPENBLAS threads khóa theo profile. Pin-load QA3s dùng
  2×|S|threads, sau đó restore|S|. Giữ effective upper|S|+.15, lower.5 với1CPU,
  .7|S| với2CPU và allowedCPU utilization>=70% theo hw.verify_pin; QA ở đầu/cuối.
- torch2.13.0+cpu,torchvision.28.0+cpu,Ultralytics8.4.90. Fixed compiled
  torchvision.ops.nms qua locked Ultralytics wrapper, class-aware, nc80,
  conf.25,IoU.45,maxdet300,maxnms30000,CPU FP32 unfused eval, batch1.
- Hai input clean đầu tiên theo thứ tự M3full20:000000000009,000000000025.
  Tensor[1,3,640,640]float32 lấy byte-exact từ archive verified
  confirmation_20261106/codex_evidence_20261106.zip, member
  m3_native_full_20261106/<id>_inputs.pt. Archive chứa cảadv nhưng smoke KHÔNG
  forwardadv. Weights yolov8n-seg SHA a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c.
- lock.json ghiSHA code/config/input/installednms+ops và version. Manifestrun
  ghi tag/commit/lockSHA/weight+inputSHA/profile/power/parallel environment.
  Runpathmới, exclusive-create, giữ mọi raw/error/incomplete. Không tải/cài gói.

## Nguồn điện và quyền xử lý giới hạn

User nhớ đã Performance: readback đúng configuredACBestPerformance GUID
ded574b5-45a0-4f42-8737-46345c09c238; effective-overlay trả961cc777-2547-4f9d-8174-7d86181b8a7a.
Đã reapply configuredACBestPerformance bằng API Microsoft07/11, return0,
configured không đổi và effective vẫn như cũ. Pin100%,ACLineStatus1,
SystemStatusFlag0, nềnBalanced,minAC5/maxAC100. Không kết luận nguyên nhân
override hoặc xung thực từpsutil1300MHz. Chưa đổi baseplan/turbo/thermal limits.
Đây là điều kiện MỚI so phép đo cũ; không dùng để chứng minh failures cũ do power.
Source/receipt power_mode_before.json,power_reapply.json trongresults/laptop_seg_20261107.

Giữ nguyên trạng thái này suốt smoke; đổi thêm cấu hình phải kết thúc và
amendment/tag trước lượt khác. AC required trước/từng image/cuối. Không có
telemetry nhiệtCPU hợp lệ trong PythonWindows, ghi missing thay vì suy nhiệt
thấp; giới hạn runtime/small clean workload, không chạm bảo vệ nhiệt.

## Timing và thứ tự cố định

- Settle60s trướcprofileđầu, nghỉ60s giữa2profiles; worker timeout600s/profile.
- Seed20261107. Global20forward warmups. Trước mỗi reference3warmforward,
  reference15samples của ảnhclean đầu; tại start, trước từng2image,end:4reference.
- Mỗi image2warmforward+total;8repeats, mỗi repeat forwardA/B và totalA/B.
  A/B cùng tensor sạch/cùng hàm. Hai stage đổi vị trí theo block2repeats;
  A/B luân phiên, mỗi label xuất hiện đều mọi vị trí qua8repeats.
- `total`=forward→Ultralytics filtering/NMSfixedTV→process_mask_native640²
  trên detections thật. Không cộng stage medians; imagepreprocess/tensorload ngoài
  timing. GC-off đúngvùng perf_counter_ns, CPU không có CUDA async cần đồng bộ.
  Rawbuffer flush giữaimage; capturetimestamp/order/counts. Khóa source trước tag.
- Chỉ campaignworker hoạt động, parent block wait. Không web/git/phân tích/test/
  đọc log/poll tiến độ trong thời gian đo. Chỉ đọc sau campaign_done.

## Endpoint và quyết định

Feasibility gate: DONE+đủ2image×8repeats×2labels×2stages;4references và
KHÔNGreference nào abs(median/ref0−1)>.03; pin đầu/cuốiPASS; khôngAC mất.
Không bỏ ảnh/outlier theo thời gian. FAIL/incomplete giữ nguyên, khôngretry.

A/A geometric ratio theo2image median(A)/median(B); MDE80 dùngt approximation
trênlogratios, df1, exp(MDE)−1. MDE với2image chỉ diagnostic, không power
validation. Không CI hay p-value population từ smoke. K/count/absolute medians
là descriptives; feasibilityPASS không thành timingtấn công, equivalence hoặcDoS.

Sau smoke báo raw/gates/ước lượng runtime; **dừng trước full** theo
sponge-research§1.4. Full/A-A20 hoặc attack replay cần protocol/tag riêng và
smoke review; không tự chuyển sang full chỉ vì smoke đẹp.
