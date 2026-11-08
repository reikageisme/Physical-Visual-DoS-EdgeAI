# M4B fixed-tv real pipeline confirmation, 06/11/2026

M4 full20 already seen and failed >2 drift flags (4); preserve it unchanged.
M2 CPU4 and GPU passed; CPU1 failed. User requested full authorized work.
This is a NEW prospective stability/construct follow-up, not a replacement or
silent selection of a favorable M4 run. Report both and their quality gates.

Reuse exact 20 frozen clean/attacked inputs from prospective M3 native full
(IDs as M4, hash every .pt before timing). No optimization or best-iterate
selection here. Manifest image identity checked against loaded clean image.
Fixed torchvision NMS only; omit slow TorchNMS workload from timing schedule.
YOLOv8n-seg FP32 cuda, resize640, nc80, conf.25,iou.45,max_det300,max_nms30000.
Compare actual masks from same detector, process_mask(upsample=False) and
process_mask_native; total forward+decode/NMS+actual mask measured directly.
Stages forward/NMS/mask separately; no sum of medians for total.

Primary: full-res total attacked/clean geometric paired ratio across images,
median15 repeats per cell, bootstrap20,000 seed20261106. Secondary proto total
and absolute per-stage deltas. A/A duplicate randomized same image total-full
calls, MDE central-t on log ratios, report exp(MDE)-1.
Warmup50 fixed forwards before reference, 5 per job before timing. Smoke first2
images, warmup2,repeats3; full20/15. Job randomization same M4 seed schedule.
GC off, synchronize before/after timer, cloning outside NMS timing.
Reference15 fixed-clean forward samples start/each image/end, >3% flag,
>2 flagged points => quality fail. Report all plus excluding flagged image
blocks. Do not keep repeating until pass. Preserve every result regardless.
No concurrent workloads during timing. Tag prereg-m4b-fixed-pipeline-20261106.
