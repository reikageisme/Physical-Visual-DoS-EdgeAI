# M4 — explicit NMS dispatch, 06/11/2026

Prospective diagnostic informed by historical GPU observations (already seen).
This does NOT retroactively preregister those results or M2/M3.

## Hypothesis and primary endpoint
Ultralytics 8.4.203 chooses TorchNMS.nms when torchvision is absent from
sys.modules, and torchvision.ops.nms when present. This may explain historical
0.8–175 ms discrepancies. Primary: per-image median latency ratio pure/tv on
identical decoded attacked predictions, including identical Ultralytics
preprocessing and class offsets. Count and compare complete kept detections;
differences/ties are reported, never silently treated as equivalent.

## Fixed design
Windows RTX 5060 Ti; yolov8n-seg.pt; FP32, 640 square direct resize, conf .25,
IoU .45, nc 80, max_det 300, max_nms 30000. torch threads/inter-op = 1.
GPU host affinity is recorded, not claimed as CPU pinned-core measurement.
First 20 lexicographically sorted coco128-seg/train2017 JPGs, IDs:
9,25,30,34,36,42,49,61,64,71,72,73,74,77,78,81,86,89,92,94.
Manifest binds image hashes and weights hash before timings.
Existing gpu_mechanism.py:attack is reused unchanged (100 steps, zero init,
eps 16/255, alpha 2/255); source SHA recorded. This is mechanistic replication,
not independent confirmation of an attack discovered on this dataset.

Smoke: first 2 IDs, 2 warmups and 3 repetitions. Full: 5 warmups, 15 repetitions.
Within each repetition randomize all clean/attacked stage jobs, seed 20261106
+100*image_index+rep. GPU sync before/after perf_counter_ns; GC disabled.
Clone predictions outside every NMS timing. Explicitly substitute only the
kernel callable in the same Ultralytics wrapper; restore after every call.
Keep raw ns, start timestamp, image/arm/stage/rep, counts and prediction hashes.
Save clean/attacked inputs for replay. No network downloads or installs.

## Secondary endpoints / limitations
Fixed-tv direct full pipeline (forward+decode/NMS+actual mask), clean vs attacked,
for process_mask(upsample=False) and process_mask_native. Time forward, NMS and
mask separately, never sum separate medians to infer total. Proto-res here is
the historical microbenchmark setting, not a claim about default predictor
upsampling. Report ratios and absolute deltas; high relative mask cost alone
does not prove mask dominates total. No synthetic K replaces actual detections.

## Analysis and quality, fixed before timing
Unit=image, median repetitions then geometric mean paired ratios with percentile
image bootstrap (20,000, seed 20261106); raw distributions p50/p90/max too.
A/A: independent randomized total-full-mask A and B on SAME image/arm;
per-image log median(A)/median(B). MDE80 uses central-t approximation
(t_.975,n-1+t_.8,n-1)*SD(log ratios)/sqrt(n), report exp(MDE)-1.
Reference=9 clean GPU forwards, before start/before each image/at end.
Flag if median differs >3% from start; report ALL, plus sensitivity excluding
flagged images. >2 flagged reference points => quality fail; never replace
historical failed laptop runs with this GPU diagnostic. Smoke is only feasibility,
not paper evidence. Full run requires approval after smoke per sponge-research.
Incomplete/error/OOM => preserve run and mark incomplete. No optional stopping.

## Sequence
Commit protocol, runner and provenance source, tag prereg-m4-nms-dispatch-20261106
before smoke. Remote script hash must match tagged script; manifest records tag
commit and protocol hash. Stop for smoke approval before full run.
CPU multithread, COCO full validation and YOLO26 counting require separate
protocols; do not infer those endpoints from this run.
