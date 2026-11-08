# Prospective M2/M3 confirmation, 06/11/2026

Historical results already seen; this is a new confirmation on a fixed subset,
not retroactive preregistration. User explicitly authorized full runs after M4 smoke.
Scripts committed and tagged before even confirmation smoke. No optional stopping.

## Population and fixed configurations
First 20 lexicographic coco128-seg train2017 JPGs (same IDs as M4). Smoke first 2;
full all 20. These are NOT COCO validation or independent held-out data.
Bind hashes of files, weights, scripts and historical helper source in manifest.
torch 2.11/cu128, Ultralytics installed 8.4.203, image resize 640 square FP32,
conf .25, IoU .45, nc80, max_det300; fixed torchvision NMS imported explicitly.

## M2 synthetic workload and device axis
Run separate worker processes: CPU threads1 affinity[2]; CPU threads4 affinity
[2,3,4,5]; GPU threads1 (host affinity recorded). OS exposes 18 single-logical
cores, all efficiency_class0, despite i7-12700K name. Treat as exposed vCPUs;
do NOT label physical P/E cores or claim host topology. This replaces neither
the laptop model nor its cohort, but can provide a clean CPU confirmation.
Verify affinity readback and load via copied hw.verify_pin at start/end,
lower bound CPU1 .5 effective cores, CPU4 2.0; otherwise quality fail.

K=0,10,50,100,300; retina false/true; baseline10, deterministic fixtures reused
unchanged from gpu_full.py:make_masks (seeds6100/6101) with source hash.
Actual image forward and actual decoded NMS; only mask stage uses synthetic K.
Total is measured directly forward→NMS→mask, not sum of stage medians.
For each image shuffle (K,retina,copy) independently per repetition seed
20261106+100*image_index+rep; K0 has two A/A copies. Full 5 repeats; smoke2;
2 warmups per cell, 50 global forward warmups before reference.
Record total/mask raw ns, timestamp and ordering identifiers; GC off;
GPU synchronize both sides; CPU uses same timer without CUDA calls.
Reference=15 forward samples of same fixed first image at start/each image/end.
Flag abs relative median drift >3%; >2 points flagged -> quality fail.
Report all and sensitivity excluding flagged image blocks; do not discard failed runs.

Primary K* per retina = smallest K>10 with lower image bootstrap ratio CI > exp(MDE).
Median repeats per image then geometric ratio against K10. Bootstrap20,000 seed
20261106. MDE80 central t on log median(copy0 K0)/median(copy1 K0), across images;
report exp(MDE)-1. Budget secondary 1+median(maskK-mask10)/median(total10).
Report numeric error rather than relabel pass after seeing results.
Proto mode upsample=False is historical microbenchmark, not default predictor claim.

## M3 attack confirmation and genuine transfer
Source yolov8n-seg, reuse unchanged historical gpu_mechanism.py:attack, zero
initialization, steps100, eps16/255, alpha2/255, temp.02, same conf .25.
Evaluate FINAL iterate (not best-over-trajectory; explicitly differs from old M3
draft). Report both negative and positive final counts, no best-image selection.
Endpoint: paired K_final vs K_clean; fraction >=50, >=100, ==300; median gain.
No significance inference on availability based on counts alone.
Target yolo26n-seg receives SAME perturbation optimized on v8, no target optimization.
Assert target head end2end and native output [1,300,38]; count score column4>.25
DIRECTLY. No non_max_suppression or IoU suppression may be applied to target.
Unexpected shape => stop and preserve incomplete run; do not fabricate K.
Save every native target output and clean/attacked input for replay.
Bootstrap20,000 image unit seed20261106; report counts, medians, CIs and denominators.
This run contains no latency measurement; timing follows M4 fixed backend.

## Sequence
Smoke then full sequential, as authorized by user 06/11; no other work while timing.
Run M2 CPU1, CPU4, GPU as separate runs; run M3 native-target smoke/full.
Use new run directories always. Tag prereg-m2-m3-confirmation-20261106.

## Implementation amendment A (before any M3 count results)
Initial smoke stopped at end2end assertion BEFORE creating run directory or
forwarding images. Installed checkpoint has one2one_cv2 but _end2end unset,
so direct model default is one-to-many. Explicitly set target head.end2end=True
and assert it. This is the intended NMS-free intervention, not a change to
threshold/endpoint/subset/attack. Record target_end2end=True in manifest.
Tag prereg-m3-native-amendment-a-20261106 before rerun. No count data observed.
