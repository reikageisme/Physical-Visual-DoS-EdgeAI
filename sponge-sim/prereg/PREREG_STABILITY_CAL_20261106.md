# Controls-only stability calibration, 06/11/2026

Prior M4/M4B reference traces have been seen. They failed 4/3 flags under a
3% drift criterion. User requested improving stability, with all old data kept.
No effect maximization or selection of favorable attack outcomes is permitted.

One calibration campaign, exactly three profiles, in seeded order20261106;
each profile uses a fresh subprocess: all exposed logical CPUs, CPU[2], or
CPU[2,3,4,5]. One torch intra-op and one inter-op thread in all cases. Read back
affinity before torch import. GPU clocks/power/turbo unchanged, no installations.
Fixed yolov8n-seg FP32 cuda, one CLEAN COCO128 image ID9 resized640; no attacked
input or attack count is loaded by the calibration runner.

200 initial forwards, five warmup forwards before each reference; references
contain31 independent wall-clock forward samples at start, blocks0..19, end.
GC disabled inside measurements, CUDA synchronize both boundaries. Reference
median deviating >3% from first flags a point; max2 flags allowed, unchanged
from prior studies. Each block contains31 randomized paired A/B calls on SAME
clean image. Log each wall ns and monotonic start timestamp; write after block.
15 separate CUDA-event forward times per block are diagnostic device-only
measurements, never relabelled as user-visible end-to-end latency.

MDE80 on20 block log median(A)/median(B), central-t approximation, report exp-1.
Eligible if complete, <=2 flags and MDE<=1%. Choose among eligible by lowest
flag count, then lowest MDE, then fixed tie priority four/all/one. Selection
uses controls ONLY, not attack ratios. Keep all profiles including failures.
No repeats of this campaign to obtain an eligible profile. If none eligible,
report that this host cannot support the intended timing conclusion.

Before timing tag prereg-stability-calibration-20261106 and bind code, weights,
image, config and commit hashes. After calibration write selection.json. Before
any further attack-pipeline timing, commit/tag an exact lock referencing selection
hash and selected affinity. Three full confirmation sessions then planned, all
reported; no running until a preferred effect size appears. Full exact details
must be locked before those timings. Historical real-attack inputs remain fixed.
