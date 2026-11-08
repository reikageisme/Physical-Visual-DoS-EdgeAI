# Prospective stability confirmation pipeline — draft awaiting calibration lock

Prior real-attack estimates and failed reference gates have been seen. The goal
is reproducible measurement, not an increased effect or selection of preferred
outcomes. Exact host affinity will be chosen only by the controls-only registered
calibration. Before any timing here, selection.json and its SHA must be committed
and bound by a NEW full-confirmation tag; until then this protocol is a draft.

Fixed20 images and final-iterate inputs from m3_native_full_20261106, unchanged
SHA identities. Model yolov8n-seg FP32 cuda, fixed torchvision NMS, conf.25,
IoU.45,nc80,max_det300,max_nms30000. No new attack optimization or seed selection.
CPU host affinity from locked calibration, one torch intra/inter-op thread.
No power/turbo/clock changes, no software installs or concurrent workloads.

Protocol changes from M4B (explicit before new observations): 200 warm forwards;
reference31 samples preceded by five warm forwards, same3% relative criterion
and max2 flags; buffer raw records in memory and flush after each image rather
than between measurements; reuse existing snapshots instead of saving duplicates.
These can change absolute timings, so report as a NEW protocol, not a hidden
repair or direct pooled reanalysis of M4B.

Eight stages per arm: forward, NMS, proto mask, native full mask, proto total,
native full total, same-input native-total A and B. Each full image has16 repeats,
two balanced cyclic Latin cycles for stage position. Clean/attacked within each
stage are paired and alternate order eight/eight times. Random seed20261106+
100000*session+100*image_index. Same timing boundaries: CUDA sync before/after,
GC off, perf_counter_ns; NMS input clone outside measured interval; total measured
directly, never assembled by adding stage medians. Proto mode upsample=False is
a microbenchmark setting, not claimed to be the deployment default.

One smoke on first2 images (2 repeats) for feasibility only; then exactly THREE
full20 sessions, even if an earlier session fails quality. Runtime errors stop
affected campaign for diagnosis; preserve incomplete runs. Do not repeat this
three-session campaign until all pass or until an effect looks preferable.

Primary: full-resolution total attacked/clean geometric mean of per-image median
ratios in EACH session, bootstrap20,000 image units seed20261106, pointwise95% CIs.
Report A/A MDE80 using central-t approximation on per-image log median(A)/median(B),
exp(MDE)-1; report both clean and attacked noise floors. Each session gate:
complete counts/repetitions, <=2 reference flags, and clean/attacked A/A MDE<=1%.
Call the confirmation stable ONLY if all3 sessions pass. No effect-sign or
effect-magnitude eligibility rule. Primary aggregate uses per-image median log
ratio across ALL THREE sessions, then image bootstrap; never select the best
session, and do not mix these ratios with synthetic-mask device controls.

Secondary: per-stage absolute deltas, proto total, all references, and sensitivity
excluding flagged image blocks, alongside rather than replacing primary data.
CUDA-event calibration timings remain a separate device-only diagnostic;
host-visible full latency is not renamed device kernel time to improve a gate.

Pending: populate exact calibration selection SHA, affinity, source hashes and
full tag before starting this confirmation.

## Exact lock before full timing — 06/11/2026
- Calibration tag prereg-stability-calibration-20261106; selected profile all.
- Selection SHA256: 5750c9c16cb97edfe0d7cc2f852d3a2db036e453fca914587e9fe7b9199c7761
- Host affinity: logical CPUs [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]. No physical/P/E identity claim.
- Selection used clean controls only: all flags1/MDE0.3605%; one flags3; four flags3. All calibration raw retained.
- Full tag: prereg-stability-full-20261106. Smoke first2 then exactly three full20 sessions under this same lock, seeds20261106+100000*session+100*image_index.
- Full runner studies/stability_pipeline/run.py; no full latency observations seen at locking time.
