# Real segmentation attack replay on two physical laptop P cores

## Known outcomes before registration

Historical P2 smoke before carryover/output-lifetime corrections passed pin
but failed reference drift; preserve that FAIL. Later corrected P1 total-only
sessions passed, pre300 defense passed, out50 smoke failed, prepared-component
NMS full/MASK smoke failed and their protocols are CLOSED. Forward-only traces
showed >3% drift without NMS/MASK; neither cadence is selected for this study.
No new P2 model/pin-load/timing observation yet. No phone load or rented host.

Question: under the SAME corrected whole-pipeline protocol and frozen inputs,
what attacked/clean latency ratio is observed within a fixed TWO-physical-P-core
configuration? This supplies the missing real-input multi-core condition. It
does not retry a component control, prove a cross-core treatment effect or
replace the historical failed P2 smoke with a revised label.

## One fixed configuration, finite plan

Windows topology identifies P-core pairs [0,1] and [2,3]. Worker allowed CPUs
EXACTLY[0,2], two intra-op threads and one inter-op. SMT siblings1,3 are not
worker CPUs. Temporarily exclude0,1,2,3 from OWN ChatGPT process tree only,
retaining journaled before/readback/restore. Reject a changed topology or empty
remaining own-app mask before measurement. Other system/user processes are
not moved; CPU0 interrupts may remain. This is not isolation of physical cores.

Reuse unchanged run.py, attack_replay.py, output_lifetime.py, windows_qos.py,
PDH and hardware helpers, each source-SHA bound. New pcores.py only adapts the
owned-app transaction to both P cores and verifies topology. No power/turbo/
registry/package/network changes. Same locked AC/power GUIDs, AboveNormal and
HighQoS on own measurement worker, FP32 unfused YOLOv8n-seg640, confidence.25,
IoU.45,classaware fixedtorchvisionNMS/maxdet300/maxnms30000/native640masks.

Same frozen20 clean/whole-image-PGD pairs, model/input SHAs, epsilon16/255,
no optimization/input selection. CPU thread count may change floating-point
predictions and counts relative to P1: record every count/predictionSHA and
report differences, never require byte equality across core configurations or
discard a changed image. Smoke/full first-two prediction/count comparisons
are descriptive consistency observations, not an added post-hoc gate.

ONE smoke2 first, n2 only for feasibility:48repeats xAA_A/AA_B x2arms x2images
=384timings,4refs;1800s timeout. Gate complete/exact schedule+counts/pin/env/
validrestoration and ZERO referenceflags atabs(ratio-1)>.03. n2MDEdiagnostic,
not full precision evidence. IfFAILcancelFULL and STOP. No retry/E-core/core/
thread/warmup/repetition sweep. IfPASSreport,thenONEfull20 (3840timings,22refs,
7200s timeout). Standing user continuous authorization covers conditional full.

Full requires <=2/22referenceflags, both arm duplicate-label A/A MDEfraction
<=.01 UNROUNDED, complete+exactcoverage/counts/schedule/pin/env/restore. Formula
and unchanged analyze_attack.py estimator are bound before timing. All spikes
and images retained; missing/error outcomes invalidate or remain partial,
never imputed/replaced. After one full STOP regardless of result.

Same carryover-balanced48schedule seed20261107+100*imageindex, exact same
callable for A/B within arm, one first-arm burn, global20clean/attackedpipeline
warmup pairs on first image,2warmups/job/image,ref3warm+15forward samples,
start/beforeeachimage/end,pinload3sstart/end,settle60s. Output released AFTER
both clocks and BEFORE next invocation. Timing only tensor->native masks;
no JPEG/network/queue/outputcleanup/service deadline included.

## Estimation and instrumentation scope

Primary per-image median of96pooledtimings/arm, geometric mean attacked/clean
and percentile95%CI from20000imagebootstrap draws RNG20261107. Secondary mean
paired image-median difference(ms) and same bootstrapCI. One primary ratio
contrast; no inferential p-value/cadence/core comparison or multiple-test
family is claimed here. AA MDE uses same registered log-ratio formula. CI is
conditional on the fixed known20images/onePGDpair each, not new images/seeds.

ProcessCPU/wall measures effective cores, can exceed1 legitimately on P2;
do not call it the fraction of a two-core budget. PDH samples only logicalCPU0
via the unchanged worker, not both CPUs. Keep counter status/quantization and
interval limitations, never infer actualGHz/thermal or unique scheduler cause.
Historical P1 comparison is descriptive only: different session, app exclusion
and possibly numerical predictions. No pure parallel speedup or independence,
physical patch/universal attack/ARM/Pi/DoS/steady-tail claim.

## Registration and outputs

locks lock_pcores_smoke.json and lock_pcores_full.json bind sources, protocol,
current topology, versions, AC/power, model and all20inputSHAs before tag
prereg-laptop-pcores-20261107. Known hw.py workingCRLF vs GitLF exception is
explicit, both hashes recorded; every other source/lock matches tag bytes.
Results laptop_pcores_20261107/{smoke_01,full_01}/P2 plus sibling affinityjournal.
No old raw/FAIL overwrite. While timing ONLY benchmark+registered instruments,
no assistant file/git/analysis/test/web work or progresspoll. Wait completion;
restore audit FIRST, exclusive saved analysis+receipt after. Four-section
report/source maps/figures; include failed controls. Do not infer scientific
qualification from successful process exit or this diagnostic's smoke alone.
