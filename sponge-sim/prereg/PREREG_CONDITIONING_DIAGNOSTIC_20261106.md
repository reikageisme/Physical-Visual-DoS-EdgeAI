# Matched-load reference diagnostic, 06/11/2026

M6's THREE registered sessions are complete and failed,14/14/10 flags. Their
ratios1.925/1.976/1.960 and A/A MDE<0.56% have already been seen. They remain
failures, not eligible for the primary timing table. No more M6 sessions run.
This is one new bounded mechanistic controls-only experiment testing whether
the forward-only reference baseline is sensitive to previous mask-heavy workload.
Idle telemetry read shows P8/SM180MHz/memory405MHz; this does NOT prove the cause
of earlier timing shifts. No hardware clocks, power or turbo are modified.

Fixed host affinity from the prior controls-only lock(all exposed0..17),threads1,
fixed CLEAN ID9,YOLOv8n-seg FP32 cuda. No attacked inputs or attack counts loaded.
Before EVERY reference including start/end, run10 cycles of clean forward plus
deterministic synthetic full-resolution mask K300 (gpu_full.py fixture seeds
6100/6101). This makes reference prehistory explicit and matched. Then5,50,or200
forward warmups, in separate fresh processes with seeded randomized setting order.
Only these three settings, once each,20 blocks per setting; no repeats to pass.
Other clean A/A/reference sampling and3%/max2 flags/MDE<=1% gates unchanged from
PREREG_STABILITY_CAL. Device-only CUDA events remain separate diagnostics.
Read NVML SM/memory clocks,pstate,temperature after load/after warmup/after reference;
unsupported fields remain null with error codes. Reads occur outside timer.

Select eligible setting by lowest flags,then lowest controls-only MDE,then
shortest warmup tie-break. Report all three,not just selected. If none eligible,
STOP timing experiments on this machine; report environmental stability limit.
No further calibration/full campaign without a concrete external environment change.

If a setting is eligible, commit selection and a NEW exact full protocol/tag
before a NEW three-session matched-conditioning confirmation. That new protocol
must report all M4/M4B/M6 failures and must not claim to repair their data.
All3 full sessions must meet unchanged gates to call that confirmation stable;
otherwise stop, no repeat. Conditioning is outside timing and changes operating
context, so conclusions are explicitly for a warmed steady-load testbed,not
unconditioned interactive-service latency. Tag prereg-conditioning-diagnostic-20261106.
