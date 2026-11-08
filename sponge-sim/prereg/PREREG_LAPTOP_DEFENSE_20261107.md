# Paired CPU defense timing on identical frozen attacked inputs

## Known results and hypotheses

All earlier failure history is disclosed in LAPTOP_CARRYOVER_20261107.md.
Corrected carryover first session and registered repeated session both passed
their fixed gates; ratios4.235134263867301 and4.222968253034303 on SAME20pairs.
Earlier stage profiles failed overall precision and stay diagnostic. Prior
COCO5000 clean accuracy and cap count replays exist; they do NOT measure
defense latency. Known counts suggest each cap changes a different boundary.
No new defense timing has been observed at registration.

Primary question: on the SAME attacked tensor, how does (a) max_nms300 or
(b) max_det50 change measured tensor-to-native-mask return time relative to
baseline max_nms30000/max_det300? Two fixed interventions, not a tuning grid.
Baseline is freshly measured in EACH paired comparison, not borrowed from
earlier sessions. Reduction is a hypothesis, not a quality eligibility rule.

## Locked design and finite sequence

Use the same20 known whole-image PGD100/eps16/255 tensors, model SHA and CPU
environment as the accepted sessions. No optimizer/input/pixel changes.
YOLOv8n-segFP32unfused,640native mask,conf.25/IoU.45/classaware/fixed compiled
TVkernel. CPU2/oneintra/interthread/AboveNormal/HighQoS/AC/lockedpowermodes,
temporary ownappCPU2,3exclusion with journaled restoration. No global system
settings, thermal changes, installs, network or phone/rentedhost access.

Two comparison blocks in FIXED order: pre300, then out50. Each has same
attacked tensor in BOTH arms named baseline/defended. A/B labels per arm use
the same callable. 48 repeats perlabel/arm/image,96 calls pooled perarmmedian,
same context-balanced schedule20261107+100*imageindex,onefirst-armburn/image.
Global20baseline+defended attacked fullwarmup pairs,2warm/jobs,60settle,pin3s;
refs3warm+15samples of fixed CLEAN image0forward before/between/afterimages.
ProcessCPU/PDHmetadata unchanged; no isolated-stage profiling in this campaign.
Release each output AFTER wall+CPUclocks, BEFOREnextcall. Endpoint excludes
outputcleanup,preprocessing/transport/serialization/queueing.

Per block EXACTLY one smoke2/384trials/4refs/timeout1800s. Eligibility complete,
0flags at>3%,bothpinchecks,environment/readbacks,journalapply/restorevalid.
If FAIL: no full for that block; retainfailure. If PASS: report and exactly
one full20/3840trials/22refs/timeout7200s. Full gate complete,<=2flags at>3%,
botharmAA_MDE<=.01 using unrounded log tformula,validpin/environment/restore.
No reruns or changed repeats/core/caps/gates. Proceed to the other distinct
predeclared comparison even if the first fails; do not substitute its result
as a rescue. After both finite blocks STOP this protocol. User's explicit
standing authorization covers smoke and conditional full, no reapproval loop.
No assistant tasks/filepolling while either measurement process is active.

New run.py optional defense branch supplies caps to the SAME Ultralytics
wrapper and renderer. Existing default paths keep their math. Warmup uses
the two actual attacked conditions, distinct from clean/attack load in old
sessions, so do not compare absolute timings across protocols as causal.
Actual torchvision kernel input count is observed by a temporary wrapper
ONLY in untimed count probes; restored in finally BEFOREwarmup/timing. Calls
inside measured full pipelines use the original compiled function. Record
prediction/detection SHA, n_tau,kernel_n,K and exact capoptions. Probe guards
kernel_n=min(n_tau,max_nms),K<=max_det; no posthoc output-dependent selection.
Input probes must match saved accepted-session predictionSHA/count baseline.

## Analysis, multiplicity and preservation

analyze_attack.py accepts explicit arm names; primary ratio defended/baseline
perimage median then geometric mean, pairedmean delta ms, imagebootstrap20k
RNG20261107, same A/Aandreference formula/coverage as old source. Smoke n2MDE
diagnostic only. All images/spikes retained; no timing exclusions. Count caps
are mechanistic secondary endpoints, not evidence of deadline/DoS resistance.

Two prespecified primary contrasts form one family. Report separate pointwise
95%bootstrapCIs, NOT simultaneousCIs. For eligible full comparisons compute
two-sided exactsignflip on mean perimage log ratio (all2^20signassignments),
Holm adjustment across the TWO comparisons. Set testingp=1 for missing or
qualityfailed comparison solely to preserve family size; never interpret it
as evidence of no effect and never publish its latency as accepted. Do not
pool sessions. Effect direction/significance does not determine eligibility.
Report all failures, MDEs and adjustedp with image-unit scope. Ratios near1
cannot establish equivalence; include MDE. No extrapolation to cleanaccuracy
under these confidence/resize conditions or to adaptive attackers trained
against the defense. Existing cleanAP uses a different confidence/letterbox
protocol; these are separate observations, not a matched tradeoff curve.

Prospective locks bind new protocol, runtime/analysis sources, inputs/model
and packageSHA, commit/tag before any new timings. Legacy hw.py CRLFworking
vsLFtag exception remains explicit, exact executedSHA bound in lock.
Root results/laptop_defense_20261107/{pre300,out50}/{smoke_01,full_01}.
Exclusive raw/summaries/journals/receipts; preserve all old outputs/runs.
Four-section report, source map and paper update after both blocks. No Pi/ARM,
physical/universal/localizedpatch, multi-model or serviceDoS claim.
