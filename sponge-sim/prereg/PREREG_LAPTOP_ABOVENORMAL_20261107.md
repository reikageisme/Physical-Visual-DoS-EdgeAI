# Process priority diagnostic and fixed A/A20 —07/11/2026

Written after PDH diagnostic data: threeforward highwall436.8/451.0/583.6ms
had processCPU218.75ms, CPU/wall.501/.485/.375. A reference1531.38ms had
CPU218.75ms. Reported per-coreperformance~246–248% was comparatively narrow;
this supports investigating off-CPU stalls, not proving a named preemptor.
HighQoS smoke hadalreadyFAIL2/4flags,fullcancelled. PDH diagnostic0/4flags
is NOT a qualification or reason to rename old failures. All data known.

New single intervention: ABOVE_NORMAL_PRIORITY_CLASS32768 for OWN worker,
readback beforeTorch, eachimage,end; processHighQoS retained. No HIGH_PRIORITY
orREALTIME, no external-process priority/termination, no global power/thermal/
registry change. On exit policy ends. PinonePcore[2],leaving othercores toOS.
This is user-authorized ordinaryperformance work. API-only own-probe QA before
timing verifies priority/QoS; not a latency observation. No claim guaranteed
responsiveness or that priority uniquely causes any improvement.

## Fixed campaign, not iterative priority tuning

ONEsmoke2; iff samecomplete/pin/QoS/AC/0referenceflags gatePASS, ONEfull20.
Source/config/input hashes locked undernewtag before data. User's continued
authorization covers this sequence; report smoke before proceeding. If either
FAIL, STOP this priority campaign, nootherpriority/core/repeatgrid or more
images to rescue PASS. Failure then requires a distinct external environment
change or new independent mechanism before any further latency campaign.

Same workload as PDHprobe:16repeats/label/stage, two clean smokeIDs0009/0025,
fullall20cleanM3sorted. No adv,PGD,injection/defenses. torch2.13CPU/TV0.28CPU/
Ultralytics8.4.90,FP32unfused/batch1/640square/conf.25/IoU.45/nc80/fixedTV
classaware/maxdet300/maxnms30000/native640²mask.1intraop/interop;OMP/MKL/BLAS1.
Global20forwardwarmup,3warm+15samples/reference,2warmforward+total/image,
QA3sstart/end,settle60s,timeout600smoke/1800full. No work outside measurement
while active; no progressreads. Same powerconfig/AC requirement.

CPU-time, PDHaftertrial and source metadata exactly as prereg-PDH. PDHvalues
above100 retained,statusinvalid=>null; intervals imperfectlyalign and observer
overhead disclosed.16reps is differentfrom oldAA20's8,so don'tattribute any
old-new improvement uniquely topriority. CurrentPDHprobe16 is onlydescriptive
historicalcomparison; no randomizedprioritycausalclaim.

## Prespecified gates and analysis

Smoke:2images×16reps×2labels×2stages=128timings,4refs,0flags,
pin/API/QoS/priority/AC/complete pass. MDE n2 diagnostic only.
Full20:1280timings,22refs<=2flags absratio−1>.03,pin/readbacks/AC/complete,
BOTH forward/totalMDE<=1%. Oneimage is oneunit; nooutlier/flaggedblockremoval.
SameMDE t approximation and image-logmedianA/B formulation as AA20. Report
allqualitydecisions, samplecoverage, rawSHA, CPU/wall/highwall endpoints and
PDHcoverage/status/ranges/correlations exactly as previously registered.

Even fullPASS would be one changed-condition clean sizing session, not DoS,
causalproof or multi-session replication. Realattack/confirmation must have
newfixedprotocol/tag before its own outcomes. No automatic attack start here.
Do not change1%budget/referencegate after seeingnewresults.

Official priority semantics checked before run:
https://learn.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setpriorityclass
