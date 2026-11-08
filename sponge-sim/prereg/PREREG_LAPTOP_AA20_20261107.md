# Prospective clean A/A20 — P1 CPU laptop, awaiting full approval

READY FOR REVIEW, UNRUN. Written AFTER smoke results are known07/11.
User authorized the smoke under prereg-laptop-seg-smoke-20261107; it is closed,
P1 feasibilityPASS0/4flags, P2FAIL1/4flags. This next protocol is separate.
P1 selected by feasibility, no attacked effect has been observed here.
Old smoke source/lock preserved in tag9cafdd8; current generic metadata/analysis
changes apply only to new runs. Do not relabel or rewrite any old artifact.

## Question, fixed conditions and source

Can P1_cpu2=[2] maintain drift and A/A MDE gates over20 clean images? One
session, no retries; no P2 repeat, PGD, adv inference, injection or defenses.
All20 clean M3 tensors in original sorted image order, IDs and byte SHA in
lock_aa20.json, from the same previously verified ZIP. Weights identical to
smoke. Same CPU FP32 unfused model/runtime/fixed-TV class-aware semantics,
640-square/conf.25/IoU.45/nc80/maxdet300/maxnms30000/native640²masks.

P1 same pin-before-Torch/1intraop/1interop, env threading variables1; QA3s
with2threads at start/end, effectiveCPU upper1.15/lower.5 and allowedCPU>=70%.
Seed20261107, global20forwardwarmup, per-reference3warmforward+15samples,
per-image2warmforward+total,8repeats×A/B×forward/total. Same balanced stage and
label orders as smoke, process/time boundaries unchanged. Tensor load outside
timing; total directly measured. No preprocessing latency claim.

Same configuredACBestPerformance/effective-overlay961cc777/baseBalanced and
AC requirement. No additional power, priority, QoS or turbo change. New
post-smoke probe process returned power-throttlingControlMask0/StateMask0,
normalpriority; it is NOT telemetry from exited smoke workers, and does NOT
exclude heuristic OS power management. No frequency/thermal cause claimed.

Settle60s before one worker; wall timeout1800s. No actual CPU temperature
telemetry available; missing recorded explicitly, thermal protection unchanged.
Do not run other assistant work, tests, git, web, log reads or progress checks
during timing. Wait for campaign completion before analysis. AC/mode/affinity
changes, worker errors/timeouts stop and preserve partial data.

## Endpoints, acceptance and analysis

- Complete DONE,20images×8reps×2labels×2stages, exactrepIDs, pin start/endPASS.
-22references, flag if abs(median/reference0−1)>.03; <=2flags. This scales the
  previous22reference full protocol, not a relaxation of the4reference smoke.
- Both forward and total A/A MDE<=1%: d_i=ln(medianA_i/medianB_i),
  MDE=exp((t.975,19+t.8,19)·SD(d,ddof1)/sqrt20)−1. One image is one unit.
- Report exp(mean d), all image medians, MDE, every reference, max drift,
  rawSHA, input/source/config identity, AC/affinity and pin verification.
  All images in primary; no outlier or flagged-block exclusions to rescue gate.
  No significance/equivalence/DoS claim. This is sizing/control validation.
- Any gateFAIL: report FAIL and STOP this campaign. No extra sessions until
  pass, no changing the1% budget after results. SmokeMDE was48.6686%total with
  n2/df1; it does not establish power at20, and PASS is not guaranteed.

After full control report, a real-attack study requires its own prospectively
locked endpoints and tag. This control campaign alone cannot supply an attack
slowdown, even if it passes. Full starts only after smoke review per project
skill; the executable command and all sources/inputs are ready for approval.
