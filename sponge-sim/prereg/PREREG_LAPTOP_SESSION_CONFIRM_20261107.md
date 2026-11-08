# One repeated-session assessment, registered after first carryover PASS

## Existing observations and purpose

All prior outcomes are known and retained in LAPTOP_CARRYOVER_20261107.md.
First carryover full20 PASSED: 1/22 reference flags, clean MDE fraction
.00996621759496496 (close to .01), attacked .0021343183695886384; ratio
4.235134263867301, CI [4.0249535047979865,4.441712278610986]. Earlier default
AA20, QoS/priority smokes, mixed attack and total48 campaigns FAILED as saved.
One successful session after harness development is insufficient to describe
repeated-session precision. This new sequence asks whether the SAME frozen
method meets its quality gates again; it does not tune the attack effect.

User explicitly authorized continuing experiments and comparisons without a
time limit. That standing authorization covers the conditional full below
after reporting smoke. No further approval round is required. Phone idle and
network/charging declarations are NOT inferred; no phone or rented host use.

## Fixed design, before any new timing

Production sources UNCHANGED from a57662d carryover campaign. One fresh process,
same deterministic image order and within-image schedule, same CPU2/one Torch
intra/inter thread/AboveNormal/HighQoS/temporary own-app CPU2,3 exclusion, AC
and locked power modes, same 20 PGD100 whole-image tensor pairs and SHA-bound
YOLOv8n-seg weights. FP32/unfused,640x640,native uint8 mask renderer, compiled
torchvision NMS, class-aware/conf .25/IoU .45/maxdet300/maxnms30000 unchanged.
Same release-after-both-clocks and context-balanced labels, randomization seed
20261107+100*imageindex. No new attack optimization, input selection, new seed,
larger repeats/core sweep, service load or hardware claims.

Same 48 repeats, adjacent AA labels, primary per-image median pooling 96 calls
per arm; A/A median48 per label/arm/image. Global20 mixed full warmup pairs,
2 warmups/job/image,1 first-arm burn,60s settle,pin3s,refs3warm+15samples.
Process CPU and native PDH metadata unchanged; no side monitoring process.
Recorded pin sampler can have two helper threads, while model intraop/interop
remain one. Output cleanup is outside the recorded endpoint, between calls.

EXACTLY one smoke2 (384 trials,4 refs,timeout1800s). Eligibility: complete,
0 flagged refs at >3%,pin checks,AC/power/priority/QoS readbacks,valid app
apply/restore journal. If FAIL: cancel full, retain smoke, STOP this sequence.
If PASS: report and run EXACTLY one full20 (3840 trials,22refs,timeout7200s).
Full quality: complete, <=2 flagged refs at >3%, both arm A/A MDE fractions
<=.01, pin/environment/app restore valid. Same t(.975)+t(.8) log-ratio MDE
formula; no rounding before comparison. n=2 smoke MDE is diagnostic only.
No assistant tasks or progress-file polling during either measurement.

Full effect eligibility has NO dependence on sign/magnitude/CI. Estimate
geometric mean of attack/clean per-image median ratios and paired mean delta
in ms, image bootstrap20k RNG20261107, same analyze_attack.py. Keep all trials
and spikes, all 20 images, no posthoc exclusions. Report full session even if
FAIL, marked diagnostic when its quality fails. Do NOT replace first PASS.

Repeated-session assessment uses this new session's gates independently.
If either smoke/full fails, do not report both sessions stable, do not rerun,
change gates/repeats or search another core to rescue it. If PASS, report two
separate estimates with quality metrics, retaining the first result's narrow
clean-MDE margin. No pooling or combined CI is planned: images, attacks and
schedule are shared, so observations are not independent image/device samples.
Comparison of session medians/ratios is descriptive, not an equivalence test.

## Scope, provenance and output

Tensor-to-native-mask-return latency under one-inflight-output lifetime,
not JPEG/network/serialization/cleanup cost. Frozen digital whole-image PGD;
not localized/universal/physical patch. No deadline/queue/DoS/Pi/ARM claim.
Old stage decompositions remain diagnostic; this total-only repeat cannot
attribute slowdown uniquely to MASK vs NMS. Protocol/locks/code/input SHA and
tag precede new data. Legacy shared hw.py working CRLF/tag LF exception is
already disclosed: exact executed working SHA bound in committed lock; all
other locked sources match tag bytes. Do not alter shared file or old tags.

Root results/laptop_session_confirm_20261107/{smoke_01,full_01}. Preserve all
raw/journals, exclusive summaries, offline receipts/plots, update four-section
report/handoff/progress. After the finite sequence, STOP further timing in this
protocol and assess the remaining research gaps offline. Any future defense
or image expansion requires its own prospective protocol and scientific aim.
