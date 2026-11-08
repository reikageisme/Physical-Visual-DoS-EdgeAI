# Forward-reference trace: two finite cadences, no attack timing

## Prior observations and purpose

All previous results are known: two accepted CPU total sessions, pre300 defense
PASS, out50 smoke FAIL, and the CLOSED component protocol (NMS full FAIL20/22
reference flags, 19 faster and one slower; MASK smoke FAIL1/4, full cancelled).
The component PDH stage-name aggregation omission has been identified offline;
raw counters are preserved, no old gate is changed. A36 load gate remains
closed, its SSH stopped; rented host deleted. No new trace measured yet.

Question: what forward-reference wall/CPU/PDH trajectories occur WITHOUT any
NMS or mask work, under two fixed between-block cadences? This is a diagnostic
control for reference behavior, not a repeat of either component experiment,
not selection of a warmup/cadence that makes an attack experiment pass.

## Fixed finite design

Exactly TWO fresh processes, one each: continuous (no intentional between-block
pause), pause2 (two seconds between blocks). Shuffle the initial list
[continuous,pause2] with random.Random(20261107) ONCE before registration; the
result is bound in campaign metadata. Each condition is run ONCE, no full
follow-on, no replacement/retry, no warmup/repeat/core grid. Then STOP.

Same laptop CPU2 (sibling3 excluded from own application), one intra-op and
one inter-op, FP32 unfused YOLOv8n-seg, clean tensor from the FIRST frozen known
pair 000000000009_inputs.pt; original model and input SHAs retained. No attack,
postprocessing, NMS or mask code is called inside the workload. Output lifetime
uses the previously corrected timer: release after BOTH clocks, before the next
call. Import torchvision before Ultralytics as before, although no NMS is used.

Per condition: settle60s before fresh-worker launch, original pin-load3s gate
before/after, exactly12 blocks, each3 untimed forward warmups +15 timed forwards
(180timings/condition,360total). No global adaptive/stability warmup. Between
blocks wait0 or2s as assigned. Query PDH once after each block's warmup to set
its interval, then after EACH timed forward. Preserve initial/invalid counter
states. Buffer JSON rows and flush after each block outside timing. Wall timer
and process-CPU timer bracket only forward; PDH and output cleanup are outside
both. PDH remains an interval counter including observer work, not exact
instantaneous hardware frequency or a temperature/scheduler trace.

AboveNormal + HighQoS on OWN worker only, AC/power GUID readbacks as previous
locks, own-application temporary CPU2/3 separation and mandatory restoration
journal. No global OS/power/package/network changes. After safety/environment,
pin, source or runtime failure stop the sequence, preserve partial/missing
condition, no retry. Worker timeout600s per condition. No phone/remote access.

## Endpoints, analysis and finite interpretation

Registered diagnostic endpoints: all12 per-block medians and their ratios to
the FIRST block; all180 raw wall/processCPU/PDH samples per condition; observed
within-block sample order and between-block cadence. Descriptively flag
abs(ratio-1)>.03, but NO pass/fail promotion, no new precision gate and no
selection of an initial reference from the resulting trace. No alternative
baseline, exclusion of spikes or post-hoc warmup-truncation result.

One process per cadence, same single clean image/device: no independent
session replication, inferential cadence treatment effect, CI/p-value,
equivalence, causal clock/thermal identification, or A/A MDE claim. Report
all raw values/flags and artifact hashes. A/A/MDE remain requirements for
future attack studies; a stable diagnostic trace cannot satisfy those gates.
Compare descriptively with historical references only with scopes explicit.
No component ratio/DoS/service/tail/ARM claim and no retrospective rescue.

## Source registration and output

New reference_trace.py plus imported run/separate_app/windows_qos/output_lifetime/
pdh_cpu/hw dependencies, exact runtime versions, model/input/config/protocol
SHAs bound in lock_reference_trace.json. Tag prereg-forward-reference-20261107
BEFORE any real model/pin-load/timing. Known shared hw.py CRLF working vs LF
Git exception remains explicit; record both bytes and normalize only for that
one source's registration audit, never normalize raw. All other source bytes
must match the tag. Results in results/laptop_reference_trace_20261107/run_01,
restore journal sibling. No other work while workers run, wait for completion
instead of log polling. Restore audit FIRST, then exclusive new analysis,
source/coverage/schedule checks, plot and four-section report. Standing user
authorization covers this one finite diagnostic; no additional consent loop.
