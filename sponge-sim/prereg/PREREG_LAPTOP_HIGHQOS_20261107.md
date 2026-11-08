# Prospective HighQoS + process CPU-time diagnostic —07/11/2026

Written after all smoke/AA20 failures were seen. User authorized continued
code/experiments and ordinary performance commands; latest instruction allows
continued work. This is finite mechanism investigation, NOT repeat-until-pass.
All earlier raw and FAIL decisions remain. One new process-local intervention,
one technical smoke, at most one full20. No grid search or additional sessions.

## Rationale and limits

Old A/A20 had3/22flags, MDEforward2.36%/total3.09%, isolated wall-time peaks
near1.7s. Its worker QoS was not recorded; a separate probe read0/0. Old data
cannot identify scheduling, clock, runtime, background tasks or thermal cause.
Microsoft documents explicit HighQoS via SetProcessInformation, and that
automatic QoS may otherwise be heuristic. User configuredBestPerformance alone
does not prove the benchmark worker's effective QoS.

https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation
https://learn.microsoft.com/en-us/windows/win32/procthread/quality-of-service
https://docs.python.org/3.14/library/time.html#time.process_time_ns

No randomized concurrent default-vs-HighQoS comparison is performed. Comparisons
to oldAA20 are descriptive and confounded by time and added instrumentation.
Even improvement does not identify QoS as the sole cause of old failures.

## Intervention and locked runtime

Fresh worker, CPU affinity[2] BEFORE Torch pools. Apply documented process-local
HighQoS: PROCESS_POWER_THROTTLING_STATE Version1/ControlMask1/StateMask0,
SetProcessInformation class4, only GetCurrentProcess handle. Read before/after,
before every image and at end; execution-speed bit must be controlled and OFF.
Priority class must remain unchanged by set_high; no realtime/high priority,
global power-plan, turbo, registry, system package or thermal-limit changes.
Process ends normally; policy applies only for its lifetime. API errors abort.

Same i5-1335U,P1[2],1intraop/1interop,OMP/MKL/OPENBLAS1; torch2.13CPU,
TV0.28CPU,Ultralytics8.4.90; samecleanM3inputs/sameweights,CPUFP32unfused,
batch1/640square/conf.25/IoU.45/nc80/maxdet300/maxnms30000/fixedTVclassaware,
process_mask_native640². No adv,PGD,injection,defense or attack outcome selection.

## New instrumentation

Wall-time boundaries remain GC-off/perf_counter_ns around function. Additional
process_time_ns calls bracket that wall interval, OUTSIDE its wall boundaries;
record process_cpu_ns for EACH trial and cpu_samples_ns for references. CPU
interval includes tiny counter overhead, is not exactly aligned with wall;
reported nanoseconds do not guarantee true granularity. No clipping fractions
above1 or below0. Store clock implementation/resolution in manifest.
No output tensors retained across reference samples; telemetry buffered.

CPU-time excludes periods when this process does not execute, but captures
user+system execution; it is not a frequency counter or scheduler trace.
LowCPU/wall on highwall trials is compatible with descheduling/waits, not
proof of a named background task. If wall andCPU both grow, this still cannot
uniquely distinguish frequency from more computational work/cache effects.

## Fixed sequence and gates

Smoke: two firstcleanimages,8repeats×A/B×forward/total,4refs×15samples;
global20forwardwarmups,3warmforwards before eachref,2warmforward+total/image.
Settle60s,QA3sstart/end,timeout600s. One fresh worker with highQoS readback.
Gate: complete,API/readbacks/AC/pin pass,0referenceflags(absratio−1>.03).
MDE n2 diagnostic only. FAIL=>stop, nofull or smoke retry.

If smokePASS, user standing authorization covers proceeding to ONE full20
under its separately locked config; report smoke before starting full. This
explicit continuation replaces another permission round, not a scientific gate.
Full: all20cleanM3 sortedIDs, same8repeats,22refs/15samples,QA/warmup unchanged,
settle60s,timeout1800s. Complete exactIDs/pinPASS/readbacks/AC,<=2referenceflags,
both forward/totalMDE<=1% are required. Any gateFAIL=>STOPcampaign, preserveall.
No extra20/100/1500images or reruns to obtain a preferred gate.

Bothlocks include source/config/input/installedNMS+ops hashes and protocol;
tagBEFORE smoke/full. Fresh result folders underresults/laptop_highqos_20261107.
No concurrent assistant work during timing; parent blocks, no progress logreads.

## Analysis and endpoints

Registered timing sizing: image mediansA/B, d_i=ln(medA/medB), exp(mean d),
MDE80=exp((t.975,n−1+t.8,n−1)SD(d,ddof1)/sqrt(n))−1. Images are units, not
repeats. Keep all trials,images,flags,spikes; no rescue exclusions.

Additional descriptive mechanism endpoints: perstage medianCPU/wall,
walltrialmedian/max; fraction of highwall trials (>1.5× sameimage/stage median)
and CPU/wall among them, including share below.75; everyreference pairedCPU/wall.
Thresholds were selected after viewing old failures but before new outcomes.
Report rawselector/formula and no causal/scheduling equivalence claim. Do not
select HighQoS by attack effect or claim DoS from successful clean controls.

If fullPASS it is ONE clean-control session of a changed protocol, not an
independent reproduction of old condition or validated attack timing. Any
real-attack timing/confirmation requires a new fixed protocol/tag and controls
for its own mixed mask workload. No attack measurement is authorized by this
scientific protocol itself; user authorization does not supply missing endpoints.
