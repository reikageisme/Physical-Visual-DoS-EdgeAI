# Read-only per-core performance diagnostic —07/11/2026

Written AFTER HighQoS smokeFAIL2/4flags; fullHighQoS20 was cancelled, not run.
Its registered CPU/wallmedians~.98/.96 do not support attributing all noise to
off-CPU waits. This is a separate finite instrumentation question, not another
attempt to pass the failed HighQoS gate. One run, two fixed clean images,
16repeats×2labels×forward/total; NO eligibility search or full follow-up here.
No attack/PGD/DoS endpoint. User standing authorization covers continued work.

## Hypotheses and scope

Describe relation of wall/processCPU durations with Windows' reported per-core
performance counters. Counter correlation does not prove clock/thermal cause.
No data exclusion, no p-value or populationCI from two images/repeats. Do not
rename this measurement an accepted timing-control campaign. A/A MDE and all
reference flags still reported; any failure remains failure. There is NO gate
whose PASS would authorize attack timing from this diagnostic.

Preflight hardware: ASUSVivobookX1404VA,i5-1335U,HypervisorPresentTrue. CIM
reportedProcessorFrequency1300,PercentPerformanceLimit100,LimitFlags0 in an
idle/newread snapshot; not under-load ground truth, no conclusion of no limits.
NativePDH open/add returned0 for all five counters before timing. No sensors,
services, registry, kernel driver or protection enabled/disabled.

## Locked workload and intervention

Keep HighQoS ownprocess, affinity[2],samepriority,1intraop/1interop,source/runtime/
weights/cleaninputs0009,0025 as priorHQ; source/inputSHA inlock. Seed20261107,
globalwarm20,referencewarm3/sample15,perimagewarm2,16reps balanced AAorder.
ProcessCPUtimeoutsidewallboundaries unchanged. Pin3sstart/end. AC and power
mode unchanged; ifAC/QoS/affinity/readback changes abort preserve. Settle60s,
worker timeout600s. Only this instrumentedworker, parent blocks, no assistant
work/logpolling during run. Warm-up/telemetry overhead is part of new protocol.

New nativePDH query for `Processor Information(0,2)`:
ProcessorFrequency, %ProcessorPerformance, %PerformanceLimit,
PerformanceLimitFlags, %ProcessorTime. UsePdhAddEnglishCounterW, initial collect
after warmup; collect and format once AFTER EACH wall/CPU trial, outside both
intervals. PDH_FMT_DOUBLE|PDH_FMT_NOCAP100; values above100 retained. Return
codes and CStatus recorded; failures/nonfinite=>null, not0 or silently dropped.
Rate counters may require two collects; initial invalid samples remain logged.
Query-close finally; no external monitoring process or concurrent sampler.

PDH interval starts at previous collect, so it spans previous telemetry/logging
and current workload, and first measured trial's interval also includes refs/
warmup. Save intervalstart/collecttimestamps/readduration; disclose imperfect
alignment and observereffect. Do not claim per-function exact clock or convert
nominalFrequency×Performance% into independently measuredMHz. Missing counters
do not become claims of constant frequency. Hypervisor/counter implementation
may affect interpretation; no virtualization setting is changed.

## Predeclared descriptive endpoints

- Completeness/QA/QoS/AC, allrefs/flags, AAmedians/MDE as before (MDE n2 diagnostic).
- Perstage wallmedian/max, CPU/wallmedian and highwall>1.5×ownimage/stagemedian,
  CPUfraction below.75 as previouslyregistered; retain quantization fractions>1.
- Perstage validcoverage and min/median/max of all five counters; exactstatus
  counts; PDHreadduration median/max. No mean imputation or clipping.
- Perstage Spearman descriptive correlation: wallNS vs %ProcessorPerformance,
  processCPU_NS vs %ProcessorPerformance, wallNS vs processCPU_NS, only paired
  finitevalidsamples. Report n, no significance/causality claims. Constant
  series=>undefined/null rather than zero. Two inputIDs and16repeats do not
  become64 independent images. Inspect each highwall trial withrawline/PDH.
- If reportedperformance changes inversely with durations and CPU/wall stays
  high, say compatible with changing reportedexecutionperformance, NOT proven
  clock/heatcause. If counters constant or invalid, keep unresolved. Even a
  validLimitFlags0 cannot rule out every hardware/firmware constraint.

STOP after this one diagnostic; no next timing selected from prettiest outcomes.
Further measurements need a concrete new mechanism or environment change,
newprereg/tag, not repeat-until-pass. All old experiments/outputs remain intact.

Primary documentation checked before run:
https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcountervalue
https://learn.microsoft.com/en-us/windows/win32/api/pdh/ns-pdh-pdh_fmt_countervalue
https://learn.microsoft.com/en-us/windows/win32/perfctrs/creating-a-query
