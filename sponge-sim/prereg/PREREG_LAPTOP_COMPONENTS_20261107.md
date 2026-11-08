# Prepared NMS-kernel and native-mask controls on laptop CPU

## Prior outcomes and scope

Known before registration: whole-image PGD20 CPU total sessions passed after
carryover/output-lifetime correction (ratios4.235134263867301 and4.222968253034303).
Earlier mixed-stage replay FAILED clean A/A precision, so its stage results
remain diagnostic. Total48 failed too. Pre300 defense full passed; out50
smoke failed/cancelled. A36 A0/A0B are metadata only, load gate closed and
sshd stopped; rented host deleted. No new component observations yet.

Question: under explicit prepared-input conditions, what are the clean/attack
cost distributions of the FIXED torchvision NMS facade/kernel and the native
mask renderer? This is a component control, not an independent attack/seed/
image replication or new pipeline/service/availability result.

## Two predetermined component blocks, fixed order

First nms_kernel, then mask_native. Each has its own one smoke2 and conditional
one full20 on the SAME byte-SHA frozen pairs/model as accepted laptop sessions.
Same CPU2/oneintra+oneinterop/FP32unfused/AboveNormal/HighQoS/AC+lockedpower,
temporary ownChatGPTtreeCPU2,3exclusion and journaledrestore. No OS settings,
package installation, network/phone/rentedhost work or attack/input changes.

Preparation runs real model forward and fixed Ultralytics wrapper with
nc80/conf.25/IoU.45/classaware/maxdet300/maxnms30000. Temporarily intercept
torchvision.ops.nms ONLY outside timing to copy exact boxes/scores/IoU and
returned fullindices. Restore original in finally BEFORE warmups/measurement.
Record predictionSHA/kernel-inputSHA/full-indicesSHA,n_tau,kernel_n,kernel_kept,
Kafteroutputcap and proto_shape; match prioracceptedCPU predictionSHA/counts.
Never equate equal candidate counts across images with equal overlap/classes.

NMS endpoint: one timed block of EXACTLY64 original torchvision.ops.nms calls
on the identical captured cached boxes/scores, releasing each index result
inside the block. Timer includes facade/Pythonloop/indexcleanup, excludes
confidence filtering/sort/class-offset preparation, model/mask work. Rawns is
the WHOLEblock; per-call estimate=blockns/64. Primary perimage median pools96
blocks (6144calls/arm/image), then divide64. This is median of block averages,
NOT median individual-call/frame latency. No empty-loop subtraction or claim
of pure C++ execution. No calls-per-block change after outcomes.

MASK endpoint: one process_mask_native call on cached real proto+retaineddet
per trial, output native640x640,uint8 according to lockedinstalledrenderer.
Record untimed outputshape/dtype, release probe before warmup. Timer includes
coefficient/prototype product, interpolation/chunks,crop,threshold; output
release AFTER both clocks as in correctedtotal. No forward/NMS work in timer.
No claim of a monolithic all-float Kx640x640 resident allocation.

All prepared payload hashes checked before/after untimed probes and after
measured image block; mutation/boundary mismatch is an error, not excludeddata.
Stageconditions are DIFFERENT cache/lifetime scopes; do NOT add their medians
or subtract from earlier pipeline total to assign percentage responsibility.
Do not compare absolute timings as an unchanged protocol. Results can support
or contradict a conditional component-cost account, not unique causation of
historical GPU/CPU stalls or denial of service.

## Schedule, controls and finite stopping rules

Both blocks use same48repeats/adjacentAA_A,AA_B labels perarm (exact samecallable),
original carryover-balanced48schedule20261107+100*imageindex,onefirst-armburn.
Global20matchedstageclean+attackpairs on first image (64calls per NMS arm/block,
1call per MASK arm), then release warmup feature buffers. Two warmups perjob
perimage, refs3warm+15fixedcleanimageforward samples,start/beforeeachimage/end,
pinload3sstart/end,settle60s. No isolatedprofile other than the block's one
endpoint. ProcessCPU/PDHoutside wall timer. For short NMS blocks, Windows
processCPU is coarsely quantized: zero or >1 CPU/wall is retained, NOT a causal
off-CPU/clock inference. CPU/PDHdiagnostics use RAWblock units, not per-callns.

Perblock ONEsmoke2=384trials/4refs/timeout1800s. Gate complete+immutableinput
checks+pin/environment+validrestorejournal AND0refsflagged atabs(ratio-1)>.03.
n2MDEdiagnostic only. IfFAILcancelthatfull/no retry. IfPASSreport,thenONEfull20
=3840trials/22refs/timeout7200s. Fullgate:complete/exactcoverage,immutableinputs,
pin/readbacks/AC/restorevalid,<=2flags3%,BOTHarmAA_MDEfractions<=.01(unrounded).
MDEformula unchanged logratio(t.975+t.8)*SD/sqrt20 thenexpm1.

Proceed to the other DISTINCT prespecified component even if previousfails,
not as a rescue. No64/128/256grid,repetitionincrease/core/sensorselection or
same-blockrerun to obtainPASS. After both finiteblocks STOP this protocol.
Userstandingauthorization coversreportedsmoke→conditionalfull, no extraapproval
loop. While timing ONLYbenchmark and registered instrumentation; no assistant
files/git/web/analysis/progresspoll. Waitcompletion,restoreauditBEFOREanalysis.

## Estimators and disclosure

Perimage medianpooled96trial/arm, normalizedcalls asabove; geometricmeanattack/
cleanratio and pairedmeanmedianΔms, imagebootstrap20000 RNG20261107. All20known
images/spikes retained, no imputation/exclusion. Both A/Amedian48label/arm
andMDE reported. Rawtimings withrep/position/timestamps preserved. Report
perimage tails/quantiles descriptively, distinguish within-tensor variability
from different images. GatePASS is median-precision evidence, not stabletails.

Two prespecified primary ratio contrasts form one family; report pointwise95%
CI (not simultaneous). Eligiblefulls: exact two-sided signflip all2^20image
logratio signassignments, Holm acrossTWO. Failed/missingfull testingp1 solely
familyplaceholder, not no-effectevidence. Signexchangeability undernull needed;
fixed20images/singlePGDpairperimage/oneCPUmodel do not yield population/device
or architecture independence. No equivalence claim from CI containing1.

Newtagprereg-laptop-components-20261107/source/config/inputlock before any new
model/timing. Resultroot laptop_components_20261107/{nms_kernel,mask_native}/
{smoke_01,full_01}. Source/tagrawhashreceipts,figures/source maps andfour-section
report after bothblocks. Legacysharedhw.py workingCRLF/tagLF exception remains
explicit; exact executedworkingSHA bound in taglock. All other lockedsource
bytes must matchtag. Preserve ALLoldraw/FAILs and userdirtypaths.
