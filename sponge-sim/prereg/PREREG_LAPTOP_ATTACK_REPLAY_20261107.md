# Real frozen PGD CPU segmentation timing — prospective07/11/2026

Written AFTER cleancontrolsAppAffinity20PASS (1/22flags,MDEforward.5965%,
total.5073%) and earlierfailures. Onechanged-conditioncleancontrol does not
prove mixed-attacktiming validity or cause. User authorizes continuedresearch.
At mostone2imagesmoke,iffPASSone20full. No effect-basedeligibility, no retry
untilsignificant orPASS. No newattackoptimization/objective/seedselection.

## Inputs/threatmodel and runtime

Same20COCO128-derivedM3 frozenpairs clean/attacked, sortedIDs and exactPTSHA in
lock. Attackedinputs were whole-imagePGD100steps/eps16/255 onYOLOv8n-seg, not
localized/universal/physicalpatch. These counts/GPUoutcomes werealreadyseen;
CPUreal-attacklatencies here unseen. Sameweights,CPUFP32unfused,batch1,
640square,nc80,conf.25,IoU.45,maxdet300,maxnms30000,fixedcompiledTVclassaware.
Nativefull-resolution mask path process_mask_native640² is selected mode,
not claimed as everydeployment/defaultAPIpath. NoYOLO26timing/transferhere.

ProcessCPU2,AboveNormal32768,HighQoS,1intraop/interop,OMP/MKL/BLAS1. Temporary
ownChatGPTtreeexclusionCPU2/3 andrestoretransaction exactly previousprotocol;
anyjournalready/restore/readbackissueinvalidatesstage. NoSystem/otherappkill,
globalpower/turbo/thermal change. SameAC/configured/effective mode/runtime.

Beforetiming check everyinputSHA,shape/range/finite andmaxLinf<=16/255+1e-6
(floatroundoffallowance, notperturbationrepair). No input reconstructed/changed.
Tensorloading/preprocessing excluded. Counts and predictionhashes onCPUrecorded;
not required to bebitidentical toCUDA acrossruntime versions.

## Stages, schedule and clocks

For eachimage armclean/attacked, preparepred/proto/detections once outside
stage timing. NMSstage clonesprediction OUTSIDE wallANDCPUintervals. Stages:
forward,nms_tv(wrapperfilter/sort+kernel),mask_native(fixedproto/dets),total
(net→NMS→nativeMask directly),aa_A,aa_B (samefulltotal duplicate labels).
Total not assembled from stage medians; stageinputs preparedtensors differ
from freshfullpipeline, reportscope. A/A botharmstake their actualmixedload.

12repeats, sixstagegroups withbotharms paired; twoLatinpositioncycles,
6/6clean-first/attack-first perstage,seed20261107+100*imageindex. Exactbalanced
orderalgorithm boundbysourcehash, unit-tested. Two warmcalls perjob/image.
Oneglobalwarmup=20clean+attackedfullpairs ofFIRSTinput, BEFORE baseline,
instead of20cleanforwards. This prospectivelyfixedmatchedloadconditioning is
different from oldA/A20; disclose, no poolingabsolute timings. No pilotgrid.

References start,beforeeachimage,end:15forward samples with3warmforwards of
firstcleaninput. Same3%absolute median/ref0criterion. Maskmatchedwarmup doesn't
automaticallyfixreference; keepanyflags. PinQA3sstart/endperoldbounds. GC-off
perf_counter_ns aroundfunction,CPU-timeoutsidewall,PDHaftertrial unchanged,
allrawbuffered/flushbetweenimages. Storememoryinfo afterimage. CPU-only sync
is synchronous; no CUDAcalls. No otherassistantwork/logpoll whiletimingactive.

Smoke2images,12reps,288timings,4refs/0flags,pin/readbacks/AC/complete required,
AA_MDE n2diagnostic; timeout1200s. IfFAILcancel full. IfPASSreport thenproceed
toonefull20understandingauthorization. Full2880timings,22refs<=2flags,
bothclean/attackedtotalA/A_MDE<=1%,exactIDs/reps/pin/readbacks/AC+restoration
valid. Fulltimeout5400s. Settle60s eachstage; no additionalruns or exclusions.
Runtimeestimate smoke~severalminutes/full~15–30min, unverifiedbefore smoke.

## Registered endpoints and inference

Primary: geometricmean perimage median(attackedtotal)/median(cleantotal),
percentileCI95% imagebootstrap20000,RNG20261107. Also pairedmean deltaMS and
samebootstrapCI.20images areunits; repeats notindependentimages. This estimates
thefixed20-imageconvenienceset underonesession, not1500population/devicevariance.

Secondary descriptivepointwise CIs for forward,NMSwrapper,masknative; N_tau/K
perarm; counts don't substitute for latency. SameCPU/wall/highwall/PDHdescriptors
computed SEPARATELY perarm so trueattackworkload isn't called anoiseoutlier.
Noformal pvaluefamily/Holm claim; pointwiseCIs labeledsecondary. No count of
failed quality endpoints discarded. Report allqualitydecisionsBEFOREeffects.

A/Anoise:MDE80=t approximation onimage ln(medianaa_A/medianaa_B), exp(MDE)−1,
botharms<=1%fullgate. CIs containing1 neverequivalence. AnygateFAIL=>STOP,
effectsDIAGNOSTIC only, no acceptedheadline. No effectsize/sign gate. IfPASS,
oneCPU-latencyresult only; no independentconfirmation, queue/deadline/drop,
camera/physical,energyexhaustionorDoSclaim. Availability needs a separatefixed
deadline/cleanfeasibility/realworkers experiment. No automaticnextmeasurement
or parameterselection by preferredattackratio.
