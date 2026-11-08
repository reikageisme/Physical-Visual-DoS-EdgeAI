# Independent-measurement total-only precision confirmation —07/11/2026

Written AFTER mixed-stage20replay:2/22refsPASS,attackedAA_MDE.7341%PASS,
cleanAA_MDE1.8887002461466407%FAIL, overallFAIL. Observeddiagnostictotalratio
4.1104 and allstageeffects areknown. Keep oldFAIL and do not promote its CI.
This is ONEnewprospectiveconfirmation, not anotherattempt oftheoldprotocol.
User standingauthorization covers fixedsmoke/full sequence.

## Why a new design, and fixed precision budget

Prior primary/AA calls were interleaved with isolatedNMS/mask/forwardbenchmarks
using preparedtensors. Suchprofiling isn't the sameworkload as repeatedly
executingfullpipeline and may affect allocator/cache/AAprecision. Newdesign
removesisolatedstageprofiling entirely; only identicalfullpipelinefunctions
execute during measuredtrials. This is a hypothesis about harness workload,
not an establishedcause, and changesabsolute timingconditions.

Budget derivesONLYfromcleanAAnoise, notattackeffect, CI orp-value:
12*(log1p(.018887002461466407)/log1p(.01))²=42.43200801315981 repeats.
Roundup to48 (complete balancedblocks, compatiblewith4positions). Square-root
precision scaling is a planningheuristic; temporalcorrelation and changed
schedule mean PASS/power NOTguaranteed. Do not recalculatebudget afternewdata.
No extra48/72/96 or sessions untilPASS. If thisconfirmationFAIL, stop further
CPUattackmeasurement until a genuinelydifferentenvironment/mechanism is set.

## Conditions held and conditions changed

Sameall20frozenM3clean/whole-imagePGDadvpairs,exactSHA/weights,CPU2/
AboveNormal32768/HighQoS/1intraop+interop,ownappCPU2/3temporaryexclusionwith
fullrestoration,CPUFP32unfused,640square/conf.25/IoU.45/nc80/maxdet300/
maxnms30000/fixedTVclassaware/process_mask_native640². Sameinputbudgetguard,
CPUtime/PDHoutsidewallclocks,GCoff,nootherassistantworkduringmeasurement.
Do not optimizeinputs, selectimages/seeds or alterqualitygates.

Beforemeasuredblock, preparecounts/predictionhash once perarm; no prepared
mask/NMSsnapshot retained forstagebenchmarks. Eachrep hascleanA/cleanB and
attackedA/attackedB, samefunctionwithintwo labels. Duplicatecalls ofeacharm
areADJACENT; armfirstalternates inbalanced2repblocks, labelorderalternates;
48reps gives12observationsperlabelineachofthefourpositions. Orderalgorithm
unit-tested andsourcebound. Same2warmcalls perjob/image,global20clean+adv
fullpairsbeforebaseline,refs3warm+15forward samples,settle60s,QA3sstart/end.

Primaryimage latency is median of ALL96total calls perarm (pool bothAAcopies),
NOTmedianofmedians and NOTsummingstages. This estimatordiffers from old12
primarytotalcalls; comparisonsold/new aredescriptive, notprioritycausalproof.
AAnoise stillseparate48callmedians A/B perarm, imageunitn20. No isolatedstage
latency claims fromthisconfirmation; oldstageeffects remain diagnostic.
AdjacentA/Aprobes short-timescalenoise and cannotbound everycross-armcache/
allocator/loadbias. Balancedarmorderandreferences mitigatebutdonotproveabsence
of suchbias; do not equate lowerMDE acrosschanged designs withrootcauseproof.

## Sequence, gates and endpoints

One2imagesmoke:384timings,4refs0flags,pin/AC/QoS/priority/restore/complete
required; AA_MDE n2diagnosticonly. Timeout1800s. IfFAILSTOP/fullcancelled.
IfPASSreportthenONEfull20:3840timings,22refs<=2flags(absratio−1>.03),
bothclean/attackedA/A_MDE<=1%,allreadbacks/exactIDs/reps/budget/journalvalid.
Fulltimeout7200s. Restore app affinity aftereachstagebeforeanalysis; record
allerrors/skips. User's continuedauthorization covers alreadylockedfull.

Primarygeomean of perimagemedianattacked/clean total, percentileimageCI95%
20000RNG20261107, pairedmeanΔms andsamebootstrapCI. All20images retained,
nooutlier/flaggedblockremoval; repeatsnotindependentimages. SameMDE80tformula
on20image logmedianA/B. No effectsize/sign eligibility; negative/nullresults
kept. CPU/wall,PDH/highwall descriptors perarmtotal with pooledlabels,
status/missing/cost disclosed; this is not kernel/clockgroundtruth.

Alloldoutcomes disclosed inreport; fullqualificationfirsttheneffects. IfPASS,
oneindependent NEWmeasurement of SAME20knownimages underchangedprotocol,
not independentimage/device replication or three-sessionconfirmation. No
DoS/deadline/drop/queue/camera/physical/universalpatch claim. No increasing
cohort/repetitionsby significance or treatingtheoldfailedrun asvalid afterward.
