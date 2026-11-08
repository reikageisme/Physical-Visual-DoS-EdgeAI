# Prospective output-lifetime/carryover harness correction —07/11/2026

Alloldoutcomesseen.48-repeatconfirmationFAILED3/22refs,cleanAA_MDE1.9258%,
attacked.4350%; total4.0583x remains diagnostic. No additionalrepeatbudget,
core/priorityselection or oldresult relabeling. Newindependentharness mechanism
was identified inposthocsource/schedule audit, not anotheridenticalretry.

## Concrete defect/ambiguity found

Oldfour-callcycle balanced absolutepositions but NOT previouscondition within
AA labels. In cleanimageevenindices, AA_Afollowedattacked12times/AA_B24;
oddindices24/12. This deterministicallycouples AA label and previous large-mask
condition. Posthocimbalance-vs-logAAratio association.84288 is descriptive,
not causalproof. Conditional2000balancedlabelreassignments onoldraw gave
cleanMDEmedian.9339%vsobserved1.9258%; NOT a validrandomizedtest, no oldgate
changed or counterfactuallabels substituted forraw. Exactsources saved.

Oldtimed returnedTensorheldincaller`_` untilnextcall'sreturn. Thuspreviousmask
could remainlive DURINGnextfunction; maskK/dtype determinespreviousbufferstate.
This is an output-lifetime ambiguity relevant toA/Acarryover, not proof that
it caused everyoldstall. Installedrenderer matcheswheelRECORD; no packagefix.

## Corrections, verified before newdata

1. Time realCPU function tocompletion, stopwallandprocessCPUclocks, THEN
   release its output immediately before PDH/nextinvocation. The function's
   work is not moved outsideclock. A productionhelperreturnsNoneafterrelease;
   weakrefunittest verifies no previousoutput remainslive. Endpoint is
   tensor→mask-return computation underone-inflight-output lifetime; excludes
   outputdeallocation/serialization/queueing. Do not call this fullservicecost.
2. AA labelsuse EXACTSAME callableobject perarm, nottwoidentical lambdacode
   objects. Math/model/inputs/weights/NMS/renderer/outputbits unchanged.
3. Keep previous48-rep armorder andadjacentduplicates; randomizeA/Blabelorder
   WITHIN strata(arm,pairposition,previousarm). ChoosehalfAfirst perstratum
   usingfixedseed20261107+100*imageindex. No timingused inassignmentselection.
   Oneunmeasured first-armfullcall afterordinarywarmups initializesprevious
   workload soallstrata even.48reps yields12visits/label/absoluteposition and
   IDENTICAL preceding-armcounts forA/B; unit-tested forbothimageparities.

These are a bundle; any improvement cannot uniquely distinguish release vs
carryoverrandomization/callableidentity. AdjacentAA stillcannotcapture all
between-armbias. Disclosechangedmemory/schedulecondition; oldvalid/invalidruns
stay asregistered. Do not compareabsolute timings as unchangedprotocol.

## Fixedconditions, scope, gates

Samefrozen20whole-imagePGDpairs/byteSHA/weights,CPU2/AboveNormal/HighQoS/
1intraop+interop/ownapp2,3exclusion+restore/FP32unfused/native640mask/fixedTV
semantics/power/AC,48reps,primarypooled96trials/arm/image,AA48perlabel.
Global20mixedfullpairs,warm2perjob,burn1firstarm,refs3warm+15forward samples,
settle60s,pinQA3s,bothclocks/PDHmetadataunchanged exceptrelease outsideclocks.
No isolatedstageprofiles inthiscampaign. Allsource/config/inputSHA/newtagBEFORE.

ONEsmoke2images/384trials/4refs0flags/complete+pin+readbacks+AC+journalvalid;
timeout1800s. IfFAILSTOP/fullcancelled. IfPASSreportthenONEfull20/3840trials/
22refs<=2flags3%/botharmAA_MDE<=1%;timeout7200s. Appaffinityrestoreaftereach
stagebeforeanalysis,allerrors/skipsreported. Userstandingauthorizationcovers
fixedsequence. Nootherassistantwork/logpoll whiletiming. No retries/budgetgrid.

Primaryratio/delta/20kimagebootstrapRNG20261107 andMDEtformula unchanged from
totalconfirmation. Noeffectsign/sizegate. Keepallraw/spikes/images; no1%-gate
relaxation. CPU/PDHdiagnosticsseparatebyarm. No DoS,physical,universalpatch,
independentimage/device ormulti-sessionclaim. PASS=one newcorrected-harness
CPUtimingcondition only. IfFAIL, stoplatencycampaigns here; don't invent another
core/repeat/memorycondition merely to obtainPASS.
