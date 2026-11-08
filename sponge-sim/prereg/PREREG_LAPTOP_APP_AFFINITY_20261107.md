# Temporary separation of assistant app from measured core —07/11/2026

Written AFTERAboveNormal smokeFAIL3/4flags; its full20cancelled. Priority and
HighQoS interventions are closed, not retried. New external condition only:
temporarily excludeCPU2AND its SMTsiblingCPU3 from own ChatGPT root and its
descendant app/tool processes. Worker is a freshchild that EXPLICITLY pinsCPU2
beforeTorch, overriding its inheritedhelper mask. Otherapps/kernel/interrupts
remain unconstrained; this is NOTexclusiveCPU isolation orproof of cause.

Motivation: offCPUstalls persisted despiteAboveNormal; a post-run snapshot
showed ChatGPT/otherbackground activity. Snapshot is not telemetry from past
stalls. Currenttool ancestor chain identifiesownChatGPT root. No process is
terminated, no priority changed outsideworker, no System/svchost/antivirus/
driver/registry/power/thermal change. User authorized autonomous performance
work; reversible own-app affinity only. If ownroot not found, no intervention.

## Transaction and restoration

Controller selectsownChatGPTancestor,currentuser; enumeratesonlythatroot/tree.
For eachPID+createTime: preserveoriginalmask, journalintent+flushBEFOREset,
newmask=oldminus{2,3}, nonemptyrequired, exactreadback. Up to3enumerationpasses
fornewhelpers; ifmissingpermission/readback/racefails, ABORTbeforebenchmark and
restorepartialchanges. No silentpartialseparation. WorkerCPU2pinmuststillPASS.

Parent blocks throughmeasurement. Finallyrestoreoriginalmasks for surviving
samePID/createTime only. If mask externallychanged, do notoverwrite;logskip.
Newhelperdescendants are restored to nearestrecordedparent's original mask
onlyif their currentmask equals expectedinheritedrestriction. ExitedPIDs
recorded,PIDreuse skipped. Log allrestoreerrors/readbacks,reportcoverage;
do NOTsayfullyrestoredifrecordsdisagree. Mutationjournal persistsforrecovery.
No journal/raw overwrites. Helpers maystill use sharedcache/memory/package;
unrelatedprocesses/OS can stillpreemptCPU2.

## Fixed campaign and existing conditions

One smoke2, ifPASSonefull20, no additionalcore/priority/repeatsgrid. Same
AboveNormal+HighQoS,16reps/label/stage,PDH/processCPUinstrumentation,allwarmups/
settle/timeouts/CPU2/inputs/weights/runtime as previouspriorityprotocol. THIS
externalappCPUallocationchange distinguishes newcampaign; nocausalclaim from
historicalcomparison, noattackeffectselection. Source/locks/newtagBEFOREdata.

Smoke completeness128timings/pin/QoS/priority/AC and0/4referenceflags required.
IfFAILfullcancelled. Full1280timings/20images/22refs<=2flags3%,bothforward/total
MDE<=1% unchanged. Restore app affinity after EACH campaign (including errors)
before any analysis. Full uses a newtransaction, same exclusion. Standinguser
authorization covers bothregisteredsteps ifgatepasses; reportsmokefirst.
The stage is eligible only if journal containsready and no restoreerror,
unexplainedmaskchange or mismatchedrestore; exitedhelpers are not restoreerrors.

RecordAAmedians/MDE,allrefs/trials/spikes,CPU/wallandPDHendpoints unchanged.
Posthocdiagnostics never qualifyattacktiming. PASS is atmostonechanged-condition
cleansizing session; noDoS or multi-sessionclaim. AnyFAILSTOPthiscampaign. No
continuing to anotherhelpermask/core/samplingbudget to getPASS. Ifstillnoisy,
reporttheenvironmentlimitation and require a distinct cleanenvironment/new
independentmechanism before further latencymeasurement. Keep all earlierFAILs.
