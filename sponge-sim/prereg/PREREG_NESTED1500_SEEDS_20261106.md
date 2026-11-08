# Nested1000/1500, five starts and target26 threat-path study

User explicitly authorized more images,seeds,and investigation of0/500 transfer;
rental window5–6hours. Prior500 and four-image diagnostic observations are known.
Diagnostic: native gradients finite/nonzero; direct target-many attack raised
nativeK105–300,while transfer nativeK1–4. This motivates a NEW cohort experiment,
not a retroactive preregistration of that diagnostic. No claim native immunity.

Lock1500 COCO-val images: original locked500 filenames,then500 new,then500 new.
New1000 sampled without replacement from remaining4500 with Random(20261107);
sort each500 block before concatenation. Prefix1000 and full1500 are NESTED,
not independent replications. Recompute every case under the new homogeneous
execution configuration; do not reuse older500 measurements. Analysis may also
show prefix500 descriptively. Lock64 audit IDs uniformly by Random(20261108).

Five starts[0,12345,23456,34567,45678]:zero plus four Uniform[-eps,eps] starts,
projected into valid image range. Per-image/model random generator keyed by
SHA256(seed|weight_filename|image_filename). The same target26 initialization
is shared across its two objective conditions. Do not call different manual
seeds with identical zero starts independent random-start attacks.

Four tasks: white-box v8-many,white-box11-many,target26-many objective,target26
native objective. Each1500×5 starts=7500; total30,000 attack cases. Both target26
objectives evaluated on native one-to-one AND one-to-many head; source v8/11
perturbations transferred to native26. Native always counted directly at score
column4>.25; no native NMS. Many count uses fixed torchvision NMS,IoU.45,nc80,
conf.25,maxdet300. Additional cap variants output50/preNMS300/preNMS1000 are
descriptive workload endpoints,not native defense settings.

AllFP32,unfused,eval,frozen weights,640-square resize,100 PGD steps,eps16/255,
alpha2/255,temp.02,final iterate. Many proxy unchanged class-max soft count on
all anchors; native proxy same soft threshold on native top300 scores. Select
engine/batch solely from eligible resource pilot; bind execution_lock SHA before
run. All fusion candidates failed predeclared numerical agreement and are excluded.
CUDA Graph captures ONE update,reset buffers before EXACTLY100 replays. Warmup
uses OLD4 calibration images,not large-cohort inputs. No changes to power/clocks.

Stages: first1000 across all four tasks/five starts,then added500 across all tasks.
No optional stopping by effects. Store every per-case count,candidate score
vector,post-NMS detection and native output with SHA,plus exact float32 adversarial
inputs on locked64 audit IDs per task/start. All-image deltas are NOT stored in
this new study; inputs have SHA+image/model/config/seed provenance. This policy
is prospective for bounded storage,not outcome-based deletion. Prior500 full
deltas stay untouched. Never claim full input replay for non-audit cases.
Per-batch completion markers; interrupted attempts retained. Resume skips first
completed attempt per batch,never choose by effect. No raw overwrites.

Primary per task/stage: image-level average over five starts of pairedK gain and
K>=50 rate; report each start separately. Two-level image×start percentile
bootstrap20,000,seed20261106. Primary count distribution uses median across images
of meanK across starts. Also native transfer rates per source. Wilson95% per
start(n=images),and image-level ANY-start reach(n=images); do not treat7500
correlated image-start cases as7500 independent images. Any/best-of-five is a
secondary500-gradient-query budget,not equal-cost comparison with a single start.
Report native scores,not latency. Nested stages not pooled as independent studies.
Comparisons of two26 objectives are paired same images/initialization. Source
model comparisons descriptive,not architecture causality. Preserve negative data.

Smoke first4 prefix1000 images,all tasks/starts; do not adjust scientific settings
after smoke counts. Then full campaign. Tag prereg-nested1500-five-starts-20261106
before smoke. Exact cohort and execution hashes committed alongside protocol.

Implementation amendment A after smoke(80 cases),before full: startup gradient
cosines are floating diagnostics and can change slightly after process restart.
Resume compares all fixed config/provenance/weights/images,retains original
manifest,and appends a new resume-QA record instead of requiring diagnostic
float equality. Completed markers are not overwritten. No cohort,objective,
seed,budget,engine or analysis endpoint change. New amendment tag before full.
