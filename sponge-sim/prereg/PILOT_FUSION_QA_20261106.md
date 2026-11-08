# FP32 fusion runtime QA before large study

Prior runtime pilot results observed; no large-study effect data observed.
Evaluate standard model.fuse(verbose=False) on same old4 calibration images,
models v8n/11n/native26,batches4/8,fresh subprocess. Native26 one-to-one enabled
BEFORE fusion; fusion removes the unused one-to-many inference branch. It is
not a change to weights training or threat budget. Original weight hashes kept.
Require unfused/fused one-step delta cosine>.999 and max class-score error<1e-4
(native sorted confidence scores to allow tied top-k ordering), in addition to
prior finite/budget/memory/capture QA gates. Record every outcome/error.
Choose overall eligible fused/unfused+engine+batch by processing seconds/image,
never K/effect. Final exact choice locked before any large cohort measurements.
Record FP32 fusion as a new execution configuration, not bitwise equivalence.
Tag qa-fusion-large-seed-20261106 before pilot.
