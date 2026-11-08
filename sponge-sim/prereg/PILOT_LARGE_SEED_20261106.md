# Runtime-engine QA before nested1000/1500 multi-seed study

User has5–6 rental hours and requested1000/1500 images,multiple seeds,and native
YOLO26 diagnosis. This pilot is compute-resource QA,not inference-latency evidence.
Fixed OLD COCO128 IDs9,25,30,34 (not new validation cohort). Evaluate batches4/8/16
in fresh subprocesses for v8n/11n/26n-seg. FP32,eval,weights frozen. Native26 head
one-to-one explicitly enabled; its objective uses native confidence column4.
For v8/11 same candidate proxy as earlier. Each uses100steps,eps16/255,alpha2/255.

Compare eager in-place projection with a captured CUDA Graph step. Verify one-step
delta cosine>.999 vs eager,finite,perturbation budget,peak reserved<=12GiB. Keep
all pilot results/errors. Choose per model eligible execution+batch by smallest
processing seconds/image,then smaller batch. No K count/effect criterion.
CUDA Graph captures one update; buffers reset BEFORE exactly100replays. Capturing
must not add an optimization step to the measured attack. No dtype/budget changes.
Record hashes of pilot/engine/weights and tag qa-large-seed-engine-20261106 before
execution. Final engine and cohort locked/tagged separately before main study.
