# Candidate cap workload check, 06/11/2026

Already seen M2/M3/M4 results and COCO clean accuracy; this is a new prospective
mechanistic count endpoint, not a timing study. Locked20 final-iterate M3 inputs,
YOLOv8n-seg cuda FP32 conf.25 IoU.45 nc80 (accuracy eval separately conf.001).
Variants baseline(max_det300,max_nms30000), output50(50,30000), prenms300(300,300),
prenms1000(300,1000). Instrument ONLY torchvision.ops.nms to record actual input
box count, delegate unchanged to original kernel. Record output K and input hashes.
Primary: candidate count reaching kernel with each pre-NMS cap vs baseline;
secondary retained K and change induced by output cap. Assert candidate<=cap and
output<=max_det. No latency/availability claim or accuracy claim from these counts.
Smoke2 then full20 as authorized. No optimization, no image selection or changed
threshold after results. Tag prereg-cap-workload-20261106 before counts.
