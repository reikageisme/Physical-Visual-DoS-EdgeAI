# Full COCO validation defense study, 06/11/2026

Tag prereg-coco-defense-20261106 before setup/smoke/evaluation.
User explicitly authorized full work on rented machine. Official COCO val2017
5000 images only (not train images); segmentation labels from installed
Ultralytics coco.yaml's ASSETS_URL/coco2017labels-segments.zip. Record archive
hash/URL, integrity check, individual image/label SHA; never alter existing datasets.
Set train=val directory only to satisfy schema; NO training/optimization occurs.

YOLOv8n-seg and YOLO11n-seg pretrained, fixed torchvision NMS. imgsz640 letterbox,
batch8, device0, workers0, FP32 default, conf.001, IoU.7, multi_label validation
default. This mAP threshold differs intentionally from attack conf.25.
Four configurations per model: baseline(max_det300,max_nms30000), maxdet50
(50,30000), prenms300(300,300), prenms1000(300,1000).
Monkeypatch ONLY max_nms argument of installed non_max_suppression for pre-NMS
caps; output max_det unchanged. Restore callable after each variant. End2end
models excluded from this study. No hyperparameter selection from val results.

Primary endpoint: segmentation mAP50-95 absolute change against baseline using
Ultralytics polygon-label validator. Secondary box mAP, P/R and segmentation
mAP50. These are Ultralytics metric implementation, NOT a claim of independently
recomputed official COCO API metrics. Preserve detection-level metric input
arrays (tp/conf/pred_cls/target_cls and mask equivalents), dataset/model hashes,
validator source, results, and denominators for replay. No fabricated CIs.
No significant-difference/equivalence claim from descriptive mAP deltas.
No inference of attack resistance from clean mAP alone: report workload cap
mechanism separately, max_det caps masks not work before NMS.

Smoke first16 lexicographic validation images, same8 model/variant evaluations;
then full all5000 as user authorized. Exact seen count required. Incomplete
download/missing labels/API mismatch: stop affected stage and keep artifacts;
log diagnosis, fix before full under explicit amendment if implementation changes.
No timing claims from validator-reported speeds; this is accuracy evaluation.

Implementation amendment A before dataset download/any accuracy result:
PyYAML is absent; use installed Ultralytics YAML.load/save rather than install
new packages. No dataset, metric, cap or threshold changes.

Implementation amendment B before any accuracy result: official images download
completed but rented Python CA store rejected GitHub certificate. Fetch SAME
official label URL with local Windows curl (TLS verified), transfer archive via
pscp, record hash. Setup resumes only verified ZIPs and byte-compares existing
extracted files, never overwrites differing evidence. Network reads timeout60s.

Implementation amendment C: first smoke baseline16 completed metric calculation
but bookkeeping failed because model.val does not expose model.validator in this
version. Capture the constructed RawValidator instance directly to read seen and
save_dir. One baseline smoke result seen; no full or defense results seen. Keep
failed smoke directory, rerun in smoke_v2, unchanged metric/caps/data.
