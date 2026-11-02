"""CPU drift and exclusive-GPU controlled BN-buffer restoration diagnostic."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from pipeline_utils import configure_project_environment,require_local_venv,project_path,sha256_file,utc_timestamp,write_json,PROJECT_ROOT
configure_project_environment()

def state_from(path):
    import torch
    value=torch.load(project_path(path,must_exist=True),map_location="cpu",weights_only=False)
    return (value.get("ema") or value["model"]).float().state_dict()

def cpu_drift(initial,trained):
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(project_path("yolov13_src",must_exist=True)))
    source=state_from(initial);destination=state_from(trained);groups={"bn_running_buffers":[],"other_float_tensors":[]};per_tensor=[]
    for key,first in source.items():
        second=destination[key]
        if first.shape!=second.shape:raise RuntimeError(f"Diagnostic shape mismatch: {key}")
        if not first.is_floating_point():continue
        group="bn_running_buffers" if key.endswith(("running_mean","running_var")) else "other_float_tensors"
        norm=float(first.float().norm());delta=float((second.float()-first.float()).norm())
        groups[group].append((norm,delta,first.numel()));per_tensor.append({"key":key,"group":group,"initial_l2":norm,"delta_l2":delta,"relative_l2":delta/norm if norm>1e-12 else None,"numel":first.numel()})
    import math,statistics
    summaries={}
    for group,items in groups.items():
        baseline=math.sqrt(sum(x[0]**2 for x in items));drift=math.sqrt(sum(x[1]**2 for x in items))
        ratios=[x["relative_l2"] for x in per_tensor if x["group"]==group and x["relative_l2"] is not None]
        summaries[group]={"tensor_count":len(items),"numel":sum(x[2] for x in items),"pooled_initial_l2":baseline,"pooled_delta_l2":drift,"pooled_relative_l2":drift/baseline,"median_per_tensor_relative_l2":statistics.median(ratios)}
    variances=[v for k,v in destination.items() if k.endswith("running_var")]
    return {"status":"complete","device":"CPU","initial_checkpoint":initial,"initial_checkpoint_sha256":sha256_file(initial),"trained_checkpoint":trained,"trained_checkpoint_sha256":sha256_file(trained),"groups":summaries,"ratio_denominator_policy":"Per-tensor ratio omitted when initial L2 <=1e-12; pooled ratio is L2delta/L2initial, unlike per-element relative drift. BN and parameter ratios describe different tensor groups and are not equivalent causal effect sizes.","bn_running_var_min":min(float(v.min()) for v in variances),"bn_running_var_nonpositive_elements":sum(int((v<=0).sum()) for v in variances),"largest_bn_relative_drift":sorted([x for x in per_tensor if x["group"]=="bn_running_buffers" and x["relative_l2"] is not None],key=lambda x:x["relative_l2"],reverse=True)[:12],"interpretation":"Tensor drift is descriptive; does not demonstrate BN-buffer causality without matched protocol evaluation.","audited_at_utc":utc_timestamp()}

def evaluate_swap():
    from optimization_train_one import evaluate,load_api
    import torch
    record=json.loads(project_path("results/optimization/screening_records/adamw_transfer_restored-seed42.json",must_exist=True).read_text(encoding="utf-8"))
    if record["status"]!="complete":raise RuntimeError("Currentrestoredcandidate must finish before exclusive GPU diagnostic")
    initial=record["initialization"]["initial_checkpoint"];trained=record["checkpoint"]
    output={"status":"running","trained_checkpoint":trained,"trained_selected_native_epoch":record["selected_epoch"],"initial_checkpoint":initial,"frozen_policy":"results/optimization/protocol.json adaptive_sixth_candidate_policy"}
    output["unchanged_fp32_batch8_validation"]=evaluate(trained,run_name="bn-diagnostic-unchanged-fp32-batch8")
    output["initial_fp16_batch16_validation"]=evaluate(initial,run_name="bn-diagnostic-initial-fp16-batch16",batch=16,half=True)
    output["trained_fp16_batch16_validation"]=evaluate(trained,run_name="bn-diagnostic-trained-fp16-batch16",batch=16,half=True)
    YOLO=load_api();model=YOLO(str(project_path(trained,must_exist=True)));source=state_from(initial);destination=model.model.float().state_dict();copied=[]
    with torch.no_grad():
        for key,value in destination.items():
            if key.endswith(("running_mean","running_var","num_batches_tracked")):
                value.copy_(source[key].to(value.dtype));copied.append(key)
    model.ckpt={"ema":None,"optimizer":None,"train_args":{"task":"detect","imgsz":416,"data":"dataset/data.yaml"}}
    modified=project_path("runs/optimization/diagnostics/restored-selected-with-initial-bn.pt");modified.parent.mkdir(parents=True,exist_ok=True);model.save(str(modified))
    output["bn_restored_checkpoint"]=str(modified.relative_to(PROJECT_ROOT)).replace("\\","/");output["bn_restored_checkpoint_sha256"]=sha256_file(modified);output["bn_buffer_keys_copied"]=copied
    output["bn_restored_fp32_batch8_validation"]=evaluate(modified,run_name="bn-diagnostic-initial-buffers-fp32-batch8")
    output["map95_gain_from_only_bn_buffer_replacement"]=output["bn_restored_fp32_batch8_validation"]["map50_95"]-output["unchanged_fp32_batch8_validation"]["map50_95"]
    output["cpu_drift"]=cpu_drift(initial,trained)
    output.update(status="complete",measured_at_utc=utc_timestamp(),interpretation="Controlled selected-checkpoint evaluation: trained parameters identical, only BN runningmean/var/counters changed; evidence specific to this checkpoint/protocol, not complete causaltrainingablation.")
    write_json("results/optimization/bn_controlled_diagnostic.json",output)
    # Preserve CSV-native checkpoint selection and separate deployment ranking.
    record["validation_fp32"]=output["unchanged_fp32_batch8_validation"]
    record["validation_fp32_measured_at_utc"]=utc_timestamp()
    record["selection_protocol"]={"within_trajectory":"maximum CSVnative FP16batch16 valmAP95 among6epochs","candidate_ranking":"ONLYnative-selected checkpoint revalidated FP32batch8; not exhaustive FP32epoch search","test_access":False}
    write_json("results/optimization/screening_records/adamw_transfer_restored-seed42.json",record)

def audit_swap_proof():
    import torch
    torch.set_num_threads(2);sys.path.insert(0,str(project_path("yolov13_src",must_exist=True)))
    path=project_path("results/optimization/bn_controlled_diagnostic.json",must_exist=True)
    output=json.loads(path.read_text(encoding="utf-8"))
    trained=state_from(output["trained_checkpoint"]);modified=state_from(output["bn_restored_checkpoint"]);initial=state_from(output["initial_checkpoint"])
    buffer_keys=[key for key in trained if key.endswith(("running_mean","running_var","num_batches_tracked"))]
    other_keys=[key for key in trained if key not in buffer_keys]
    differences=[key for key in other_keys if not torch.equal(trained[key],modified[key])]
    incorrect_buffers=[key for key in buffer_keys if not torch.equal(initial[key],modified[key])]
    if differences or incorrect_buffers:raise RuntimeError({"changed_non_bn_tensors":differences,"incorrectly_restored_buffers":incorrect_buffers})
    output["serialized_swap_tensor_proof"]={"non_bn_tensor_count":len(other_keys),"all_non_bn_tensors_exactly_equal":True,"bn_buffer_tensor_count":len(buffer_keys),"all_bn_buffers_exactly_equal_to_initial":True,"audited_at_utc":utc_timestamp(),"trained_checkpoint_sha256":sha256_file(output["trained_checkpoint"]),"modified_checkpoint_sha256":sha256_file(output["bn_restored_checkpoint"])}
    write_json(path,output)

if __name__=="__main__":
    require_local_venv();parser=argparse.ArgumentParser();parser.add_argument("--evaluate-swap",action="store_true");parser.add_argument("--audit-swap-proof",action="store_true");args=parser.parse_args()
    if args.evaluate_swap:evaluate_swap()
    elif args.audit_swap_proof:audit_swap_proof()
    else:
        record=json.loads(project_path("results/optimization/screening_records/adamw_transfer_restored-seed42.json",must_exist=True).read_text(encoding="utf-8"))
        initial=record["initialization"]["initial_checkpoint"];first=str(Path(initial).parent/"epoch0.pt")
        write_json("results/optimization/bn_first_epoch_cpu_drift.json",cpu_drift(initial,first))
