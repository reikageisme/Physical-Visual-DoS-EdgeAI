"""Sequential GPU screening, independent-seed confirmation, locked test and timing."""
from __future__ import annotations
import argparse,json,os,random,statistics,subprocess,sys,time
from pathlib import Path
from pipeline_utils import PROJECT_ROOT,configure_project_environment,require_local_venv,project_path,sha256_file,utc_timestamp,write_json
configure_project_environment()
OUT=project_path("results/optimization")
LATENCY_PROTOCOL="actual-batch1-txt-loader-v2"

def read(path):return json.loads(project_path(path,must_exist=True).read_text(encoding="utf-8"))
def child(args):
    env=os.environ.copy();env["PYTHONUTF8"]="1"
    result=subprocess.run([sys.executable,*args],cwd=PROJECT_ROOT,env=env,check=False)
    if result.returncode:raise RuntimeError(f"Child failed: {args} (exit {result.returncode})")

def summarize(values):
    from scipy.stats import t
    values=[float(x) for x in values];n=len(values);mean=statistics.mean(values);sd=statistics.stdev(values) if n>1 else None
    half=float(t.ppf(.975,n-1))*sd/n**.5 if n>1 else None
    return {"n":n,"mean":mean,"sd":sd,"ci95_low":mean-half if half is not None else None,"ci95_high":mean+half if half is not None else None}

def benchmark(checkpoint,original80=False,warmup_frames=32,measured_frames=500):
    import numpy as np
    from optimization_train_one import load_api
    YOLO=load_api();model=YOLO(str(project_path(checkpoint,must_exist=True)))
    manifest=read("dataset/fullscale_person/manifest.json")
    images=sorted(str(project_path(x["image"],must_exist=True)) for x in manifest["split_manifest"]["test"])
    selected=random.Random(2026).sample(images,warmup_frames+measured_frames);timings={x:[] for x in("preprocess","inference","postprocess")}
    source=project_path(f"results/optimization/latency_sources/sample-{warmup_frames}-{measured_frames}.txt");source.parent.mkdir(parents=True,exist_ok=True);source.write_text("\n".join(selected)+"\n",encoding="utf-8")
    from collections import Counter
    forward_batches=Counter();loader_evidence={};callback_batches=[]
    def check_forward(module,inputs):
        size=int(inputs[0].shape[0]);forward_batches[size]+=1
        if size!=1:raise RuntimeError(f"Timing requires actual batch1, got forwardbatch{size}")
    hook=model.model.register_forward_pre_hook(check_forward)
    def check_start(predictor):
        dataset=predictor.dataset
        if dataset.__class__.__name__!="LoadImagesAndVideos" or dataset.bs!=1:raise RuntimeError("Timing loader is not genuine batch1 fileloader")
        if set(dataset.files)!=set(selected):raise RuntimeError("Timing source images changed")
        # LoadImagesAndVideos sorts text-list paths; restore the frozen random
        # sample order before iteration so the intended first32 are warmup.
        dataset.files=list(selected)
        loader_evidence.update(class_name=dataset.__class__.__name__,actual_batch_size=dataset.bs,source_type="project-local text file",sample_order="random.Random(2026).sample(sorted(test_images),532); restored after loader sorting" if len(selected)==532 else "probe sample",source_file=str(source.relative_to(PROJECT_ROOT)).replace("\\","/"),source_file_sha256=sha256_file(source))
    def check_batch(predictor):
        size=len(predictor.batch[1]);callback_batches.append(size)
        if size!=1:raise RuntimeError(f"Prediction callback saw actual batch{size}")
    model.add_callback("on_predict_start",check_start);model.add_callback("on_predict_batch_start",check_batch)
    kwargs={"classes":[0]} if original80 else {}
    try:
        stream=model.predict(source=str(source),imgsz=416,batch=1,device="0",half=False,stream=True,verbose=False,save=False,plots=False,**kwargs)
        for index,result in enumerate(stream):
            if index>=warmup_frames:
                for key in timings:timings[key].append(float(result.speed[key]))
    finally:hook.remove()
    if len(timings["inference"])!=measured_frames or len(callback_batches)!=len(selected) or set(forward_batches)!={1}:raise RuntimeError("Timing batch/frame verification failed")
    totals=[sum(timings[k][i] for k in timings) for i in range(measured_frames)];lat=float(np.median(totals))
    return {"protocol_id":LATENCY_PROTOCOL,"benchmark_source_sha256":sha256_file("run_optimization_study.py"),"checkpoint_sha256":sha256_file(checkpoint),"latency_median_ms_per_frame":lat,"fps_from_median_latency":1000/lat,"stage_median_ms_per_frame":{k:float(np.median(v)) for k,v in timings.items()},"latency_p95_ms_per_frame":float(np.percentile(totals,95)),"warmup_frames":warmup_frames,"measured_frames":measured_frames,"imgsz":416,"batch":1,"actual_batch_size":1,"actual_prediction_batches":len(callback_batches),"network_forward_batch_size_counts":dict(forward_batches),"network_forward_count_includes_backend_warmup":True,"loader":loader_evidence,"per_frame_total_ms":totals,"unique_measured_total_times":len(set(totals)),"timing_precision":"fp32","timing_scope":"Ultralytics preprocessing + inference + postprocessing; excludes decode/I/O","sample_seed":2026,"measured_at_utc":utc_timestamp()}

def evaluate_record(path):
    from optimization_train_one import evaluate
    value=read(path)
    lock=read("results/optimization/winner_lock.json")
    protocol=read("results/optimization/protocol.json")
    if value["configuration"] not in("original_sgd",lock["winner_configuration"]) or value["seed"] not in protocol["confirmation_seeds"]:
        raise RuntimeError("Only locked winner/control independent confirmation records may access test")
    if value.get("test") is not None and value.get("latency",{}).get("protocol_id")==LATENCY_PROTOCOL and value.get("domain_validation") and value.get("domain_test"):return
    key=f"{value['configuration']}-seed{value['seed']}"
    if value.get("test") is None:
        value["test"]=evaluate(value["checkpoint"],split="test",run_name=f"locked-{key}-test")
        value["test_accuracy_measured_at_utc"]=utc_timestamp()
    # Persist accuracy before any resource measurement, so a timing error cannot
    # discard completed locked accuracy evaluation.
    write_json(path,value)
    value["latency"]=benchmark(value["checkpoint"])
    write_json(path,value)
    if value.get("domain_validation") and value.get("domain_test"):
        return
    import yaml
    manifest=read("dataset/fullscale_person/manifest.json")
    domains={"coco":"COCO 2017","advtshirt":next(x["source"] for x in manifest["split_manifest"]["test"] if "Adv" in x["source"])}
    value["domain_validation"]={};value["domain_test"]={}
    for domain,source in domains.items():
        domain_config={"path":str(project_path("dataset/fullscale_person")),"nc":1,"names":{0:"person"}}
        sizes={}
        for split in("val","test"):
            selected=[x for x in manifest["split_manifest"][split] if x["source"]==source]
            images=project_path(f"results/optimization/domain_lists/{domain}-{split}.txt");images.parent.mkdir(parents=True,exist_ok=True)
            images.write_text("\n".join(str(project_path(x["image"],must_exist=True)) for x in selected)+"\n",encoding="utf-8")
            domain_config[split]=str(images);sizes[split]={"images":len(selected),"person_instances":sum(x["person_count"] for x in selected),"negative_images":sum(x["person_count"]==0 for x in selected)}
        domain_config["train"]=str(project_path("dataset/fullscale_person/images/train"))
        config_path=project_path(f"results/optimization/domain_lists/{domain}.yaml");config_path.write_text(yaml.safe_dump(domain_config,sort_keys=False),encoding="utf-8")
        for split in("val","test"):
            metrics=evaluate(value["checkpoint"],split=split,run_name=f"locked-{key}-{domain}-{split}",data_yaml=config_path)
            value["domain_validation" if split=="val" else "domain_test"][domain]={**sizes[split],"metrics":metrics}
    value["test_evaluated_at_utc"]=utc_timestamp()
    write_json(path,value)

def execute():
    from optimization_train_one import CONFIGS
    require_local_venv();OUT.mkdir(parents=True,exist_ok=True)
    pause_flag=OUT/"pause_for_protocol_amendment.flag"
    if pause_flag.exists():pause_flag.unlink()
    protocol_path=OUT/"protocol.json"
    protocol={"study_id":"yolov13s-bounded-optimization-20261013","frozen_at_utc":utc_timestamp(),"model_key":"yolov13s","candidate_configurations":CONFIGS,"screening_seed":42,"confirmation_seeds":[100,2026,777],"epochs_per_run":6,"maximum_training_runs":11,"selection_criterion":"maximum validation mAP50:95 of any completed epoch; highest seed42 candidate wins; original_sgd excluded from optimized candidate pool","ties":"configuration order as frozen in candidate_configurations","test_access":"Only after winner configuration is locked, on independent confirmation seeds, no test-driven retuning","dataset_manifest_sha256":sha256_file("dataset/fullscale_person/manifest.json"),"pretrained_weights_sha256":sha256_file("weights/yolov13s.pt"),"untuned_control":"original SGD0.01 linear scheduler horizon300, stopped after6epochs; optimized scheduler horizon6 is an intervention","compute_match":"same six full epochs,7786primaryimage draws per epoch,416px,batch8; report actual times because budgets are not exactly equal wall time","sampling_note":"Bundled advanced candidate uses replacement WeightedRandomSampler25%expectedAdv; secondarymosaic indices remainuniform; not component-isolating ablation","planned_analysis":"paired seedwise test mAP95 delta, sample SD and Studentt95%CI; exploratory, n3 per group"}
    if protocol_path.exists():protocol=read(protocol_path)
    else:write_json(protocol_path,protocol)
    state_path=OUT/"orchestrator_state.json"
    state={"status":"running","started_at_utc":utc_timestamp(),"phase":"pretrained_validation","jobs":[]};write_json(state_path,state)
    if not (OUT/"pretrained_validation.json").exists():child(["optimization_train_one.py","--pretrained-val"])
    screening=[]
    for config in protocol["candidate_configurations"]:
        state.update(phase="screening",current_configuration=config,current_seed=42);write_json(state_path,state)
        training_args=["optimization_train_one.py","--config",config,"--seed","42","--stage","screening","--epochs","6"]
        recovery_path=OUT/"interruption_recovery_event.json"
        if recovery_path.exists():
            recovery=read(recovery_path)
            if recovery["configuration"]==config:training_args.extend(["--run-tag",recovery["replacement_run_tag"]])
        child(training_args)
        child(["evaluate_optimization_selection_fp32.py","--record",f"results/optimization/screening_records/{config}-seed42.json"])
        record=read(f"results/optimization/screening_records/{config}-seed42.json");screening.append(record);state["jobs"].append({"stage":"screening","configuration":config,"seed":42,"status":"complete"});write_json(state_path,state)
    optimized=[r for r in screening if r["configuration"]!="original_sgd"]
    winner=max(optimized,key=lambda r:r["validation_fp32"]["map50_95"])["configuration"]
    lock_path=OUT/"winner_lock.json"
    if lock_path.exists():
        lock=read(lock_path)
        if lock["winner_configuration"]!=winner:raise RuntimeError("Winner lock disagrees with validation results")
    else:
        lock={"locked_at_utc":utc_timestamp(),"winner_configuration":winner,"criterion":protocol["selection_criterion"],"screening": [{"configuration":r["configuration"],"validation":r["validation"],"validation_fp32":r["validation_fp32"],"selected_epoch":r["selected_epoch"],"checkpoint":r["checkpoint"]} for r in screening],"test_metrics_seen_before_lock":False,"test_access_statement_scope":"New optimization-candidate test metrics only; prior fullscale study already examined this test split."};write_json(lock_path,lock)
    deployment_path=OUT/"deployment_choice_lock.json"
    if deployment_path.exists():deployment_choice=read(deployment_path)
    else:
        restored=next(r for r in screening if r["configuration"]=="adamw_transfer_restored")
        trained=next(r for r in screening if r["configuration"]==winner)
        zero=restored["initialization"]
        prefer_zero=zero["zero_finetuning_validation"]["map50_95"]>=trained["validation_fp32"]["map50_95"]
        deployment_choice={"locked_at_utc":utc_timestamp(),"choice":"zero_restored_person" if prefer_zero else "trained_winner","trained_configuration":winner,"criterion":protocol["deployment_choice_policy"],"zero_validation_fp32":zero["zero_finetuning_validation"],"trained_validation_fp32":trained["validation_fp32"],"checkpoint":zero["initial_checkpoint"] if prefer_zero else trained["checkpoint"],"checkpoint_sha256":zero["initial_checkpoint_sha256"] if prefer_zero else trained["checkpoint_sha256"],"n":1,"seed42_selection_only":True,"test_seen_before_decision":False};write_json(deployment_path,deployment_choice)
    confirmations=[]
    for seed in protocol["confirmation_seeds"]:
        for config in("original_sgd",winner):
            state.update(phase="independent_confirmation",current_configuration=config,current_seed=seed);write_json(state_path,state)
            if config=="original_sgd" and protocol.get("confirmation_control_mode")=="historical_deterministic_prefix":
                child(["audit_optimization_control_reuse.py","--reuse-seed",str(seed)])
            else:
                child(["optimization_train_one.py","--config",config,"--seed",str(seed),"--stage","confirmation","--epochs","6"])
            path=f"results/optimization/confirmation_records/{config}-seed{seed}.json";confirmations.append(path)
            child(["evaluate_optimization_selection_fp32.py","--record",path])
            state["jobs"].append({"stage":"confirmation","configuration":config,"seed":seed,"status":"complete","training_origin":"historical_deterministic_prefix" if config=="original_sgd" and protocol.get("confirmation_control_mode")=="historical_deterministic_prefix" else "fresh"});write_json(state_path,state)
    state.update(phase="locked_test_and_latency");write_json(state_path,state)
    pretrained_test_path=OUT/"pretrained_locked_test_diagnostic.json"
    if not pretrained_test_path.exists():
        from optimization_train_one import evaluate
        pretrained_test={"status":"complete","model_key":"yolov13s","n":1,"checkpoint":"weights/yolov13s.pt","head_classes":80,"prediction_classes":[0],"interpretation":"zero-finetuning diagnostic, original80classhead; not an independentseedgroup or proposedarchitecture","test":evaluate("weights/yolov13s.pt",split="test",original80=True,run_name="locked-pretrained-person-test"),"latency":benchmark("weights/yolov13s.pt",original80=True),"winner_lock_at_utc":lock["locked_at_utc"],"measured_at_utc":utc_timestamp()};write_json(pretrained_test_path,pretrained_test)
    elif read(pretrained_test_path).get("latency",{}).get("protocol_id")!=LATENCY_PROTOCOL:
        previous=read(pretrained_test_path)
        archive=OUT/"invalid_latency_protocol"/"pretrained_locked_test_diagnostic-original-list-loader.json"
        if not archive.exists():write_json(archive,previous)
        previous["invalid_previous_latency_archive"]=str(archive.relative_to(PROJECT_ROOT)).replace("\\","/")
        previous["latency"]=benchmark(previous["checkpoint"],original80=True);previous["latency_corrected_at_utc"]=utc_timestamp();write_json(pretrained_test_path,previous)
    zero_test_path=OUT/"zero_locked_test_diagnostic.json"
    if deployment_choice["choice"]=="zero_restored_person" and not zero_test_path.exists():
        from optimization_train_one import evaluate
        checkpoint=deployment_choice["checkpoint"]
        zero_test={"status":"complete","name":"YOLOv13s restored person zero-finetuning","model_key":"yolov13s","n":1,"checkpoint":checkpoint,"checkpoint_sha256":sha256_file(checkpoint),"head_classes":1,"prediction_classes":[0],"test":evaluate(checkpoint,split="test",run_name="locked-zero-restored-person-test"),"latency":benchmark(checkpoint),"deployment_choice_locked_at_utc":deployment_choice["locked_at_utc"],"measured_at_utc":utc_timestamp(),"interpretation":"Deterministic zero-finetuning deployment choice selected by validation before test; no independent-seed SD or CI"};write_json(zero_test_path,zero_test)
    for path in confirmations:child(["run_optimization_study.py","--evaluate-record",path])
    records=[read(path) for path in confirmations];groups=[]
    for config in("original_sgd",winner):
        selected=[r for r in records if r["configuration"]==config]
        metrics={k:summarize([r["test"][k] for r in selected]) for k in("precision","recall","map50","map50_95","parameters_m","flops_g")}
        metrics.update(latency_ms=summarize([r["latency"]["latency_median_ms_per_frame"] for r in selected]),fps=summarize([r["latency"]["fps_from_median_latency"] for r in selected]))
        domains={split:{domain:{"images":selected[0][split][domain]["images"],"person_instances":selected[0][split][domain]["person_instances"],"negative_images":selected[0][split][domain]["negative_images"],"metrics":{k:summarize([r[split][domain]["metrics"][k] for r in selected]) for k in("precision","recall","map50","map50_95")}} for domain in("coco","advtshirt")} for split in("domain_validation","domain_test")}
        groups.append({"name":"YOLOv13s untuned 6epochs" if config=="original_sgd" else f"YOLOv13s optimized ({config})","model_key":"yolov13s","configuration":config,"n":3,"metrics":metrics,"validation_metrics":{k:summarize([r["validation_fp32"][k] for r in selected]) for k in("precision","recall","map50","map50_95")},"actual_training_seconds":summarize([r["actual_training_seconds"] for r in selected]),"domain_metrics":domains,"seed_records":selected})
    paired=[]
    for seed in protocol["confirmation_seeds"]:
        control=next(r for r in records if r["seed"]==seed and r["configuration"]=="original_sgd")
        proposed=next(r for r in records if r["seed"]==seed and r["configuration"]==winner)
        paired.append(proposed["test"]["map50_95"]-control["test"]["map50_95"])
    final={"status":"complete","study_id":protocol["study_id"],"completed_at_utc":utc_timestamp(),"protocol":protocol,"winner_configuration":winner,"winner_lock":lock,"pretrained_validation":read(OUT/"pretrained_validation.json"),"pretrained_locked_test_diagnostic":read(pretrained_test_path),"head_transfer_audit":read(OUT/"head_transfer_audit.json"),"screening_records":screening,"confirmation_records":records,"groups":groups,"paired_test_map95_delta":summarize(paired),"limitations":["Exploratory tuning on reused validation set; only3independent confirmationseeds; not globaloptimum.","Test split previously examined in prior study; no new untouched holdout; exploratory replication, not fresh confirmation.","COCO training-image exposure in initial pretraining and near-frame/person correlations mayinflateperformance.","Advanced candidate bundles loss, augmentation andsampling changes; no isolated attribution toindividualcomponents.","Native pretrained diagnostic retains80classhead andfilters person; untuned training resets oneclass classifier and mismatched HyperACE tensors; not anidentical initializationcontrol.","Same6epochs but differing schedulerhorizons, architecture reconstruction/initialization interventions and actualwalltimes; combined engineering-recipe evidence, not a newarchitectural invention."]}
    final["deployment_choice"]=deployment_choice
    if (OUT/"latency_protocol_correction.json").exists():final["latency_protocol_correction"]=read(OUT/"latency_protocol_correction.json")
    if (OUT/"interruption_recovery_event.json").exists():
        final["interruption_recovery"]=read(OUT/"interruption_recovery_event.json")
        if (OUT/"interruption_replay_audit.json").exists():final["interruption_replay_audit"]=read(OUT/"interruption_replay_audit.json")
        final["limitations"].append("One tiny-LR screening attempt was interrupted after two complete epochs and part of epoch3, then rerun from initial weights with the identical frozen six-epoch recipe. Partial evidence is preserved and excluded from selection; extra compute is disclosed, not counted as another hyperparameter candidate.")
    final["zero_locked_test_diagnostic"]=read(zero_test_path) if zero_test_path.exists() else None
    final["bn_controlled_diagnostic"]=read(OUT/"bn_controlled_diagnostic.json") if (OUT/"bn_controlled_diagnostic.json").exists() else None
    final["limitations"].append("Two-stage checkpoint selection ranks only native-FP16-best checkpoints using FP32 validation; maymiss FP32best among other epochs. Adaptive6th recipe and zero deployment option are validation-informed; neither proves globaloptimality nor isolates components.")
    write_json(OUT/"final_summary.json",final);state.update(status="complete",phase="complete",finished_at_utc=utc_timestamp());write_json(state_path,state)
    print(json.dumps({"status":"complete","winner":winner,"paired_test_map95_delta":final["paired_test_map95_delta"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    require_local_venv();parser=argparse.ArgumentParser();parser.add_argument("--evaluate-record");a=parser.parse_args()
    if a.evaluate_record and (OUT/"pause_for_latency_protocol_repair.flag").exists():
        print("Intentional pause before next evaluation child for verified loader-protocol repair.",flush=True)
        raise SystemExit(75)
    if a.evaluate_record:evaluate_record(a.evaluate_record)
    else:execute()
