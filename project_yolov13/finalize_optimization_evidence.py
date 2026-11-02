"""Audit completed evidence, record incremental cost and preserve source hashes."""
from __future__ import annotations
import csv,json,math,sys,statistics
from datetime import datetime
from pathlib import Path
from pipeline_utils import PROJECT_ROOT,configure_project_environment,require_local_venv,project_path,sha256_file,utc_timestamp,write_json
configure_project_environment()

def main():
    require_local_venv()
    summary_path=project_path("results/optimization/final_summary.json",must_exist=True)
    summary=json.loads(summary_path.read_text(encoding="utf-8"));protocol=summary["protocol"]
    if summary["status"]!="complete":raise RuntimeError("Study must be complete before final audit")
    records=summary["screening_records"]+summary["confirmation_records"]
    if len(summary["screening_records"])!=len(protocol["candidate_configurations"]) or len(summary["confirmation_records"])!=6:raise RuntimeError("Incomplete candidate/confirmation coverage")
    errors=[];checks=[];lock_time=datetime.fromisoformat(summary["winner_lock"]["locked_at_utc"])
    for record in records:
        checkpoint=project_path(record["checkpoint"],must_exist=True);csv_path=project_path(record["results_csv"],must_exist=True)
        if sha256_file(checkpoint)!=record["checkpoint_sha256"]:errors.append(f"Checkpointhash {record['configuration']} seed{record['seed']}")
        if sha256_file(csv_path)!=record["results_csv_sha256"]:errors.append(f"CSVhash {record['configuration']} seed{record['seed']}")
        with csv_path.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
        if len(rows)!=6:errors.append(f"Unexpectedepochcount {record['configuration']} seed{record['seed']}")
        best=max(rows,key=lambda x:float(x["metrics/mAP50-95(B)"]))
        if int(float(best["epoch"]))!=record["selected_epoch"]:errors.append(f"Wrongval-selected epoch {record['configuration']} seed{record['seed']}")
        metric_columns={"precision":"metrics/precision(B)","recall":"metrics/recall(B)","map50":"metrics/mAP50(B)","map50_95":"metrics/mAP50-95(B)"}
        for key,column in metric_columns.items():
            if float(best[column])!=record["validation"][key]:errors.append(f"Selectedvalmetric {record['configuration']} seed{record['seed']} {key}")
        protocol_fp32=record.get("validation_fp32",{}).get("evaluation_protocol",{})
        if protocol_fp32.get("half") is not False or protocol_fp32.get("batch")!=8 or protocol_fp32.get("imgsz")!=416 or protocol_fp32.get("split")!="val":errors.append("CandidateFP32rankingprotocolmismatch")
        if record["stage"]=="screening" and datetime.fromisoformat(record["validation_fp32_measured_at_utc"])>lock_time:errors.append("ScreeningFP32afterwinnerlock")
        if record["dataset_manifest_sha256"]!=sha256_file("dataset/fullscale_person/manifest.json"):errors.append("Datasetmanifestchanged")
        if record["stage"]=="confirmation":
            if record["seed"] not in protocol["confirmation_seeds"] or record["seed"]==42:errors.append("Invalidindependentseed")
            if datetime.fromisoformat(record["test_evaluated_at_utc"])<lock_time:errors.append("Testbeforelock")
            for key in metric_columns:
                value=record["test"][key]
                if not math.isfinite(value) or not 0<=value<=1:errors.append(f"Invalidtestmetric {key}")
            latency=record["latency"]
            if latency["measured_frames"]!=500 or latency["warmup_frames"]!=32 or latency["batch"]!=1 or latency["imgsz"]!=416 or latency["timing_precision"]!="fp32":errors.append("Latencyprotocolmismatch")
            if latency["latency_median_ms_per_frame"]<=0 or abs(latency["fps_from_median_latency"]-1000/latency["latency_median_ms_per_frame"])>1e-10:errors.append("LatencyFPSinconsistent")
            if latency.get("protocol_id")!="actual-batch1-txt-loader-v2" or latency.get("actual_batch_size")!=1 or latency.get("actual_prediction_batches")!=532:errors.append("Unverifiedactualbatch1timing")
            if set(latency.get("network_forward_batch_size_counts",{}))!={"1"} or latency["network_forward_batch_size_counts"].get("1",0)<532:errors.append("Networkforwardbatchproofinvalid")
            if latency.get("loader",{}).get("class_name")!="LoadImagesAndVideos" or latency["loader"].get("actual_batch_size")!=1:errors.append("Timingloaderinvalid")
            if sha256_file(latency["loader"]["source_file"])!=latency["loader"]["source_file_sha256"]:errors.append("Timingsourcehashmismatch")
            samples=latency.get("per_frame_total_ms",[])
            if len(samples)!=500 or len(set(samples))<2 or abs(statistics.median(samples)-latency["latency_median_ms_per_frame"])>1e-10:errors.append("Perframetiminginvalid")
            for domain,expected in{"coco":1500,"advtshirt":168}.items():
                if record["domain_test"][domain]["images"]!=expected:errors.append("Testdomaincountincorrect")
        checks.append({"configuration":record["configuration"],"seed":record["seed"],"stage":record["stage"],"selected_epoch":record["selected_epoch"],"checkpoint_sha256_verified":True,"history_sha256_verified":True,"training_origin":record.get("training_origin","fresh")})
    if errors:
        write_json("results/optimization/metric_integrity_audit.json",{"status":"failed","errors":errors,"audited_at_utc":utc_timestamp()});raise RuntimeError(errors)
    fresh=[record for record in records if record.get("fresh_training",True)]
    reused=[record for record in records if not record.get("fresh_training",True)]
    summary["execution_cost"]={"fresh_training_runs":len(fresh),"fresh_training_epochs":sum(record["epochs_completed"] for record in fresh),"historical_control_prefixes_reused":len(reused),"historical_control_prefix_epochs":sum(record["epochs_completed"] for record in reused),"actual_total_new_training_seconds":sum(record["actual_training_seconds"] for record in fresh),"historical_prefix_equivalent_training_seconds":sum(record["actual_training_seconds"] for record in reused),"incremental_new_training_seconds_for_reused_controls":0,"clock_scope":"CSVelapsed includes per-epoch training andvalidation; excludes pretraineddiagnostics, finaltest/latency, setup andoverall orchestration idle; historicalprefix durations are NOT newconsumption"}
    if summary.get("interruption_recovery"):
        event=summary["interruption_recovery"]
        summary["execution_cost"].update(interrupted_extra_attempts=1,total_fresh_training_attempts=len(fresh)+1,completed_new_training_seconds=summary["execution_cost"]["actual_total_new_training_seconds"],interrupted_completed_epoch_seconds=event["known_completed_training_seconds"],interrupted_incomplete_epoch_seconds=None,total_new_training_seconds_lower_bound=summary["execution_cost"]["actual_total_new_training_seconds"]+event["known_completed_training_seconds"],total_new_training_seconds_is_exact=False)
        summary["execution_cost"]["clock_scope"]+="; actual_total_new_training_seconds sums COMPLETE fresh records only. Extra interrupted attempt consumed at least interrupted_completed_epoch_seconds plus unknown partial epoch3; total_new_training_seconds_lower_bound includes that known overhead."
    correction_path="results/optimization/latency_protocol_correction.json"
    verification_path="results/optimization/latency_batch1_verification.json"
    if project_path(correction_path).exists():
        verification=json.loads(project_path(verification_path,must_exist=True).read_text(encoding="utf-8"))
        if verification["status"]!="passed":raise RuntimeError("Actual batch1 method verification failed")
        summary["latency_protocol_correction_evidence"]={"correction_path":correction_path,"correction_sha256":sha256_file(correction_path),"verification_path":verification_path,"verification_sha256":sha256_file(verification_path),"legacy_latency_excluded":True,"probe_performance_excluded":True}
        limitation="Archived historical latency/FPS came from a Python-list loader that processed532images in one batch despite batch1 requested; those timings are invalid for batch1 and excluded. Corrected timings use verified fileloader batch1 with forward-hook proof,32warmup and500 measured frames. Accuracy protocol and locked checkpoints are unchanged."
        if limitation not in summary["limitations"]:summary["limitations"].append(limitation)
    historical_path="results/optimization/historical_latency_retimes.json"
    if project_path(historical_path).exists():
        historical=json.loads(project_path(historical_path).read_text(encoding="utf-8"))
        if historical["status"] not in("complete","complete_with_exclusions"):raise RuntimeError("Historical retiming incomplete")
        for item in historical["records"]:
            if sha256_file(item["checkpoint"])!=item["checkpoint_sha256"] or sha256_file(item["accuracy_source_path"])!=item["accuracy_source_sha256"]:raise RuntimeError("Historical checkpoint/accuracy-source hash mismatch")
            timing=item["latency"]
            if timing["protocol_id"]!="actual-batch1-txt-loader-v2" or timing["actual_prediction_batches"]!=532 or timing["actual_batch_size"]!=1 or set(timing["network_forward_batch_size_counts"])!={"1"}:raise RuntimeError("Historical batch1 verification failed")
            if len(timing["per_frame_total_ms"])!=500 or abs(statistics.median(timing["per_frame_total_ms"])-timing["latency_median_ms_per_frame"])>1e-10:raise RuntimeError("Historical latency median mismatch")
            if item["accuracy_reevaluated"] or not item["legacy_latency_excluded"]:raise RuntimeError("Historical accuracy or legacy timing status invalid")
        summary["historical_latency_retiming_evidence"]={"path":historical_path,"sha256":sha256_file(historical_path),"status":historical["status"],"successful_records":len(historical["records"]),"excluded_records":len(historical["exclusions"]),"accuracy_reevaluated":False,"legacy_latency_excluded":True}
    sources=["optimization_train_one.py","run_optimization_study.py","inspect_optimization_head_transfer.py","probe_optimization_loader.py","audit_optimization_control_reuse.py","finalize_optimization_evidence.py","finalize_optimization_recovery_evidence.py","verify_optimization_latency_loader.py","retime_historical_sweetspots.py","diagnose_optimization_bn.py","evaluate_optimization_selection_fp32.py","amend_optimization_protocol.py"]
    summary["optimization_source_sha256"]={source:sha256_file(source) for source in sources}
    if reused:summary["historical_control_reuse_audit"]=json.loads(project_path("results/optimization/historical_control_reuse_audit.json",must_exist=True).read_text(encoding="utf-8"))
    summary["final_integrity_audited_at_utc"]=utc_timestamp()
    write_json(summary_path,summary)
    write_json("results/optimization/metric_integrity_audit.json",{"status":"complete","audited_at_utc":utc_timestamp(),"records_checked":len(records),"independent_confirmation_records":6,"new_candidate_test_evaluations_after_lock":True,"fresh_training_runs":len(fresh),"reused_historical_control_prefixes":len(reused),"records":checks,"final_summary_sha256":sha256_file(summary_path)})
    print(json.dumps(summary["execution_cost"],ensure_ascii=False),flush=True)
if __name__=="__main__":main()
