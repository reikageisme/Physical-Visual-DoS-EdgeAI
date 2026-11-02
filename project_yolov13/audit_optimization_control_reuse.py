"""Verify deterministic historical six-epoch controls before any reuse."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
from pipeline_utils import configure_project_environment,require_local_venv,project_path,sha256_file,utc_timestamp,write_json,PROJECT_ROOT
configure_project_environment()

def read(path):return json.loads(project_path(path,must_exist=True).read_text(encoding="utf-8"))
def history(path):
    with project_path(path,must_exist=True).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def perform_audit():
    require_local_venv();import torch,yaml
    torch.set_num_threads(2);sys.path.insert(0,str(project_path("yolov13_src",must_exist=True)))
    current=read("results/optimization/screening_records/original_sgd-seed42.json")
    old=read("results/fullscale_experiments/records/yolov13s/seed-42.json")
    fresh_rows=history(current["results_csv"]);old_rows=history(old["results_csv"])[:6]
    if current["status"]!="complete" or len(fresh_rows)!=6:raise RuntimeError("Fresh6epochcontrol must be complete before historicalreuseaudit")
    csv_mismatches=[]
    for new,previous in zip(fresh_rows,old_rows):
        for key in new:
            if key!="time" and float(new[key])!=float(previous[key]):csv_mismatches.append({"epoch":new["epoch"],"key":key,"new":new[key],"old":previous[key]})
    excluded_args={"name","project","save_dir"}
    new_args=yaml.safe_load(project_path(f"runs/optimization/{current['stage']}-yolov13s-original_sgd-seed42-6epochs/args.yaml").read_text(encoding="utf-8"))
    old_args_path=Path(old["results_csv"]).parent/"args.yaml"
    old_args=yaml.safe_load(project_path(old_args_path,must_exist=True).read_text(encoding="utf-8"))
    argument_mismatches={k:{"old":old_args.get(k),"new":new_args.get(k)} for k in set(new_args)|set(old_args) if k not in excluded_args and new_args.get(k)!=old_args.get(k)}
    provenance=read("results/fullscale_setup/provenance.json");source_bad=[]
    for relative,digest in provenance["model_sources"]["yolov13_src"]["python_files_sha256"].items():
        if sha256_file(project_path("yolov13_src")/relative)!=digest:source_bad.append(relative)
    manifest_match=sha256_file("dataset/fullscale_person/manifest.json")==provenance["dataset_manifest_sha256"]
    weights_match=sha256_file("weights/yolov13s.pt")==provenance["pretrained_weights_sha256"]["yolov13s"]
    tensor_checks=[]
    for epoch in range(1,7):
        old_checkpoint=project_path(Path(old["results_csv"]).parent/"weights"/f"epoch{epoch-1}.pt",must_exist=True)
        new_checkpoint=project_path(Path(current["checkpoint"]).parent/f"epoch{epoch-1}.pt",must_exist=True)
        a=torch.load(old_checkpoint,map_location="cpu",weights_only=False);b=torch.load(new_checkpoint,map_location="cpu",weights_only=False)
        first=(a.get("ema") or a["model"]).state_dict();second=(b.get("ema") or b["model"]).state_dict()
        bad=[k for k in first if k not in second or not torch.equal(first[k],second[k])]
        extra=[k for k in second if k not in first]
        tensor_checks.append({"epoch":epoch,"tensor_count":len(first),"nonidentical_keys":bad,"extra_keys":extra,"exact_match":not bad and not extra});del a,b,first,second
    eligible=not csv_mismatches and not argument_mismatches and not source_bad and manifest_match and weights_match and all(x["exact_match"] for x in tensor_checks)
    result={"status":"complete" if eligible else "incomparable_fresh_controls_required","audited_at_utc":utc_timestamp(),"historical_reuse_eligible":eligible,"csv_non_time_fields_first6_exact":not csv_mismatches,"csv_mismatches":csv_mismatches,"configuration_excluded_nonfunctional_arguments":sorted(excluded_args),"configuration_mismatches":argument_mismatches,"model_source_sha_mismatches":source_bad,"model_source_python_files_checked":len(provenance["model_sources"]["yolov13_src"]["python_files_sha256"]),"dataset_manifest_sha256_matches_historical":manifest_match,"pretrained_weights_sha256_matches_historical":weights_match,"saved_ema_tensor_checks":tensor_checks,"source_provenance_sha256":sha256_file("results/fullscale_setup/provenance.json"),"interpretation":"Same seeds, initialization, architecture, source, datasetmanifest, all functional arguments and exactly matching savedEMA tensors through6epochs support reusing prefix checkpoints forcontrols. Historicalwalltimes and dates differ; test andlatency must be freshly reevaluated after winnerlock.","limitation":"Manifesthash verifies splitidentity, not independently all historical image/label bytes; seed42 exact tensors/metrics provide strong end-to-end consistency evidence."}
    write_json("results/optimization/historical_control_reuse_audit.json",result);print(json.dumps({"historical_reuse_eligible":eligible,"argument_mismatches":argument_mismatches,"csv_mismatches":csv_mismatches},ensure_ascii=False),flush=True)

def reuse_control(seed):
    require_local_venv()
    audit=read("results/optimization/historical_control_reuse_audit.json")
    protocol=read("results/optimization/protocol.json")
    if not audit.get("historical_reuse_eligible") or protocol.get("confirmation_control_mode")!="historical_deterministic_prefix":raise RuntimeError("Reuse requires successfulaudit and frozenprotocolamendment")
    existing=project_path(f"results/optimization/confirmation_records/original_sgd-seed{seed}.json")
    if existing.exists():
        saved=read(existing)
        if saved.get("status")=="complete" and saved.get("training_origin")=="historical_deterministic_prefix":
            if sha256_file(saved["source_record"])!=saved["source_record_sha256"] or sha256_file(saved["checkpoint"])!=saved["checkpoint_sha256"]:raise RuntimeError("Existing historical control provenance changed")
            return
    previous=read(f"results/fullscale_experiments/records/yolov13s/seed-{seed}.json")
    rows=history(previous["results_csv"])[:6]
    if len(rows)!=6:raise RuntimeError("Missing historical6epochprefix")
    selected=max(rows,key=lambda x:float(x["metrics/mAP50-95(B)"]));epoch=int(float(selected["epoch"]))
    checkpoint=project_path(Path(previous["results_csv"]).parent/"weights"/f"epoch{epoch-1}.pt",must_exist=True)
    import yaml
    arguments=yaml.safe_load(project_path(Path(previous["results_csv"]).parent/"args.yaml",must_exist=True).read_text(encoding="utf-8"))
    # Ensure each historical control differs from audited seed42 only byseed
    # and nonfunctional runpaths, before consuming its validation checkpoint.
    audited42=read("results/fullscale_experiments/records/yolov13s/seed-42.json")
    reference=yaml.safe_load(project_path(Path(audited42["results_csv"]).parent/"args.yaml",must_exist=True).read_text(encoding="utf-8"))
    bad={k:(reference.get(k),arguments.get(k)) for k in set(arguments)|set(reference) if k not in{"name","project","save_dir","seed"} and reference.get(k)!=arguments.get(k)}
    if bad:raise RuntimeError(f"Historical control seed{seed} differs from audited functional config: {bad}")
    archive=project_path(f"results/optimization/epoch_history/original_sgd-seed{seed}-confirmation.csv");archive.parent.mkdir(parents=True,exist_ok=True)
    with archive.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    value={"status":"complete","model_key":"yolov13s","configuration":"original_sgd","seed":seed,"stage":"confirmation","training_origin":"historical_deterministic_prefix","fresh_training":False,"historical_reused_at_utc":utc_timestamp(),"maximum_completed_epochs":6,"epochs_completed":6,"source_full_trajectory_epochs":previous["epochs_completed"],"source_full_trajectory_actual_training_seconds":previous["actual_training_seconds"],"actual_training_seconds":float(rows[-1]["time"]),"incremental_training_seconds":0,"time_interpretation":"Historicaltraining time through epoch6, not newly consumedtrainingtime; fulltrajectory continued past6 independently","training_config":arguments,"source_training_started_at_utc":previous["started_at_utc"],"dataset_manifest_sha256":protocol["dataset_manifest_sha256"],"initial_weights_sha256":protocol["pretrained_weights_sha256"],"selected_epoch":epoch,"selected_elapsed_seconds":float(selected["time"]),"checkpoint":str(checkpoint.relative_to(PROJECT_ROOT)).replace("\\","/"),"checkpoint_sha256":sha256_file(checkpoint),"results_csv":str(archive.relative_to(PROJECT_ROOT)).replace("\\","/"),"results_csv_sha256":sha256_file(archive),"source_record":f"results/fullscale_experiments/records/yolov13s/seed-{seed}.json","source_record_sha256":sha256_file(f"results/fullscale_experiments/records/yolov13s/seed-{seed}.json"),"reuse_audit":"results/optimization/historical_control_reuse_audit.json","reuse_audit_sha256":sha256_file("results/optimization/historical_control_reuse_audit.json"),"validation":{"precision":float(selected["metrics/precision(B)"]),"recall":float(selected["metrics/recall(B)"]),"map50":float(selected["metrics/mAP50(B)"]),"map50_95":float(selected["metrics/mAP50-95(B)"])},"endpoint_validation_map50_95":float(rows[-1]["metrics/mAP50-95(B)"]),"endpoint_loss":{k:float(rows[-1][k]) for k in rows[-1] if "loss" in k},"selection":"HighestvalidationmAP95 in FIRST6historicalepochs ONLY; no120min checkpoint selection; testreevaluation only afterwinnerlock"}
    write_json(f"results/optimization/confirmation_records/original_sgd-seed{seed}.json",value)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--reuse-seed",type=int);args=parser.parse_args()
    if args.reuse_seed is None:perform_audit()
    else:reuse_control(args.reuse_seed)
