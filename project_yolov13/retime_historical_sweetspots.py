"""Remeasure old validation-selected checkpoints; never reevaluate accuracy."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from pipeline_utils import PROJECT_ROOT,configure_project_environment,require_local_venv,project_path,sha256_file,utc_timestamp,write_json
configure_project_environment()
OUT=project_path("results/optimization/historical_latency_retimes.json")

def read(path):return json.loads(project_path(path,must_exist=True).read_text(encoding="utf-8"))

def worker(model,seed):
    from benchmark_fullscale_latency import MODELS
    source=MODELS[model]["source"]
    old=read("results/fullscale_experiments/latency_records.json")
    budget=old["sweet_spots"][model]["sweet_spot_minutes"]
    accuracy_path=f"results/fullscale_experiments/records/{model}/seed-{seed}.json"
    original=read(accuracy_path)
    selected=next(x for x in original["milestones"] if x["budget_minutes"]==budget and x.get("status")=="complete")
    checkpoint=selected["checkpoint"];project_path(checkpoint,must_exist=True)
    import optimization_train_one
    def source_specific_api():
        sys.path.insert(0,str(project_path(source,must_exist=True)))
        from ultralytics import YOLO
        import ultralytics
        if not Path(ultralytics.__file__).resolve().is_relative_to(project_path(source)):raise RuntimeError("Historical inference source mismatch")
        return YOLO
    # Reuse the identical timing function; its local API factory is replaced
    # before any Ultralytics import so YOLOv12 loads its original source tree.
    optimization_train_one.load_api=source_specific_api
    from run_optimization_study import benchmark
    timing=benchmark(checkpoint)
    value={"status":"complete","model_key":model,"configuration":"historical_validation_sweet_spot","seed":seed,"budget_minutes":budget,"checkpoint":checkpoint,"checkpoint_sha256":sha256_file(checkpoint),"accuracy_source_path":accuracy_path,"accuracy_source_sha256":sha256_file(accuracy_path),"model_source_directory":source,"retiming_worker_source_sha256":sha256_file(__file__),"latency":timing,"accuracy_reevaluated":False,"legacy_latency_excluded":True,"completed_at_utc":utc_timestamp()}
    write_json(f"results/optimization/historical_latency_records/{model}-seed{seed}.json",value)

def execute():
    models=("yolo11n","yolov12n","yolov13n","yolov13s");seeds=(42,100,2026,777,999)
    summary={"status":"running","protocol_id":"actual-batch1-txt-loader-v2","benchmark_source_sha256":sha256_file("run_optimization_study.py"),"started_at_utc":utc_timestamp(),"records":[],"exclusions":[],"legacy_latency_excluded":True,"accuracy_reevaluated":False,"planned_records":20}
    write_json(OUT,summary)
    for model in models:
        for seed in seeds:
            accuracy_path=f"results/fullscale_experiments/records/{model}/seed-{seed}.json"
            budget=read("results/fullscale_experiments/latency_records.json")["sweet_spots"][model]["sweet_spot_minutes"]
            job={"model_key":model,"configuration":"historical_validation_sweet_spot","seed":seed,"budget_minutes":budget,"accuracy_source_path":accuracy_path}
            try:
                old_record=read(accuracy_path);job["accuracy_source_sha256"]=sha256_file(accuracy_path)
                job["checkpoint"]=next(x["checkpoint"] for x in old_record["milestones"] if x["budget_minutes"]==budget and x.get("status")=="complete")
            except Exception as exc:
                summary["exclusions"].append({**job,"status":"excluded","reason":repr(exc)});write_json(OUT,summary);continue
            record_path=project_path(f"results/optimization/historical_latency_records/{model}-seed{seed}.json")
            if not record_path.exists():
                log=project_path(f"results/optimization/logs/retime-{model}-seed{seed}.log");log.parent.mkdir(parents=True,exist_ok=True)
                with log.open("w",encoding="utf-8") as stream:
                    result=subprocess.run([sys.executable,__file__,"--model",model,"--seed",str(seed)],cwd=PROJECT_ROOT,env={**os.environ,"PYTHONUTF8":"1"},stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:
                    text=log.read_text(encoding="utf-8",errors="replace")
                    summary["exclusions"].append({**job,"status":"excluded","reason":text[-4000:],"exit_code":result.returncode,"log":str(log.relative_to(PROJECT_ROOT)).replace("\\","/")})
                    write_json(OUT,summary);continue
            value=read(record_path)
            if value["latency"]["protocol_id"]!="actual-batch1-txt-loader-v2":raise RuntimeError("Historical retime protocol mismatch")
            summary["records"].append(value);write_json(OUT,summary)
            print(f"retimed {model} seed{seed}: {value['latency']['latency_median_ms_per_frame']:.3f}ms",flush=True)
    summary.update(status="complete" if len(summary["records"])==20 else "complete_with_exclusions",completed_at_utc=utc_timestamp());write_json(OUT,summary)

if __name__=="__main__":
    require_local_venv();p=argparse.ArgumentParser();p.add_argument("--model");p.add_argument("--seed",type=int);a=p.parse_args()
    if a.model:worker(a.model,a.seed)
    else:execute()
