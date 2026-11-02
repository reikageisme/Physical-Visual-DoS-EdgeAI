"""Second-stage selection metric on native-selected checkpoint, no test access."""
import argparse,json
from pipeline_utils import configure_project_environment,require_local_venv,project_path,utc_timestamp,write_json
configure_project_environment()
def main():
    require_local_venv();p=argparse.ArgumentParser();p.add_argument("--record",required=True);a=p.parse_args();path=project_path(a.record,must_exist=True)
    record=json.loads(path.read_text(encoding="utf-8"))
    if record.get("status")!="complete":raise RuntimeError("Incomplete trainingrecord")
    if record.get("validation_fp32_measured_at_utc"):return
    from optimization_train_one import evaluate
    record["validation_fp32"]=evaluate(record["checkpoint"],run_name=f"selection-fp32-{record['stage']}-{record['configuration']}-seed{record['seed']}",batch=8,half=False)
    record["validation_fp32_measured_at_utc"]=utc_timestamp()
    record["selection_protocol"]={"within_trajectory":"maximum CSVnative FP16batch16 valmAP95 among6epochs","candidate_ranking":"ONLYnative-selected checkpoint revalidated FP32batch8; not exhaustive FP32epoch search","test_access":False}
    write_json(path,record)
if __name__=="__main__":main()
