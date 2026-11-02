"""Attach immutable interrupted-attempt provenance to a completed study summary."""
import json
from pipeline_utils import configure_project_environment, require_local_venv, project_path, sha256_file, utc_timestamp, write_json

configure_project_environment()

def main():
    require_local_venv()
    event_path="results/optimization/interruption_recovery_event.json"
    audit_path="results/optimization/interruption_replay_audit.json"
    summary_path="results/optimization/final_summary.json"
    def read(path):return json.loads(project_path(path,must_exist=True).read_text(encoding="utf-8"))
    final,event,audit=read(summary_path),read(event_path),read(audit_path)
    if final.get("status")!="complete":raise RuntimeError("Study must be complete")
    if audit.get("status")!="passed" or audit["compared_completed_epochs"]!=2 or audit["mismatches"]:raise RuntimeError("Replay audit did not pass")
    if event["partial_metrics_used_for_selection"] or event["test_seen"]:raise RuntimeError("Interrupted evidence improperly used")
    if sha256_file(event["archived_record"])!=event["archived_record_sha256"]:raise RuntimeError("Archived record changed")
    if sha256_file(audit["original_partial_csv"])!=event["partial_csv_sha256"]:raise RuntimeError("Interrupted CSV changed")
    final["interruption_recovery"]=event
    final["interruption_replay_audit"]=audit
    final["interruption_recovery_evidence"]={"event_path":event_path,"event_sha256":sha256_file(event_path),"replay_audit_path":audit_path,"replay_audit_sha256":sha256_file(audit_path),"attached_at_utc":utc_timestamp(),"partial_attempt_excluded_from_selection":True,"extra_attempts":1}
    write_json(summary_path,final)
    print(json.dumps(final["interruption_recovery_evidence"],indent=2))

if __name__=="__main__":main()
