"""Verify a single real one-image forward before corrected timing proceeds."""
import json
from pipeline_utils import configure_project_environment, require_local_venv, utc_timestamp, write_json, sha256_file
configure_project_environment()
require_local_venv()
from run_optimization_study import benchmark
result=benchmark("weights/yolov13s.pt",original80=True,warmup_frames=0,measured_frames=1)
if result["actual_prediction_batches"]!=1 or result["actual_batch_size"]!=1 or set(result["network_forward_batch_size_counts"])!={1}:raise RuntimeError("One-image method verification failed")
evidence={"status":"passed","verified_at_utc":utc_timestamp(),"verification_source_sha256":sha256_file(__file__),"measurement":result,"interpretation":"One prediction image, one actual prediction batch; forward pre-hook verifies every network input batch size1. Network-forward count additionally includes AutoBackend warmup."}
write_json("results/optimization/latency_batch1_verification.json",evidence)
print(json.dumps(evidence,indent=2))
