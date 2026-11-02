"""Audit the completed optimization evidence without running models or tests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone

from pipeline_utils import configure_project_environment, project_path, require_local_venv

configure_project_environment()


def read(path):
    return json.loads(project_path(path, must_exist=True).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(project_path(path, must_exist=True).read_bytes()).hexdigest()


def close(actual, expected, context):
    assert math.isfinite(float(actual)), context
    assert math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-9), context


def summarize(values):
    from scipy.stats import t
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = float(t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return dict(n=len(values), mean=mean, sd=sd, ci95_low=mean-half, ci95_high=mean+half)


LATENCY_PROTOCOL = "actual-batch1-txt-loader-v2"


def percentile(values, q):
    ordered = sorted(float(x) for x in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def audit_latency(latency, lock_time, expected_frames=500, expected_warmup=32):
    assert latency["protocol_id"] == LATENCY_PROTOCOL
    assert latency["imgsz"] == 416 and latency["batch"] == 1
    assert latency["actual_batch_size"] == 1
    assert latency["warmup_frames"] == expected_warmup
    assert latency["measured_frames"] == expected_frames
    assert latency["actual_prediction_batches"] == expected_warmup + expected_frames
    assert latency["timing_precision"] == "fp32"
    assert latency["benchmark_source_sha256"] == digest("run_optimization_study.py")
    assert latency["network_forward_count_includes_backend_warmup"] is True
    forward_counts = latency["network_forward_batch_size_counts"]
    assert set(map(str, forward_counts)) == {"1"}
    assert sum(int(value) for value in forward_counts.values()) >= expected_warmup + expected_frames
    loader = latency["loader"]
    assert loader["class_name"] == "LoadImagesAndVideos" and loader["actual_batch_size"] == 1
    assert loader["source_type"] == "project-local text file"
    assert digest(loader["source_file"]) == loader["source_file_sha256"]
    assert len(project_path(loader["source_file"]).read_text(encoding="utf-8").splitlines()) == expected_warmup + expected_frames
    assert "random.Random(2026)" in loader["sample_order"]
    totals = latency["per_frame_total_ms"]
    assert len(totals) == expected_frames and all(math.isfinite(float(x)) and float(x) > 0 for x in totals)
    assert latency["unique_measured_total_times"] == len(set(totals)) > 1
    close(latency["latency_median_ms_per_frame"], statistics.median(totals), "latency median")
    close(latency["latency_p95_ms_per_frame"], percentile(totals, 0.95), "latency P95")
    close(latency["fps_from_median_latency"], 1000 / statistics.median(totals), "FPS reciprocal")
    assert latency["timing_scope"] == "Ultralytics preprocessing + inference + postprocessing; excludes decode/I/O"
    assert datetime.fromisoformat(latency["measured_at_utc"]) >= lock_time


def main():
    require_local_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    path = project_path("results/optimization/final_summary.json")
    if not path.exists():
        if args.require_complete:
            raise RuntimeError("Optimization final summary has not been generated")
        print("Optimization evidence is still running; no completion assertion")
        return
    final = read(path)
    assert final["status"] == "complete"
    protocol = final["protocol"]
    seeds = protocol["confirmation_seeds"]
    assert len(seeds) >= 3 and len(set(seeds)) == len(seeds)
    assert protocol["screening_seed"] not in seeds
    assert digest("dataset/fullscale_person/manifest.json") == protocol["dataset_manifest_sha256"]
    assert digest("weights/yolov13s.pt") == protocol["pretrained_weights_sha256"]
    correction = read("results/optimization/latency_protocol_correction.json")
    assert final["latency_protocol_correction"] == correction
    assert correction["status"] == "frozen_before_corrected_full_measurements"
    assert correction["protocol_id"] == LATENCY_PROTOCOL
    assert correction["historical_latency_valid"] is False
    assert correction["historical_fps_valid"] is False
    assert correction["historical_accuracy_affected"] is False
    assert correction["legacy_timing_excluded_from_comparisons"] is True
    assert correction["corrected_accuracy_protocol_unchanged"] is True
    assert correction["selection_or_checkpoints_changed"] is False
    assert correction["actual_performance_measurements_to_repeat"] == 7
    assert digest(correction["historical_latency_source"]) == correction["historical_latency_source_sha256"]
    invalid_native_path = correction["invalid_native_diagnostic_archive"]
    assert digest(invalid_native_path) == correction["invalid_native_diagnostic_archive_sha256"]
    invalid_native = read(invalid_native_path)
    assert invalid_native["accuracy_valid"] is True and invalid_native["latency_valid"] is False
    assert invalid_native["status"] == "accuracy_complete_latency_invalid"
    assert correction["native_test_accuracy_preserved"] is True
    verification_path = correction["method_verification"]
    assert digest(verification_path) == correction["method_verification_sha256"]
    method_proof = read(verification_path)
    assert method_proof["status"] == "passed"
    proof = method_proof["measurement"]
    assert proof["protocol_id"] == LATENCY_PROTOCOL
    assert proof["actual_prediction_batches"] == 1 and proof["actual_batch_size"] == 1
    assert set(map(str, proof["network_forward_batch_size_counts"])) == {"1"}
    assert proof["loader"]["class_name"] == "LoadImagesAndVideos"
    assert proof["loader"]["actual_batch_size"] == 1
    assert digest(proof["loader"]["source_file"]) == proof["loader"]["source_file_sha256"]
    assert proof["measured_frames"] == 1 and proof["warmup_frames"] == 0
    assert correction["probe_performance_excluded"] is True
    assert digest("run_optimization_study.py") == proof["benchmark_source_sha256"]
    assert digest("verify_optimization_latency_loader.py") == method_proof["verification_source_sha256"]
    oom_log = project_path(correction["oom_evidence"]["source_log"], must_exist=True).read_text(encoding="utf-8", errors="replace")
    assert "CUDA out of memory. Tried to allocate 2.74 GiB" in oom_log
    assert correction["oom_evidence"]["interrupted_record"] == "original_sgd-seed100"
    assert "not a completed record evaluation" in correction["oom_evidence"]["test_status"].lower()
    if protocol.get("historical_control_amendment"):
        reuse = protocol["historical_control_amendment"]
        assert digest(reuse["audit"]) == reuse["audit_sha256"]
    recovery_event = None
    recovery_evidence = final.get("interruption_recovery_evidence")
    if recovery_evidence:
        recovery_event = read(recovery_evidence["event_path"])
        assert digest(recovery_evidence["event_path"]) == recovery_evidence["event_sha256"]
        assert recovery_event["status"] == "frozen_before_replacement_training"
        assert recovery_event["partial_metrics_used_for_selection"] is False
        assert recovery_event["test_seen"] is False
        assert recovery_event["candidate_count_change"] == 0
        assert digest(recovery_event["archived_record"]) == recovery_event["archived_record_sha256"]
        replay = read(recovery_evidence["replay_audit_path"])
        assert digest(recovery_evidence["replay_audit_path"]) == recovery_evidence["replay_audit_sha256"]
        assert replay["status"] == "passed" and replay["compared_completed_epochs"] == 2 and not replay["mismatches"]
        assert replay["original_partial_csv_sha256"] == recovery_event["partial_csv_sha256"]
    if protocol.get("adaptive_sixth_candidate_decision"):
        adaptive = protocol["adaptive_sixth_candidate_decision"]
        assert adaptive["test_seen"] is False
        assert digest(adaptive["diagnostic"]) == adaptive["diagnostic_sha256"]
        close(adaptive["activation_gap"], adaptive["zero_fp32_map95"] - adaptive["trained_fp32_map95"], "adaptive validation gap")
        assert adaptive["activated"] == (adaptive["activation_gap"] > 0.005)
        assert adaptive["freeze_bn_for_extra_candidate"] == (adaptive["bn_buffer_reset_fp32_gain"] >= 0.005)
    winner = final["winner_configuration"]
    screening = final["screening_records"]
    assert {r["configuration"] for r in screening} == set(protocol["candidate_configurations"])
    eligible = [r for r in screening if r["configuration"] != "original_sgd"]
    if protocol.get("selection_protocol_amendment"):
        for record in eligible:
            assert "validation_fp32" in record, "Deployment-protocol validation is missing"
            for key in ("precision", "recall", "map50", "map50_95"):
                assert math.isfinite(record["validation_fp32"][key])
                assert 0 <= record["validation_fp32"][key] <= 1
        ranking_field = "validation_fp32"
    else:
        ranking_field = "validation"
    assert max(eligible, key=lambda r: r[ranking_field]["map50_95"])["configuration"] == winner
    assert final["winner_lock"]["winner_configuration"] == winner
    lock_time = datetime.fromisoformat(final["winner_lock"]["locked_at_utc"])
    decision = final["deployment_choice"]
    assert decision["test_seen_before_decision"] is False
    assert decision["n"] == 1 and decision["seed42_selection_only"] is True
    prefer_zero = decision["zero_validation_fp32"]["map50_95"] >= decision["trained_validation_fp32"]["map50_95"]
    assert decision["choice"] == ("zero_restored_person" if prefer_zero else "trained_winner")
    assert digest(decision["checkpoint"]) == decision["checkpoint_sha256"]
    if prefer_zero:
        zero = final["zero_locked_test_diagnostic"]
        assert zero["status"] == "complete" and zero["n"] == 1
        assert zero["checkpoint_sha256"] == decision["checkpoint_sha256"]
        assert datetime.fromisoformat(zero["measured_at_utc"]) >= datetime.fromisoformat(decision["locked_at_utc"])
        assert zero["head_classes"] == 1 and zero["prediction_classes"] == [0]
        audit_latency(zero["latency"], lock_time)
        assert zero["latency"]["checkpoint_sha256"] == zero["checkpoint_sha256"]
    native = final["pretrained_locked_test_diagnostic"]
    assert native["status"] == "complete" and native["n"] == 1
    assert native["head_classes"] == 80 and native["prediction_classes"] == [0]
    assert datetime.fromisoformat(native["measured_at_utc"]) >= lock_time
    assert native.get("invalid_previous_latency_archive") == invalid_native_path
    assert native["test"] == invalid_native["test"], "Native80 accuracy changed during latency repair"
    assert digest("weights/yolov13s.pt") == native["latency"]["checkpoint_sha256"]
    audit_latency(native["latency"], lock_time)
    records = final["confirmation_records"]
    assert {(r["configuration"], r["seed"]) for r in records} == {
        (config, seed) for config in ("original_sgd", winner) for seed in seeds
    }
    metric_columns = {"precision": "metrics/precision(B)", "recall": "metrics/recall(B)",
                      "map50": "metrics/mAP50(B)", "map50_95": "metrics/mAP50-95(B)"}
    verified = []
    for r in screening + records:
        assert r["status"] == "complete"
        if r["configuration"] == "transfer_tiny_lr":
            assert datetime.fromisoformat(r["started_at_utc"]) >= datetime.fromisoformat(protocol["adaptive_sixth_candidate_decision"]["decided_at_utc"])
        assert digest(r["results_csv"]) == r["results_csv_sha256"]
        assert digest(r["checkpoint"]) == r["checkpoint_sha256"]
        with project_path(r["results_csv"]).open(encoding="utf-8-sig", newline="") as f:
            history = list(csv.DictReader(f))
        rows = [x for x in history if int(float(x["epoch"])) <= protocol["epochs_per_run"]]
        assert len(rows) == protocol["epochs_per_run"], r["configuration"]
        selected = max(rows, key=lambda x: float(x["metrics/mAP50-95(B)"]))
        assert int(float(selected["epoch"])) == r["selected_epoch"]
        for key, column in metric_columns.items():
            close(r["validation"][key], selected[column], (r["configuration"], key))
        close(r["selected_elapsed_seconds"], selected["time"], "selected elapsed time")
        if r in records:
            for key in metric_columns:
                assert math.isfinite(r["test"][key]) and 0 <= r["test"][key] <= 1
            latency = r["latency"]
            audit_latency(latency, lock_time)
            assert latency["checkpoint_sha256"] == r["checkpoint_sha256"]
            assert datetime.fromisoformat(r["test_evaluated_at_utc"]) >= lock_time
        verified.append(dict(configuration=r["configuration"], seed=r["seed"], selected_epoch=r["selected_epoch"]))
    for group in final["groups"]:
        members = [r for r in records if r["configuration"] == group["configuration"]]
        assert len(members) == len(seeds)
        assert {r["seed"] for r in group["seed_records"]} == set(seeds)
        for key in ("precision", "recall", "map50", "map50_95", "parameters_m", "flops_g", "latency_ms", "fps"):
            if key == "latency_ms":
                values = [r["latency"]["latency_median_ms_per_frame"] for r in members]
            elif key == "fps":
                values = [r["latency"]["fps_from_median_latency"] for r in members]
            else:
                values = [r["test"][key] for r in members]
            for stat, expected in summarize(values).items():
                close(group["metrics"][key][stat], expected, (group["configuration"], key, stat))
        for key in metric_columns:
            values = [r[ranking_field][key] for r in members]
            for stat, expected in summarize(values).items():
                close(group["validation_metrics"][key][stat], expected,
                      (group["configuration"], "validation", ranking_field, key, stat))
        for split in ("domain_validation", "domain_test"):
            for domain in ("coco", "advtshirt"):
                summarized = group["domain_metrics"][split][domain]
                for count in ("images", "person_instances", "negative_images"):
                    assert all(r[split][domain][count] == summarized[count] for r in members)
                for key in metric_columns:
                    values = [r[split][domain]["metrics"][key] for r in members]
                    for stat, expected in summarize(values).items():
                        close(summarized["metrics"][key][stat], expected,
                              (group["configuration"], split, domain, key, stat))
    deltas = []
    for seed in seeds:
        control = next(r for r in records if r["seed"] == seed and r["configuration"] == "original_sgd")
        optimized = next(r for r in records if r["seed"] == seed and r["configuration"] == winner)
        deltas.append(optimized["test"]["map50_95"]-control["test"]["map50_95"])
    for key, expected in summarize(deltas).items():
        close(final["paired_test_map95_delta"][key], expected, ("paired change", key))
    historical_retimes = None
    cross_cohort_runtime_check = None
    historical_path = project_path("results/optimization/historical_latency_retimes.json")
    if historical_path.exists():
        historical_retimes = read("results/optimization/historical_latency_retimes.json")
        assert historical_retimes["status"] in ("complete", "complete_with_exclusions")
        assert historical_retimes["protocol_id"] == LATENCY_PROTOCOL
        assert historical_retimes["benchmark_source_sha256"] == digest("run_optimization_study.py")
        assert historical_retimes["legacy_latency_excluded"] is True
        assert historical_retimes["accuracy_reevaluated"] is False
        assert len(historical_retimes["records"]) + len(historical_retimes["exclusions"]) == historical_retimes["planned_records"]
        assert final["historical_latency_retiming_evidence"]["sha256"] == digest("results/optimization/historical_latency_retimes.json")
        assert final["historical_latency_retiming_evidence"]["successful_records"] == len(historical_retimes["records"])
        assert final["historical_latency_retiming_evidence"]["excluded_records"] == len(historical_retimes["exclusions"])
        assert final["historical_latency_retiming_evidence"]["accuracy_reevaluated"] is False
        retimed_keys = set()
        for record in historical_retimes["records"]:
            assert record["status"] == "complete"
            assert record["configuration"] == "historical_validation_sweet_spot"
            assert record["legacy_latency_excluded"] is True and record["accuracy_reevaluated"] is False
            assert record["accuracy_source_path"] == f"results/fullscale_experiments/records/{record['model_key']}/seed-{record['seed']}.json"
            assert digest(record["accuracy_source_path"]) == record["accuracy_source_sha256"]
            assert digest(record["checkpoint"]) == record["checkpoint_sha256"]
            accuracy = read(record["accuracy_source_path"])
            milestone = next(item for item in accuracy["milestones"]
                             if item["budget_minutes"] == record["budget_minutes"] and item.get("status") == "complete")
            assert milestone["checkpoint"] == record["checkpoint"]
            assert record["retiming_worker_source_sha256"] == digest("retime_historical_sweetspots.py")
            audit_latency(record["latency"], lock_time)
            assert record["latency"]["checkpoint_sha256"] == record["checkpoint_sha256"]
            assert datetime.fromisoformat(record["completed_at_utc"]) >= lock_time
            key = (record["model_key"], record["seed"])
            assert key not in retimed_keys
            retimed_keys.add(key)
        if historical_retimes["status"] == "complete":
            expected = {(model, seed) for model in ("yolo11n", "yolov12n", "yolov13n", "yolov13s")
                        for seed in (42, 100, 2026, 777, 999)}
            assert retimed_keys == expected
            repeated_historical = next(r for r in historical_retimes["records"]
                                       if r["model_key"] == "yolov13s" and r["seed"] == 100)
            matched_control = next(r for r in records
                                   if r["configuration"] == "original_sgd" and r["seed"] == 100)
            historical_latency = repeated_historical["latency"]
            matched_latency = matched_control["latency"]
            assert repeated_historical["checkpoint_sha256"] == matched_control["checkpoint_sha256"]
            assert historical_latency["benchmark_source_sha256"] == matched_latency["benchmark_source_sha256"]
            assert historical_latency["loader"]["source_file_sha256"] == matched_latency["loader"]["source_file_sha256"]
            delta_ms = historical_latency["latency_median_ms_per_frame"] - matched_latency["latency_median_ms_per_frame"]
            cross_cohort_runtime_check = {
                "status": "material_cross_cohort_variation_observed",
                "model_key": "yolov13s", "configuration": "original_sgd", "seed": 100,
                "checkpoint_sha256": matched_control["checkpoint_sha256"],
                "benchmark_source_sha256": matched_latency["benchmark_source_sha256"],
                "sample_source_sha256": matched_latency["loader"]["source_file_sha256"],
                "matched_cohort_latency_ms": matched_latency["latency_median_ms_per_frame"],
                "standalone_retime_latency_ms": historical_latency["latency_median_ms_per_frame"],
                "absolute_difference_ms": delta_ms,
                "relative_difference_percent_of_matched": 100 * delta_ms / matched_latency["latency_median_ms_per_frame"],
                "interpretation": "Identical checkpoint and benchmark inputs varied across timing cohorts; prior historical retimings are excluded from direct local latency/FPS comparisons. This does not affect recorded accuracy.",
            }
    output = dict(status="passed", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  final_summary_sha256=digest("results/optimization/final_summary.json"),
                  screening_records=len(screening), confirmation_records=len(records),
                  winner=winner, confirmation_seeds=seeds, verified_records=verified,
                  historical_latency_records_checked=len(historical_retimes["records"]) if historical_retimes else 0,
                  cross_cohort_runtime_check=cross_cohort_runtime_check,
                  selection_ranking_field=ranking_field,
                  checks=["input and checkpoint hashes", "six eligible epochs and maximal native validation checkpoint",
                          "all selected validation metrics matching CSV", "validation-only winner selection",
                          "disjoint screening and confirmation seeds", "locked test timing", "matched latency protocol",
                          "mean sample SD and Student t 95 percent CI", "domain-specific sample counts and statistics",
                          "native person-only checkpoint diagnostic after lock", "validation-only deployment choice and single-checkpoint evidence",
                          "paired seedwise test change",
                          "historical checkpoint selection and corrected latency retiming" if historical_retimes else "no historical latency retiming artifact",
                          "same-checkpoint cross-cohort runtime variability check" if cross_cohort_runtime_check else "no cross-cohort timing comparison",
                          "interrupted-run archival and deterministic prefix replay" if recovery_event else "no interruption recovery event"],
                  limitation="This audit checks recorded evidence and arithmetic; it does not establish causal isolation or fresh test independence.")
    destination = project_path("results/updated_research/metric_integrity_audit.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("status", "screening_records", "confirmation_records", "winner")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
