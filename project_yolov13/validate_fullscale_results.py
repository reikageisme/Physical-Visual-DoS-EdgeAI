"""Check every completed seed against its actual epoch history and checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone

from pipeline_utils import configure_project_environment, project_path, require_local_venv

MODELS = ("yolo11n", "yolov12n", "yolov13n", "yolov13s")
SEEDS = (42, 100, 2026, 777, 999)
BUDGETS = (5, 10, 20, 30, 60, 120)


def ensure(condition, context):
    if not condition:
        raise RuntimeError(f"Full-scale evidence check failed: {context}")


def audit_results(*, require_complete=False):
    require_local_venv()
    configure_project_environment()
    root = project_path("results/fullscale_experiments")
    completed = available = 0
    pending, unavailable = [], []
    for model in MODELS:
        for seed in SEEDS:
            file = root / "records" / model / f"seed-{seed}.json"
            if not file.exists():
                pending.append({"model": model, "seed": seed, "status": "missing"})
                continue
            record = json.loads(file.read_text(encoding="utf-8"))
            if record.get("status") != "complete":
                pending.append({"model": model, "seed": seed, "status": record.get("status")})
                continue
            context = f"{model}, seed {seed}"
            ensure(record["model_key"] == model and record["seed"] == seed, context + " identity")
            ensure([r["budget_minutes"] for r in record["milestones"]] == list(BUDGETS), context + " budget list")
            with project_path(record["results_csv"], must_exist=True).open(encoding="utf-8-sig", newline="") as stream:
                history = list(csv.DictReader(stream))
            ensure(bool(history), context + " nonempty epoch history")
            ensure(record["epochs_completed"] == len(history), context + " completed epoch count")
            ensure(record["actual_training_seconds"] == float(history[-1]["time"]), context + " total training timer")
            ensure(record["training_config"]["seed"] == seed, context + " training seed")
            cached_tests = {}
            for milestone in record["milestones"]:
                budget = milestone["budget_minutes"]
                row_context = context + f", budget {budget}m"
                eligible = [r for r in history if float(r["time"]) <= budget * 60]
                if not eligible:
                    ensure(milestone["status"] == "no_completed_epoch_at_or_before_budget", row_context + " unavailable status")
                    unavailable.append({"model": model, "seed": seed, "budget_minutes": budget, "reason": milestone["status"]})
                    continue
                selected = max(eligible, key=lambda r: float(r["metrics/mAP50-95(B)"]))
                endpoint = max(eligible, key=lambda r: int(float(r["epoch"])))
                ensure(milestone["status"] == "complete", row_context + " status")
                ensure(milestone["selected_epoch"] == int(float(selected["epoch"])), row_context + " selected epoch")
                ensure(milestone["endpoint_epoch"] == int(float(endpoint["epoch"])), row_context + " endpoint epoch")
                ensure(milestone["selected_elapsed_seconds"] == float(selected["time"]), row_context + " selected time")
                ensure(milestone["endpoint_elapsed_seconds"] == float(endpoint["time"]), row_context + " endpoint time")
                project_path(milestone["checkpoint"], must_exist=True)
                for split in ("validation", "test"):
                    for key in ("precision", "recall", "map50", "map50_95"):
                        value = milestone[split][key]
                        ensure(math.isfinite(value) and 0 <= value <= 1, row_context + f" {split}/{key}")
                for metric_key, csv_key in (("precision", "metrics/precision(B)"), ("recall", "metrics/recall(B)"), ("map50", "metrics/mAP50(B)"), ("map50_95", "metrics/mAP50-95(B)")):
                    ensure(milestone["validation"][metric_key] == float(selected[csv_key]), row_context + f" selected validation {metric_key}")
                for loss_key, csv_key in (("train_box", "train/box_loss"), ("train_cls", "train/cls_loss"), ("train_dfl", "train/dfl_loss"), ("val_box", "val/box_loss"), ("val_cls", "val/cls_loss"), ("val_dfl", "val/dfl_loss")):
                    value = milestone["endpoint_loss"][loss_key]
                    ensure(math.isfinite(value) and value >= 0 and value == float(endpoint[csv_key]), row_context + f" endpoint loss {loss_key}")
                ensure(milestone["parameters_m"] > 0 and milestone["flops_g"] > 0, row_context + " model resource metrics")
                key = milestone["checkpoint"]
                if key in cached_tests:
                    ensure(cached_tests[key] == milestone["test"], row_context + " cached checkpoint test consistency")
                cached_tests[key] = milestone["test"]
                available += 1
            completed += 1
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if completed == len(MODELS) * len(SEEDS) else "in_progress",
        "verified_complete_records": completed,
        "verified_milestones": available,
        "unavailable_milestones": unavailable,
        "pending_records": pending,
        "checks": ["record identity and six budgets", "completed epoch count and total training time matching CSV", "configured training seed", "maximal eligible validation checkpoint", "all four selected validation metrics matching CSV", "last eligible endpoint", "exact elapsed times from CSV within budget", "checkpoint exists", "finite detection metrics within [0,1]", "six finite endpoint losses matching CSV", "positive parameters/FLOPs", "repeated checkpoint test consistency"],
    }
    (root / "metric_integrity_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if require_complete:
        ensure(not pending, "all 20 seed records must be complete")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_results(require_complete=args.require_complete), indent=2))


if __name__ == "__main__":
    main()
