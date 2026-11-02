"""Sequentially orchestrate all model/seed trajectories inside the project."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, project_path, require_local_venv

configure_project_environment()

MODEL_ORDER = ("yolo11n", "yolov12n", "yolov13n", "yolov13s")
SEED_ORDER = (42, 100, 2026, 777, 999)
RUNNER = project_path("fullscale_train_one.py", must_exist=True)
STATE_PATH = project_path("results/fullscale_experiments/orchestrator_state.json")


def parse_list(value: str, available: tuple, cast):
    if value == "all":
        return list(available)
    selected = [cast(item.strip()) for item in value.split(",") if item.strip()]
    invalid = [item for item in selected if item not in available]
    if invalid:
        raise ValueError(f"Unsupported values: {invalid}; allowed: {list(available)}")
    return selected


def write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    require_local_venv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all", help="comma-separated model keys or all")
    parser.add_argument("--seeds", default="all", help="comma-separated seeds or all")
    parser.add_argument("--cap-minutes", type=int, default=120)
    args = parser.parse_args()
    models = parse_list(args.models, MODEL_ORDER, str)
    seeds = parse_list(args.seeds, SEED_ORDER, int)
    jobs = [{"model": model, "seed": seed, "status": "pending"} for model in models for seed in seeds]
    state = {
        "study_id": "fullscale-20261011",
        "status": "running",
        "models": models,
        "seeds": seeds,
        "cap_minutes": args.cap_minutes,
        "jobs": jobs,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_state(state)

    for job in jobs:
        model, seed = job["model"], job["seed"]
        print(f"\n### Starting {model}, seed={seed}, cap={args.cap_minutes}m ###", flush=True)
        job["status"] = "running"
        job["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_state(state)
        command = [sys.executable, str(RUNNER), "--model", model, "--seed", str(seed), "--cap-minutes", str(args.cap_minutes)]
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, check=False)
        job["return_code"] = result.returncode
        job["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        job["status"] = "complete" if result.returncode == 0 else "failed"
        write_state(state)
        if result.returncode != 0:
            state["status"] = "failed"
            state["failed_job"] = {"model": model, "seed": seed, "return_code": result.returncode}
            write_state(state)
            return result.returncode

    state["status"] = "complete"
    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_state(state)
    print(f"All {len(jobs)} trajectories completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
