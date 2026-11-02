from __future__ import annotations

import csv
import json
from pathlib import Path

from pipeline_utils import configure_project_environment, project_path, require_local_venv, write_csv, write_json, utc_timestamp, sha256_file

configure_project_environment()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    require_local_venv()
    combined_path = project_path("results/benchmark_adv_tshirt_transfer.csv", must_exist=True)
    official_path = project_path("results/benchmark_yolov12_transfer.csv", must_exist=True)
    fields, combined = read_rows(combined_path)
    official_fields, official_rows = read_rows(official_path)
    if fields != official_fields:
        raise ValueError("Main and official YOLOv12 benchmark schemas differ.")
    expected = {"YOLOv12-N", "YOLOv12-S", "YOLOv12-L"}
    if {row.get("model_name") for row in official_rows} != expected or any(row.get("status") != "ok" for row in official_rows):
        raise RuntimeError("Official YOLOv12 benchmark must contain three successful N/S/L rows.")
    if any(row.get("dataset_scope", "").find("zero-shot") < 0 for row in official_rows):
        raise RuntimeError("YOLOv12 rows do not declare the zero-shot evaluation protocol.")
    failures = [row for row in combined if row.get("family") == "YOLOv12" and row.get("status") != "ok"]
    if failures:
        write_csv("results/yolov12_initial_loader_failures.csv", failures, fields)
    retained = [row for row in combined if row.get("family") != "YOLOv12"]
    merged = retained + official_rows
    expected_all = {f"YOLOv8-{s}" for s in "NSL"} | {f"YOLOv10-{s}" for s in "NSL"} | {f"YOLO11-{s}" for s in "NSL"} | expected | {f"YOLOv13-{s}" for s in "NSL"}
    names = [row.get("model_name") for row in merged]
    if set(names) != expected_all or len(names) != len(expected_all):
        raise RuntimeError(f"Combined benchmark incomplete or duplicated: {len(names)} rows.")
    merged.sort(key=lambda row: ("nsl".index(row["scale"]), ["YOLOv8", "YOLOv10", "YOLO11", "YOLOv12", "YOLOv13"].index(row["family"])))
    write_csv(combined_path, merged, fields)

    source_manifest_path = project_path("results/yolov12_v1_source_manifest.json", must_exist=True)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["local_compatibility_patch"] = {
        "file": "yolov12_v1_src/ultralytics/nn/modules/block.py",
        "sha256": sha256_file("yolov12_v1_src/ultralytics/nn/modules/block.py"),
        "description": "Optional FlashAttention import; PyTorch SDPA fallback for identical scaled-dot-product attention when FlashAttention is unavailable.",
    }
    write_json(source_manifest_path, source_manifest)

    manifest_path = project_path("results/external_transfer_benchmark_manifest.json", must_exist=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "successful_models": len(merged),
        "failed_models": [],
        "yolov12_implementation": "Official YOLOv12-v1.0 source commit b1035f8 with PyTorch SDPA fallback for unavailable optional FlashAttention; official matching v1.0 weights.",
        "yolov12_source_manifest": "results/yolov12_v1_source_manifest.json",
        "initial_loader_failures_preserved_as": "results/yolov12_initial_loader_failures.csv" if failures else None,
        "merged_at_utc": utc_timestamp(),
    })
    write_json(manifest_path, manifest)
    print(f"Merged {len(merged)} successful model rows; all YOLOv12 N/S/L rows use the matching official v1.0 implementation and weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
