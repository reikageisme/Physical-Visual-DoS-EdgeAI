"""Audit only the task dataset configured inside project_yolov13."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, load_yaml, project_path, require_local_venv, utc_timestamp, write_json

configure_project_environment()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PLACEHOLDERS = {"object", "objects", "class", "classes", "item", "thing", "unknown"}


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    require_local_venv()
    yaml_path = project_path("dataset/data.yaml", must_exist=True)
    cfg = load_yaml(yaml_path)
    raw_names = cfg.get("names", {})
    if isinstance(raw_names, list):
        names = {i: str(name) for i, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(i): str(name) for i, name in raw_names.items()}
    else:
        names = {}
    nc = int(cfg.get("nc", len(names))) if cfg.get("nc", len(names)) is not None else len(names)
    names_consistent = bool(names) and nc == len(names) and set(names) == set(range(nc))
    placeholders = [name for name in names.values() if name.strip().lower() in PLACEHOLDERS]

    data_root_value = Path(str(cfg.get("path", "./dataset")))
    data_root = data_root_value if data_root_value.is_absolute() else PROJECT_ROOT / data_root_value
    data_root = project_path(data_root)
    split_rows: dict[str, dict] = {}
    total_images = total_labels = total_missing_labels = total_invalid_rows = 0
    for split in ("train", "val", "test"):
        image_value = Path(str(cfg.get(split, f"images/{split}")))
        image_dir = image_value if image_value.is_absolute() else data_root / image_value
        image_dir = project_path(image_dir)
        label_dir = project_path(data_root / "labels" / split)
        images = image_files(image_dir)
        label_files = sorted(label_dir.rglob("*.txt")) if label_dir.is_dir() else []
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in label_files}
        missing = sorted(image_stems - label_stems)
        orphan = sorted(label_stems - image_stems)
        class_counts: Counter[str] = Counter()
        invalid_rows: list[dict] = []
        for label in label_files:
            try:
                lines = label.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as exc:
                invalid_rows.append({"file": str(label.relative_to(PROJECT_ROOT)), "line": None, "reason": f"read error: {exc}"})
                continue
            for line_no, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    invalid_rows.append({"file": str(label.relative_to(PROJECT_ROOT)), "line": line_no, "reason": f"expected 5 YOLO detection fields, got {len(parts)}"})
                    continue
                try:
                    class_value = float(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    invalid_rows.append({"file": str(label.relative_to(PROJECT_ROOT)), "line": line_no, "reason": "non-numeric class or bbox value"})
                    continue
                class_id = int(class_value)
                if class_value != class_id or class_id < 0 or class_id >= nc:
                    invalid_rows.append({"file": str(label.relative_to(PROJECT_ROOT)), "line": line_no, "reason": f"class id {parts[0]} outside [0, {nc - 1}]"})
                    continue
                x, y, width, height = coords
                if not all(0.0 <= value <= 1.0 for value in coords) or width <= 0 or height <= 0 or x - width / 2 < 0 or y - height / 2 < 0 or x + width / 2 > 1 or y + height / 2 > 1:
                    invalid_rows.append({"file": str(label.relative_to(PROJECT_ROOT)), "line": line_no, "reason": "bbox must be normalized, positive-sized, and contained in image bounds"})
                    continue
                class_counts[str(class_id)] += 1
        split_rows[split] = {
            "image_dir": str(image_dir.relative_to(PROJECT_ROOT)),
            "label_dir": str(label_dir.relative_to(PROJECT_ROOT)),
            "images": len(images),
            "label_files": len(label_files),
            "missing_label_stems": missing,
            "orphan_label_stems": orphan,
            "invalid_bbox_rows": len(invalid_rows),
            "invalid_bbox_examples": invalid_rows[:20],
            "instances_by_class_id": dict(class_counts),
        }
        total_images += len(images)
        total_labels += len(label_files)
        total_missing_labels += len(missing)
        total_invalid_rows += len(invalid_rows)

    unassigned_dir = project_path("dataset")
    unassigned_images = sorted(path for path in unassigned_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES) if unassigned_dir.is_dir() else []
    unassigned_labels = sorted(path for path in unassigned_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt") if unassigned_dir.is_dir() else []
    candidate = {
        "directory": "dataset/",
        "raster_images": len(unassigned_images),
        "annotation_txt_files": len(unassigned_labels),
        "other_files": sum(1 for path in unassigned_dir.iterdir() if path.is_file() and path.suffix.lower() not in IMAGE_SUFFIXES and path.suffix.lower() != ".txt") if unassigned_dir.is_dir() else 0,
        "eligible_as_task_split": False,
        "reason": "No train/val/test assignment or YOLO label files are present in this folder.",
    }

    status = "ready" if total_images > 0 and total_missing_labels == 0 and total_invalid_rows == 0 and names_consistent and not placeholders else "not_ready"
    result = {
        "audit_time_utc": utc_timestamp(),
        "project_root": str(PROJECT_ROOT),
        "dataset_yaml": str(yaml_path.relative_to(PROJECT_ROOT)),
        "yaml_path_setting": str(cfg.get("path", "data")),
        "classes": {"nc": nc, "names": names, "names_nc_consistent": names_consistent, "placeholder_names": placeholders},
        "splits": split_rows,
        "totals": {"images": total_images, "label_files": total_labels, "missing_labels": total_missing_labels, "invalid_bbox_rows": total_invalid_rows},
        "unassigned_candidate_folder": candidate,
        "preflight_status": status,
        "training_started": False,
        "decision": "Training must not start unless the task dataset is present in the configured splits with valid labels and a verified class map.",
    }
    write_json("results/dataset_audit_current.json", result)

    cumulative_path = project_path("results/dataset_audit.json", must_exist=True)
    cumulative = json.loads(cumulative_path.read_text(encoding="utf-8"))
    cumulative["latest_task_data_audit"] = result
    cumulative["audit_time_utc"] = result["audit_time_utc"]
    cumulative["splits"] = {name: {"image_count": row["images"], "label_txt_count": row["label_files"]} for name, row in split_rows.items()}
    cumulative["candidate_assets"] = {
        "folder": "dataset/",
        "raster_image_count": candidate["raster_images"],
        "svg_count": sum(path.suffix.lower() == ".svg" for path in unassigned_dir.iterdir() if path.is_file()) if unassigned_dir.is_dir() else 0,
        "annotation_file_count": candidate["annotation_txt_files"],
        "split_assignment": "none",
        "eligible_for_detection_training": False,
        "reason": candidate["reason"],
    }
    write_json(cumulative_path, cumulative)
    print(json.dumps({"preflight_status": status, "splits": split_rows, "classes": result["classes"], "candidate": candidate}, indent=2))
    return 0 if status == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
