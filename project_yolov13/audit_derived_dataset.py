"""Audit a project-local derived YOLO dataset YAML, including per-split labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, load_yaml, project_path, require_local_venv, utc_timestamp, write_json

configure_project_environment()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset YAML located inside project_yolov13.")
    parser.add_argument("--output", required=True, help="Audit JSON destination inside project_yolov13.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_local_venv()
    yaml_path = project_path(args.data, must_exist=True)
    cfg = load_yaml(yaml_path)
    nc = int(cfg.get("nc", len(cfg.get("names", []))))
    raw_names = cfg.get("names", {})
    names = {int(k): str(v) for k, v in raw_names.items()} if isinstance(raw_names, dict) else {i: str(v) for i, v in enumerate(raw_names)}
    names_valid = nc == len(names) and set(names) == set(range(nc)) and all(name.strip() for name in names.values())
    root = Path(str(cfg.get("path", ".")))
    root = root if root.is_absolute() else PROJECT_ROOT / root
    root = project_path(root, must_exist=True)
    split_results = {}
    failures = []
    for split in ("train", "val", "test"):
        if split not in cfg:
            failures.append(f"missing split key: {split}")
            continue
        image_dir = Path(str(cfg[split]))
        image_dir = image_dir if image_dir.is_absolute() else root / image_dir
        image_dir = project_path(image_dir, must_exist=True)
        label_dir = project_path(root / "labels" / split, must_exist=True)
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.rglob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        missing = sorted(image_stems - label_stems)
        orphan = sorted(label_stems - image_stems)
        class_counts: Counter[int] = Counter()
        invalid = []
        for label in labels:
            try:
                lines = label.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as exc:
                invalid.append({"file": str(label.relative_to(PROJECT_ROOT)), "error": str(exc)})
                continue
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                fields = line.split()
                try:
                    if len(fields) != 5:
                        raise ValueError(f"expected five fields, got {len(fields)}")
                    class_value = float(fields[0])
                    coords = [float(value) for value in fields[1:]]
                    class_id = int(class_value)
                    if class_value != class_id or not 0 <= class_id < nc:
                        raise ValueError(f"class id {fields[0]} outside [0,{nc - 1}]")
                    x, y, width, height = coords
                    if not all(0.0 <= value <= 1.0 for value in coords) or width <= 0 or height <= 0:
                        raise ValueError("xywh must be normalized with positive width and height")
                    if x - width / 2 < -1e-7 or y - height / 2 < -1e-7 or x + width / 2 > 1 + 1e-7 or y + height / 2 > 1 + 1e-7:
                        raise ValueError("bbox extends outside image bounds")
                    class_counts[class_id] += 1
                except (ValueError, OverflowError) as exc:
                    invalid.append({"file": str(label.relative_to(PROJECT_ROOT)), "line": line_number, "error": str(exc)})
        if not images:
            failures.append(f"empty {split} image split")
        if missing:
            failures.append(f"{split} images missing labels: {len(missing)}")
        if orphan:
            failures.append(f"{split} orphan labels: {len(orphan)}")
        if invalid:
            failures.append(f"{split} malformed label rows/files: {len(invalid)}")
        if len(images) != len(labels):
            failures.append(f"{split} image/label file counts differ: {len(images)} vs {len(labels)}")
        split_results[split] = {
            "image_dir": str(image_dir.relative_to(PROJECT_ROOT)),
            "label_dir": str(label_dir.relative_to(PROJECT_ROOT)),
            "images": len(images),
            "label_files": len(labels),
            "missing_label_stems": missing,
            "orphan_label_stems": orphan,
            "invalid_label_rows": len(invalid),
            "invalid_examples": invalid[:10],
            "instances_by_class": dict(class_counts),
        }
    if not names_valid:
        failures.append("class names/nc mapping is invalid")
    result = {
        "audit_time_utc": utc_timestamp(),
        "dataset_yaml": str(yaml_path.relative_to(PROJECT_ROOT)),
        "dataset_root": str(root.relative_to(PROJECT_ROOT)),
        "nc": nc,
        "names": names,
        "names_valid": names_valid,
        "splits": split_results,
        "status": "ready" if not failures else "not_ready",
        "failures": failures,
    }
    write_json(args.output, result)
    print(result["status"], result["splits"], "failures=", failures)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
