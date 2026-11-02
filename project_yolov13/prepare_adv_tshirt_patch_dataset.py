"""Build a deterministic, patch-only YOLO dataset from audited AdvT-shirt-1K labels."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, project_path, require_local_venv, utc_timestamp, write_json

configure_project_environment()

SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def patch_rows(label_path: Path) -> tuple[list[str], int]:
    output: list[str] = []
    clipped_boxes = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or float(fields[0]) != 0.0:
            raise ValueError(f"Expected single-class patch YOLO labels at {label_path}:{line_number}")
        cx, cy, width, height = map(float, fields[1:])
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid normalized box at {label_path}:{line_number}: {line}")
        x1, y1 = max(0.0, cx - width / 2), max(0.0, cy - height / 2)
        x2, y2 = min(1.0, cx + width / 2), min(1.0, cy + height / 2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Box has no area after clipping at {label_path}:{line_number}: {line}")
        if (x1, y1, x2, y2) != (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2):
            clipped_boxes += 1
        new_cx, new_cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_w, new_h = x2 - x1, y2 - y1
        output.append(f"0 {new_cx:.8f} {new_cy:.8f} {new_w:.8f} {new_h:.8f}")
    if not output:
        raise ValueError(f"No patch boxes in {label_path}")
    return output, clipped_boxes


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        destination.hardlink_to(source)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> int:
    require_local_venv()
    source = project_path("dataset/external/adv_tshirt_1k/AdvT-shirt", must_exist=True)
    audit_path = project_path("results/adv_tshirt_patch_label_audit.json", must_exist=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    integrity = audit["label_integrity"]
    invalid = any(
        integrity[key]
        for key in (
            "missing_patch_label_images",
            "orphan_patch_label_files",
            "invalid_patch_files",
            "invalid_images",
        )
    )
    if invalid:
        raise RuntimeError("AdvT-shirt patch source did not pass its integrity audit; refusing to create training splits.")

    dataset_dir = project_path("dataset/adv_tshirt_patch_detection")
    if dataset_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived dataset: {dataset_dir}")
    staging = project_path("dataset/.runtime/adv_tshirt_patch_detection_staging")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging directory: {staging}")
    for split in SPLIT_RATIOS:
        (staging / "images" / split).mkdir(parents=True, exist_ok=False)
        (staging / "labels" / split).mkdir(parents=True, exist_ok=False)

    rng = random.Random(SEED)
    split_manifest: dict[str, list[dict]] = {split: [] for split in SPLIT_RATIOS}
    group_counts: dict[str, dict[str, int]] = {}
    copy_modes: Counter[str] = Counter()
    clipped_total = 0
    box_totals: Counter[str] = Counter()

    for group in sorted(path for path in source.iterdir() if path.is_dir()):
        image_dir, patch_dir = group / "images", group / "adv_labels"
        images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        rng.shuffle(images)
        n_val = round(len(images) * SPLIT_RATIOS["val"])
        n_test = round(len(images) * SPLIT_RATIOS["test"])
        assignments = {
            "train": images[: len(images) - n_val - n_test],
            "val": images[len(images) - n_val - n_test : len(images) - n_test],
            "test": images[len(images) - n_test :],
        }
        group_counts[group.name] = {split: len(items) for split, items in assignments.items()}
        for split, members in assignments.items():
            for image in members:
                source_label = patch_dir / f"{image.stem}.txt"
                if not source_label.is_file():
                    raise FileNotFoundError(f"Missing patch annotation for {group.name}/{image.name}")
                base = f"{group.name}__{image.name}"
                image_target = staging / "images" / split / base
                label_target = staging / "labels" / split / f"{Path(base).stem}.txt"
                copy_modes[link_or_copy(image, image_target)] += 1
                rows, clipped = patch_rows(source_label)
                clipped_total += clipped
                label_target.write_text("\n".join(rows) + "\n", encoding="utf-8")
                box_totals[split] += len(rows)
                split_manifest[split].append(
                    {
                        "image": image_target.relative_to(staging).as_posix(),
                        "label": label_target.relative_to(staging).as_posix(),
                        "source": f"{group.name}/{image.name}",
                    }
                )

    staging_manifest = {
        "source": "AdvT-shirt-1K",
        "seed": SEED,
        "split_ratios_requested": SPLIT_RATIOS,
        "group_counts": group_counts,
        "image_counts": {split: len(items) for split, items in split_manifest.items()},
        "patch_box_counts": dict(box_totals),
        "rounded_edge_boxes_clipped": clipped_total,
        "image_link_modes": dict(copy_modes),
        "source_partition_status": "The source mirror has no official train/validation/test split; this is a derived, stratified-by-source-folder split.",
        "split_manifest": split_manifest,
    }
    (staging / "manifest.json").write_text(json.dumps(staging_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    staging.rename(dataset_dir)

    yaml_path = project_path("dataset/adv_tshirt_patch_detection.yaml")
    yaml_path.write_text(
        "# Derived fixed split from AdvT-shirt-1K; see results/adv_tshirt_patch_dataset_manifest.json.\n"
        "path: ./dataset/adv_tshirt_patch_detection\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: patch\n",
        encoding="utf-8",
    )
    result = {
        "created_at_utc": utc_timestamp(),
        "dataset_yaml": "dataset/adv_tshirt_patch_detection.yaml",
        "derived_dataset_root": "dataset/adv_tshirt_patch_detection",
        **staging_manifest,
        "license": "Apache-2.0 as listed on the Hugging Face dataset card; cite and retain the source attribution.",
        "source_card": "https://huggingface.co/datasets/xingjunm/Adv-T-shirt",
        "source_repository": "https://github.com/Wwangb/AdvT-shirt-1K",
    }
    result["split_manifest"] = {
        split: [
            {**item, "image": f"dataset/adv_tshirt_patch_detection/{item['image']}", "label": f"dataset/adv_tshirt_patch_detection/{item['label']}"}
            for item in items
        ]
        for split, items in split_manifest.items()
    }
    write_json("results/adv_tshirt_patch_dataset_manifest.json", result)
    print(json.dumps({"dataset_yaml": str(yaml_path.relative_to(PROJECT_ROOT)), "image_counts": result["image_counts"], "patch_box_counts": result["patch_box_counts"], "clipped_boxes": clipped_total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
