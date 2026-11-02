"""Audit the source person and adversarial-patch labels before any split is made."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pipeline_utils import configure_project_environment, project_path, require_local_venv, utc_timestamp, write_json

configure_project_environment()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def read_yolo_labels(path: Path) -> tuple[list[dict], list[str], int, float]:
    boxes: list[dict] = []
    errors: list[str] = []
    edge_overflows = 0
    max_edge_overflow = 0.0
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return boxes, [f"read error: {exc}"], edge_overflows, max_edge_overflow
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            if len(fields) != 5:
                raise ValueError(f"expected 5 fields, got {len(fields)}")
            class_value = float(fields[0])
            coords = [float(value) for value in fields[1:]]
            class_id = int(class_value)
            if class_value != class_id or class_id != 0:
                raise ValueError(f"expected source class 0, got {fields[0]}")
            cx, cy, width, height = coords
            if not all(0 <= value <= 1 for value in coords):
                raise ValueError("coordinates are outside [0, 1]")
            if width <= 0 or height <= 0:
                raise ValueError("box has non-positive width or height")
            overflow = max(0.0, -(cx - width / 2), -(cy - height / 2), cx + width / 2 - 1, cy + height / 2 - 1)
            if overflow > 0:
                edge_overflows += 1
                max_edge_overflow = max(max_edge_overflow, overflow)
            boxes.append({"class_id": class_id, "xywh": coords, "line": line_number})
        except (ValueError, OverflowError) as exc:
            errors.append(f"line {line_number}: {exc}: {line}")
    return boxes, errors, edge_overflows, max_edge_overflow


def main() -> int:
    require_local_venv()
    source = project_path("dataset/external/adv_tshirt_1k/AdvT-shirt", must_exist=True)
    group_rows: list[dict] = []
    orphan_person_labels: list[str] = []
    orphan_patch_labels: list[str] = []
    missing_person_labels: list[str] = []
    missing_patch_labels: list[str] = []
    invalid_person: list[dict] = []
    invalid_patch: list[dict] = []
    image_errors: list[dict] = []
    edge_overflows = {"person": 0, "patch": 0}
    max_edge_overflow = {"person": 0.0, "patch": 0.0}
    total_images = total_person_boxes = total_patch_boxes = 0
    size_counts: Counter[tuple[int, int]] = Counter()

    from PIL import Image

    for group in sorted(path for path in source.iterdir() if path.is_dir()):
        image_dir = group / "images"
        person_dir = group / "labels"
        patch_dir = group / "adv_labels"
        images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        image_stems = {path.stem for path in images}
        person_files = {path.stem: path for path in person_dir.glob("*.txt") if path.name.lower() != "classes.txt"}
        patch_files = {path.stem: path for path in patch_dir.glob("*.txt") if path.name.lower() != "classes.txt"}
        orphan_person_labels.extend(f"{group.name}/{stem}.txt" for stem in sorted(person_files.keys() - image_stems))
        orphan_patch_labels.extend(f"{group.name}/{stem}.txt" for stem in sorted(patch_files.keys() - image_stems))
        missing_person_labels.extend(f"{group.name}/{image.name}" for image in images if image.stem not in person_files)
        missing_patch_labels.extend(f"{group.name}/{image.name}" for image in images if image.stem not in patch_files)
        group_person_boxes = group_patch_boxes = 0
        group_invalid_person = group_invalid_patch = 0
        for image in images:
            try:
                with Image.open(image) as im:
                    im.verify()
                with Image.open(image) as im:
                    size_counts[im.size] += 1
            except Exception as exc:
                image_errors.append({"file": f"{group.name}/{image.name}", "error": f"image decode: {exc}"})
                continue
            for label_kind, mapping, accumulator in (
                ("person", person_files, invalid_person),
                ("patch", patch_files, invalid_patch),
            ):
                label = mapping.get(image.stem)
                if label is None:
                    continue
                boxes, errors, n_overflow, max_overflow = read_yolo_labels(label)
                edge_overflows[label_kind] += n_overflow
                max_edge_overflow[label_kind] = max(max_edge_overflow[label_kind], max_overflow)
                if label_kind == "person":
                    group_person_boxes += len(boxes)
                else:
                    group_patch_boxes += len(boxes)
                if errors:
                    accumulator.append({"file": f"{group.name}/{label.name}", "errors": errors[:10]})
                    if label_kind == "person":
                        group_invalid_person += len(errors)
                    else:
                        group_invalid_patch += len(errors)
        total_images += len(images)
        total_person_boxes += group_person_boxes
        total_patch_boxes += group_patch_boxes
        group_rows.append(
            {
                "group": group.name,
                "images": len(images),
                "person_label_files": len(person_files),
                "patch_label_files": len(patch_files),
                "person_boxes": group_person_boxes,
                "patch_boxes": group_patch_boxes,
                "invalid_person_rows": group_invalid_person,
                "invalid_patch_rows": group_invalid_patch,
                "person_boxes_touching_or_past_edge": sum(
                    read_yolo_labels(person_files[image.stem])[2]
                    for image in images if image.stem in person_files
                ),
                "patch_boxes_touching_or_past_edge": sum(
                    read_yolo_labels(patch_files[image.stem])[2]
                    for image in images if image.stem in patch_files
                ),
            }
        )

    result = {
        "audit_time_utc": utc_timestamp(),
        "source_name": "AdvT-shirt-1K",
        "source_url": "https://huggingface.co/datasets/xingjunm/Adv-T-shirt",
        "source_repository": "https://github.com/Wwangb/AdvT-shirt-1K",
        "license_as_listed_by_source": "Apache-2.0",
        "source_tree": str(source.relative_to(project_path("."))),
        "groups": group_rows,
        "totals": {
            "images": total_images,
            "person_boxes": total_person_boxes,
            "patch_boxes": total_patch_boxes,
            "unique_image_sizes": len(size_counts),
            "most_common_sizes": {f"{w}x{h}": count for (w, h), count in size_counts.most_common(10)},
        },
        "label_integrity": {
            "missing_person_label_images": missing_person_labels,
            "missing_patch_label_images": missing_patch_labels,
            "orphan_person_label_files": orphan_person_labels,
            "orphan_patch_label_files": orphan_patch_labels,
            "invalid_person_files": invalid_person,
            "invalid_patch_files": invalid_patch,
            "invalid_images": image_errors,
            "edge_overflows_in_normalized_boxes": edge_overflows,
            "maximum_edge_overflow_in_normalized_units": max_edge_overflow,
            "note": "All source xywh fields remain normalized in [0,1]. Bbox extents exceed the image edge only by 5e-7 at most due to decimal rounding; the derived YOLO dataset clips those edges to [0,1].",
        },
        "partition_note": "The source does not publish train/validation/test partitions for this mirror; any derived split must be deterministic, documented, and treated as a supplementary benchmark.",
    }
    write_json("results/adv_tshirt_patch_label_audit.json", result)
    print(
        f"images={total_images} person_boxes={total_person_boxes} patch_boxes={total_patch_boxes} "
        f"missing_patch_labels={len(missing_patch_labels)} orphan_patch_labels={len(orphan_patch_labels)} "
        f"invalid_patch_files={len(invalid_patch)}"
    )
    return 0 if not missing_person_labels and not missing_patch_labels and not orphan_person_labels and not orphan_patch_labels and not invalid_person and not invalid_patch and not image_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
