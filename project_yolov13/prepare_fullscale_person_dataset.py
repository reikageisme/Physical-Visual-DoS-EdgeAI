"""Build a reproducible COCO + AdvT-shirt person-detection benchmark locally.

Downloads the official COCO 2017 train/validation annotations and only a fixed,
seeded subset of images. All files are stored below this project directory.
COCO labels are filtered to the person category so they have the same semantics
as the person boxes in AdvT-shirt-1K. The adversarial-patch boxes remain in the
source archive for domain analysis and are not mislabeled as COCO objects.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from pipeline_utils import PROJECT_ROOT, configure_project_environment, require_local_venv

configure_project_environment()

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANNOTATIONS_BYTES = 252_907_541
COCO_ROOT = PROJECT_ROOT / "dataset" / "external" / "coco2017"
COCO_ANNOTATIONS_DIR = COCO_ROOT / "annotations"
COCO_IMAGE_ROOT = COCO_ROOT / "images"
ADV_ROOT = PROJECT_ROOT / "dataset" / "external" / "adv_tshirt_1k" / "AdvT-shirt"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "fullscale_person"
SETUP_ROOT = PROJECT_ROOT / "results" / "fullscale_setup"
OUTPUT_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
SPLIT_SEED = 2026
COCO_POSITIVE_IMAGES = 8_000
COCO_NEGATIVE_IMAGES = 2_000
WORKERS = 12
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
ADV_GROUPS = ("1_person", "2_persons", "3_5_persons", "6_9_persons", "10_plus_persons")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clip_normalized_xywh(cx: float, cy: float, bw: float, bh: float) -> tuple[float, float, float, float]:
    """Clip normalized center-width boxes to image bounds with rounding headroom."""
    margin = 1e-5
    x1 = max(0.0, cx - bw / 2)
    y1 = max(0.0, cy - bh / 2)
    x2 = min(1.0 - margin, cx + bw / 2)
    y2 = min(1.0 - margin, cy + bh / 2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box has no positive area after clipping: {(cx, cy, bw, bh)}")
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_resumable(url: str, destination: Path, expected_bytes: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_bytes and destination.stat().st_size == expected_bytes:
        return

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "project-yolov13-study/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        remote_size = int(response.headers.get("Content-Length", "0")) or expected_bytes
    if remote_size is None:
        raise RuntimeError(f"The source did not provide a file size: {url}")
    if expected_bytes is not None and remote_size != expected_bytes:
        raise RuntimeError(f"Unexpected source size for {url}: {remote_size} != {expected_bytes}")
    if destination.exists() and destination.stat().st_size == remote_size:
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "project-yolov13-study/1.0"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = downloaded > 0 and response.status == 206
        if not append:
            downloaded = 0
        mode = "ab" if append else "wb"
        started = time.monotonic()
        with partial.open(mode) as output:
            while block := response.read(4 * 1024 * 1024):
                output.write(block)
                downloaded += len(block)
                if downloaded % (64 * 1024 * 1024) < len(block):
                    elapsed = max(time.monotonic() - started, 0.01)
                    print(f"downloaded {downloaded / 2**20:.0f}/{remote_size / 2**20:.0f} MiB ({downloaded / elapsed / 2**20:.1f} MiB/s)", flush=True)
    if partial.stat().st_size != remote_size:
        raise RuntimeError(f"Incomplete download: {partial.stat().st_size} != {remote_size} bytes")
    partial.replace(destination)


def load_coco_instances() -> tuple[list[dict], list[dict]]:
    archive = COCO_ROOT / "annotations_trainval2017.zip"
    fetch_resumable(COCO_ANNOTATIONS_URL, archive, COCO_ANNOTATIONS_BYTES)
    print(f"COCO annotation archive SHA-256: {sha256(archive)}", flush=True)
    with zipfile.ZipFile(archive) as source:
        damaged = source.testzip()
        if damaged:
            raise RuntimeError(f"Corrupt COCO annotation archive member: {damaged}")
        outputs = []
        for split in ("train2017", "val2017"):
            member = f"annotations/instances_{split}.json"
            target = COCO_ANNOTATIONS_DIR / f"instances_{split}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with source.open(member) as input_file, target.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file, 4 * 1024 * 1024)
            outputs.append(json.loads(target.read_text(encoding="utf-8")))
    return outputs[0]["images"] + outputs[1]["images"], outputs[0]["annotations"] + outputs[1]["annotations"]


def largest_remainder(n: int) -> dict[str, int]:
    raw = {name: n * ratio for name, ratio in SPLITS}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = n - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def assign_splits(records: list[dict]) -> None:
    rng = random.Random(SPLIT_SEED)
    groups: dict[str, list[dict]] = {}
    for row in records:
        groups.setdefault(row["split_stratum"], []).append(row)
    for rows in groups.values():
        rng.shuffle(rows)
        counts = largest_remainder(len(rows))
        offset = 0
        for split in ("train", "val", "test"):
            end = offset + counts[split]
            for row in rows[offset:end]:
                row["split"] = split
            offset = end


def collect_coco_records(images: list[dict], annotations: list[dict]) -> list[dict]:
    person_boxes: dict[int, list[list[float]]] = {}
    person_annotation_ids: set[int] = set()
    for ann in annotations:
        if ann.get("category_id") != 1:
            continue
        image_id = int(ann["image_id"])
        person_annotation_ids.add(image_id)
        if ann.get("iscrowd", 0):
            continue
        x, y, width, height = (float(v) for v in ann["bbox"])
        if width <= 0 or height <= 0:
            continue
        person_boxes.setdefault(image_id, []).append([x, y, width, height])

    positive, negative = [], []
    for image in images:
        image_id = int(image["id"])
        # Image records in the concatenated list are tagged by source split below.
        source_split = image.get("_source_split")
        if source_split not in ("train2017", "val2017"):
            raise RuntimeError("COCO source split metadata was not retained")
        boxes = person_boxes.get(image_id, [])
        row = {
            "source": "COCO 2017",
            "source_split": source_split,
            "file_name": image["file_name"],
            "image_id": image_id,
            "width": int(image["width"]),
            "height": int(image["height"]),
            "person_boxes_xywh": boxes,
            "person_count": len(boxes),
            "split_stratum": "coco_person_positive" if boxes else "coco_no_person",
            "source_url": image.get("coco_url") or f"http://images.cocodataset.org/{source_split}/{image['file_name']}",
        }
        if boxes:
            positive.append(row)
        elif image_id not in person_annotation_ids:
            # Do not mislabel COCO images whose only person annotations are
            # crowd regions as clean background images.
            negative.append(row)

    rng = random.Random(SPLIT_SEED)
    if len(positive) < COCO_POSITIVE_IMAGES or len(negative) < COCO_NEGATIVE_IMAGES:
        raise RuntimeError(f"Insufficient COCO strata: positive={len(positive)}, negative={len(negative)}")
    rng.shuffle(positive)
    rng.shuffle(negative)
    selected = positive[:COCO_POSITIVE_IMAGES] + negative[:COCO_NEGATIVE_IMAGES]
    for row in selected:
        row["image_rel"] = f"images/{row['source_split']}/{row['file_name']}"
        row["label_rel"] = f"labels/{row['source_split']}/{Path(row['file_name']).stem}.txt"
    return selected


def collect_adv_records() -> list[dict]:
    if not ADV_ROOT.is_dir():
        raise FileNotFoundError(f"AdvT-shirt-1K source is missing: {ADV_ROOT}")
    records = []
    for group in ADV_GROUPS:
        image_dir = ADV_ROOT / group / "images"
        label_dir = ADV_ROOT / group / "labels"
        patch_dir = ADV_ROOT / group / "adv_labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise RuntimeError(f"Missing AdvT-shirt image/person-label directory for {group}")
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise RuntimeError(f"Missing person label for {image_path}")
            person_rows = []
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                values = line.split()
                if not values:
                    continue
                if len(values) != 5 or int(values[0]) != 0:
                    raise RuntimeError(f"Unexpected person-label schema at {label_path}:{line_no}")
                coords = [float(v) for v in values[1:]]
                if any(not math.isfinite(v) for v in coords) or coords[2] <= 0 or coords[3] <= 0:
                    raise RuntimeError(f"Invalid normalized person box at {label_path}:{line_no}")
                clipped = clip_normalized_xywh(*coords)
                person_rows.append("0 " + " ".join(f"{v:.8f}" for v in clipped))
            patch_path = patch_dir / f"{image_path.stem}.txt"
            records.append({
                "source": "AdvT-shirt-1K",
                "source_group": group,
                "file_name": image_path.name,
                "person_label_lines": person_rows,
                "patch_label_path": str(patch_path.relative_to(PROJECT_ROOT)) if patch_path.exists() else None,
                "person_count": len(person_rows),
                "split_stratum": f"adv_{group}",
                "source_image_path": str(image_path),
                "source_label_path": str(label_path),
            })
    return records


def download_one_image(row: dict) -> tuple[str, int, int]:
    source_split = row["source_split"]
    name = row["file_name"]
    if not re.fullmatch(r"\d{12}\.jpg", name):
        raise RuntimeError(f"Unexpected COCO filename: {name}")
    target = COCO_IMAGE_ROOT / source_split / name
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = (row["width"], row["height"])
    if target.exists():
        try:
            with Image.open(target) as image:
                image.verify()
            with Image.open(target) as image:
                if image.size == expected:
                    return name, target.stat().st_size, 0
        except Exception:
            target.unlink(missing_ok=True)

    last_error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(row["source_url"], headers={"User-Agent": "project-yolov13-study/1.0"})
            temp = target.with_suffix(".jpg.part")
            with urllib.request.urlopen(request, timeout=90) as response, temp.open("wb") as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
            with Image.open(temp) as image:
                image.verify()
            with Image.open(temp) as image:
                if image.size != expected:
                    raise RuntimeError(f"Dimension mismatch {image.size} != {expected}")
            temp.replace(target)
            return name, target.stat().st_size, 1
        except Exception as exc:
            last_error = exc
            target.with_suffix(".jpg.part").unlink(missing_ok=True)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download valid image {name}: {last_error}")


def ensure_all_coco_images(records: list[dict]) -> dict:
    report = {"requested": len(records), "downloaded_now": 0, "already_present": 0, "bytes": 0, "failures": []}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one_image, row): row for row in records}
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                _, size, downloaded = future.result()
                report["bytes"] += size
                report["downloaded_now"] += downloaded
                report["already_present"] += 1 - downloaded
            except Exception as exc:
                row = futures[future]
                report["failures"].append({"file_name": row["file_name"], "error": repr(exc)})
            if idx % 200 == 0 or idx == len(futures):
                elapsed = max(time.monotonic() - started, 0.01)
                print(f"COCO images checked: {idx}/{len(futures)}; new={report['downloaded_now']}; errors={len(report['failures'])}; elapsed={elapsed / 60:.1f} min", flush=True)
    if report["failures"]:
        raise RuntimeError(f"COCO image download/validation failures: {len(report['failures'])}; see {SETUP_ROOT / 'download_failures.json'}")
    return report


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def create_merged_tree(records: list[dict]) -> dict:
    if OUTPUT_ROOT.exists():
        marker = OUTPUT_ROOT / ".build_in_progress"
        manifest = OUTPUT_ROOT / "manifest.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("build_id") == "coco2017-10k-plus-advtshirt-person-v1":
                print(f"Existing merged dataset already prepared: {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
                return existing
            raise FileExistsError(f"A different dataset already exists at {OUTPUT_ROOT}")
        if marker.exists():
            shutil.rmtree(OUTPUT_ROOT)
        else:
            raise FileExistsError(f"Refusing to overwrite unrecognized data directory: {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True)
    (OUTPUT_ROOT / ".build_in_progress").write_text(utc_now(), encoding="utf-8")
    counts = {split: {"images": 0, "person_instances": 0, "coco": 0, "advtshirt": 0} for split, _ in SPLITS}
    split_manifest = {split: [] for split, _ in SPLITS}
    for row in records:
        split = row["split"]
        if row["source"] == "COCO 2017":
            source_image = COCO_IMAGE_ROOT / row["source_split"] / row["file_name"]
            source_label = None
            width, height = row["width"], row["height"]
            label_lines = []
            for x, y, w, h in row["person_boxes_xywh"]:
                x1 = max(0.0, min(float(width), x))
                y1 = max(0.0, min(float(height), y))
                x2 = max(0.0, min(float(width), x + w))
                y2 = max(0.0, min(float(height), y + h))
                if x2 <= x1 or y2 <= y1:
                    continue
                cx, cy, bw, bh = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
                label_lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
            image_name = f"coco_{row['source_split']}_{Path(row['file_name']).stem}.jpg"
            counts[split]["coco"] += 1
        else:
            source_image = Path(row["source_image_path"])
            source_label = Path(row["source_label_path"])
            label_lines = row["person_label_lines"]
            image_name = f"adv_{row['source_group']}_{Path(row['file_name']).stem}.jpg"
            counts[split]["advtshirt"] += 1
        target_image = OUTPUT_ROOT / "images" / split / image_name
        target_label = OUTPUT_ROOT / "labels" / split / f"{Path(image_name).stem}.txt"
        link_or_copy(source_image, target_image)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        counts[split]["images"] += 1
        counts[split]["person_instances"] += len(label_lines)
        split_manifest[split].append({
            "image": str(target_image.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "label": str(target_label.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source": row["source"],
            "source_split": row.get("source_split", row.get("source_group")),
            "person_count": len(label_lines),
            "coco_image_id": row.get("image_id"),
        })

    dataset_yaml = """# One-class person detection: COCO 2017 subset + full available AdvT-shirt-1K.
# Run training commands from project_yolov13/.
path: ./dataset/fullscale_person
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: person
"""
    OUTPUT_YAML.write_text(dataset_yaml, encoding="utf-8")
    (OUTPUT_ROOT / "data.yaml").write_text(dataset_yaml, encoding="utf-8")
    manifest = {
        "build_id": "coco2017-10k-plus-advtshirt-person-v1",
        "created_at_utc": utc_now(),
        "source_urls": {
            "coco": "https://cocodataset.org/#download",
            "coco_annotations_archive": COCO_ANNOTATIONS_URL,
            "advtshirt_official_project": "https://github.com/Wwangb/AdvT-shirt-1K",
            "advtshirt_download_mirror": "https://huggingface.co/datasets/xingjunm/AdvT-shirt",
        },
        "source_license_metadata": {
            "coco_annotations": "CC BY 4.0; COCO images retain their individual Flickr terms/licenses",
            "advtshirt_huggingface_card": "Apache-2.0 declared on the linked mirror; retain source attribution and verify intended use",
        },
        "task": "single-class person detection; COCO labels filtered to category_id=1; AdvT-shirt person boxes mapped to class 0",
        "class_map": {"0": "person"},
        "split_ratios_requested": {name: ratio for name, ratio in SPLITS},
        "split_seed": SPLIT_SEED,
        "coco_sample": {"positive_images_requested": COCO_POSITIVE_IMAGES, "negative_images_requested": COCO_NEGATIVE_IMAGES, "sample_seed": SPLIT_SEED},
        "image_download_report": json.loads((SETUP_ROOT / "coco_image_download_report.json").read_text(encoding="utf-8")),
        "split_counts": counts,
        "split_manifest": split_manifest,
        "notes": [
            "COCO train2017 and val2017 images were pooled and re-split locally to enforce a single fixed 70/15/15 split.",
            "COCO images are a deterministic 10,000-image subset: 8,000 with at least one non-crowd person box and 2,000 with no person annotations.",
            "Images with only crowd-marked person annotations were excluded from both COCO sampling strata; crowd boxes are not converted into YOLO boxes.",
            "The AdvT-shirt source mirror currently contains 1,123 image files while its project page reports 1,131; all locally available source images with valid person labels were included.",
            "Patch labels are preserved at their source paths and used only for a separate physical-domain interpretation; they are not added to the person-detection training target.",
            "All source images and derived files are project-local. Dataset imagery must not be pushed to a public Git remote.",
        ],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / ".build_in_progress").unlink(missing_ok=True)
    return manifest


def main() -> int:
    require_local_venv()
    SETUP_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Download/source root: {COCO_ROOT.relative_to(PROJECT_ROOT)}", flush=True)
    images, annotations = load_coco_instances()
    # Retain the official train/val origin for URL routing and provenance.
    train_json = json.loads((COCO_ANNOTATIONS_DIR / "instances_train2017.json").read_text(encoding="utf-8"))
    train_names = {row["file_name"] for row in train_json["images"]}
    for image in images:
        image["_source_split"] = "train2017" if image["file_name"] in train_names else "val2017"
    coco_records = collect_coco_records(images, annotations)
    adv_records = collect_adv_records()
    records = coco_records + adv_records
    assign_splits(records)
    print(f"Selected COCO images: {len(coco_records)}; AdvT-shirt images: {len(adv_records)}", flush=True)
    print("Downloading/validating selected images from the official COCO image host...", flush=True)
    report = ensure_all_coco_images(coco_records)
    (SETUP_ROOT / "coco_image_download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = create_merged_tree(records)
    audit_summary = {
        "build_id": manifest["build_id"],
        "created_at_utc": manifest["created_at_utc"],
        "dataset_yaml": str(OUTPUT_YAML.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "split_counts": manifest["split_counts"],
        "total_images": sum(row["images"] for row in manifest["split_counts"].values()),
        "download_report": report,
        "annotation_archive_sha256": sha256(COCO_ROOT / "annotations_trainval2017.zip"),
    }
    (SETUP_ROOT / "dataset_build_summary.json").write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
