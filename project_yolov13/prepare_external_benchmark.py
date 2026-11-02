from __future__ import annotations
import json
import shutil
from collections import Counter
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, project_path, require_local_venv, utc_timestamp, write_json

configure_project_environment()


def main() -> int:
    require_local_venv()
    from PIL import Image
    import yaml

    source = project_path('dataset/external/adv_tshirt_1k/AdvT-shirt', must_exist=True)
    destination = project_path('dataset/external/adv_tshirt_person_eval')
    images_out = destination / 'images' / 'test'
    labels_out = destination / 'labels' / 'test'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_count = 0
    label_count = 0
    boxes = 0
    missing_labels: list[str] = []
    invalid_rows: list[str] = []
    group_counts: Counter[str] = Counter()
    dimensions: Counter[tuple[int, int]] = Counter()

    for group in sorted(p for p in source.iterdir() if p.is_dir()):
        image_dir = group / 'images'
        label_dir = group / 'labels'
        if not image_dir.is_dir():
            continue
        for image_path in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes):
            source_label = label_dir / f'{image_path.stem}.txt'
            if not source_label.is_file():
                missing_labels.append(f'{group.name}/{image_path.name}')
                continue
            label_rows = [line.strip() for line in source_label.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            valid: list[str] = []
            for line_number, line in enumerate(label_rows, 1):
                values = line.split()
                try:
                    if len(values) != 5 or int(values[0]) != 0:
                        raise ValueError('expected a five-column person-class YOLO box')
                    cx, cy, width, height = map(float, values[1:])
                    if not all(0.0 <= v <= 1.0 for v in (cx, cy, width, height)) or width <= 0 or height <= 0:
                        raise ValueError('box coordinates are outside normalized bounds')
                    valid.append('0 ' + ' '.join(f'{v:.8f}'.rstrip('0').rstrip('.') for v in (cx, cy, width, height)))
                except Exception as exc:
                    invalid_rows.append(f'{group.name}/{source_label.name}:{line_number}: {exc}')
            if not valid:
                invalid_rows.append(f'{group.name}/{source_label.name}: no valid person boxes')
                continue
            target_name = f'{group.name}__{image_path.name}'
            image_target = images_out / target_name
            label_target = labels_out / f'{Path(target_name).stem}.txt'
            if not image_target.exists():
                try:
                    image_target.hardlink_to(image_path)
                except OSError:
                    shutil.copy2(image_path, image_target)
            label_target.write_text('\n'.join(valid) + '\n', encoding='utf-8')
            with Image.open(image_path) as image:
                dimensions[image.size] += 1
            image_count += 1
            label_count += 1
            boxes += len(valid)
            group_counts[group.name] += 1

    if missing_labels or invalid_rows:
        raise RuntimeError(json.dumps({'missing_labels': missing_labels[:20], 'invalid_rows': invalid_rows[:20]}, indent=2))
    if image_count == 0:
        raise RuntimeError('No labeled person images were found in the external dataset.')

    yaml_path = project_path('dataset/adv_tshirt_person_eval.yaml')
    yaml_path.write_text(
        'path: "./dataset/external/adv_tshirt_person_eval"\n'
        'train: images/test\nval: images/test\ntest: images/test\n'
        'names:\n  0: person\n',
        encoding='utf-8',
    )
    manifest = {
        'audit_time_utc': utc_timestamp(),
        'source_name': 'AdvT-shirt-1K',
        'source_repo': 'https://huggingface.co/datasets/xingjunm/AdvT-shirt',
        'license_claim': 'Apache-2.0 as shown on the Hugging Face dataset page',
        'source_subdirectory': 'AdvT-shirt/*/images and labels',
        'evaluation_scope': 'full external corpus; zero-shot COCO-pretrained person detection; no training or tuning on this corpus',
        'image_count': image_count,
        'published_source_image_count': 1131,
        'mirror_image_count_difference_vs_source_claim': 1131 - image_count,
        'label_file_count': label_count,
        'person_box_count': boxes,
        'missing_label_images': len(missing_labels),
        'invalid_label_rows': len(invalid_rows),
        'group_counts': dict(group_counts),
        'source_image_dimensions': {'unique_sizes': len(dimensions), 'min_width': min(w for w, _ in dimensions), 'max_width': max(w for w, _ in dimensions), 'min_height': min(h for _, h in dimensions), 'max_height': max(h for _, h in dimensions), 'most_common': {f'{w}x{h}': n for (w, h), n in dimensions.most_common(10)}},
        'local_images': 'dataset/external/adv_tshirt_person_eval/images/test',
        'local_labels': 'dataset/external/adv_tshirt_person_eval/labels/test',
        'dataset_yaml': 'dataset/adv_tshirt_person_eval.yaml',
        'test_is_full_corpus': True,
        'training_or_validation_split_provided_by_source': False,
        'notes': 'Only source person-box labels are used. The separate adversarial-patch labels are not mixed into the pretrained COCO person benchmark.',
    }
    write_json(project_path('results/external_transfer_dataset_manifest.json'), manifest)
    print(f'Prepared {image_count} images, {label_count} labels, {boxes} person boxes across {len(group_counts)} source groups.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


