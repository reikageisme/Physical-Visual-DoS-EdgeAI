"""Audit split identity and exact image duplication without changing the study."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from pipeline_utils import configure_project_environment, project_path, require_local_venv


def main() -> None:
    require_local_venv()
    configure_project_environment()
    manifest = json.loads(project_path("dataset/fullscale_person/manifest.json", must_exist=True).read_text(encoding="utf-8"))
    hashes = defaultdict(list)
    identities = defaultdict(list)
    sources = {}
    count = 0
    for split, rows in manifest["split_manifest"].items():
        sources[split] = dict(Counter(f"{r['source']}:{r['source_split']}" for r in rows))
        for row in rows:
            path = project_path(row["image"], must_exist=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            item = {"split": split, "image": row["image"]}
            hashes[digest].append(item)
            if row.get("coco_image_id") is not None:
                identities[str(row["coco_image_id"])].append(item)
            count += 1
    duplicate_groups = [v for v in hashes.values() if len(v) > 1]
    across = [v for v in duplicate_groups if len({r["split"] for r in v}) > 1]
    id_overlap = [v for v in identities.values() if len({r["split"] for r in v}) > 1]
    result = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "images_hashed": count,
        "source_split_counts": sources,
        "exact_duplicate_groups": duplicate_groups,
        "cross_split_exact_duplicate_groups": across,
        "cross_split_coco_identity_groups": id_overlap,
        "limitations": [
            "SHA-256 detects exact byte duplicates, not similar frames or correlated people/scenes.",
            "COCO source splits were pooled and re-split; pretrained exposure is not excluded.",
            "Image-level random splitting does not establish subject/sequence disjointness for AdvT-shirt.",
        ],
    }
    target = project_path("results/fullscale_setup/split_integrity_audit.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images_hashed": count, "exact_duplicate_groups": len(duplicate_groups), "cross_split_exact_duplicate_groups": len(across), "cross_split_coco_identity_groups": len(id_overlap), "source_split_counts": sources}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
