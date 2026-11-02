"""Clip rounded AdvT-shirt person boxes in the generated full-scale split."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, require_local_venv
from prepare_fullscale_person_dataset import ADV_ROOT, clip_normalized_xywh

configure_project_environment()


def main() -> int:
    require_local_venv()
    dataset_root = (PROJECT_ROOT / "dataset" / "fullscale_person").resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_rows = 0
    changed_files = 0
    source_out_of_bounds_rows = 0
    maximum_overrun = 0.0
    for split, rows in manifest["split_manifest"].items():
        for row in rows:
            if row["source"] != "AdvT-shirt-1K":
                continue
            output_label = (PROJECT_ROOT / row["label"]).resolve()
            source_group = row["source_split"]
            source_stem = Path(row["image"]).stem.removeprefix(f"adv_{source_group}_")
            source_label = (ADV_ROOT / source_group / "labels" / f"{source_stem}.txt").resolve()
            if dataset_root not in output_label.parents or ADV_ROOT.resolve() not in source_label.parents:
                raise RuntimeError("Resolved label path escaped its expected project-local data directory")
            rendered = []
            for line_no, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), 1):
                values = line.split()
                if not values:
                    continue
                if len(values) != 5 or int(values[0]) != 0:
                    raise RuntimeError(f"Unexpected source annotation schema at {source_label}:{line_no}")
                x, y, w, h = (float(value) for value in values[1:])
                overrun = max(0.0, -(x - w / 2), x + w / 2 - 1.0, -(y - h / 2), y + h / 2 - 1.0)
                if overrun > 1e-9:
                    source_out_of_bounds_rows += 1
                    maximum_overrun = max(maximum_overrun, overrun)
                clipped = clip_normalized_xywh(x, y, w, h)
                if any(abs(a - b) > 1e-9 for a, b in zip((x, y, w, h), clipped)):
                    changed_rows += 1
                rendered.append("0 " + " ".join(f"{value:.8f}" for value in clipped))
            if len(rendered) != row["person_count"]:
                raise RuntimeError(f"Person-box count changed for {output_label}")
            output_label.write_text("\n".join(rendered) + ("\n" if rendered else ""), encoding="utf-8")
            changed_files += 1

    note = (
        "AdvT-shirt person boxes were re-encoded with 1e-5 normalized edge headroom. "
        f"{source_out_of_bounds_rows} source rows exceeded image bounds by no more than {maximum_overrun:.8g} normalized units."
    )
    notes = manifest.setdefault("notes", [])
    if note not in notes:
        notes.append(note)
    manifest["advtshirt_box_sanitization"] = {
        "files_rewritten": changed_files,
        "rows_reencoded_or_edge_adjusted": changed_rows,
        "source_out_of_bounds_rows": source_out_of_bounds_rows,
        "maximum_source_overrun_normalized": maximum_overrun,
        "method": "xyxy clip to [0, 1] with 1e-5 edge headroom, then re-encode normalized xywh",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = PROJECT_ROOT / "results" / "fullscale_setup" / "dataset_build_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["advtshirt_box_sanitization"] = manifest["advtshirt_box_sanitization"]
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["advtshirt_box_sanitization"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
