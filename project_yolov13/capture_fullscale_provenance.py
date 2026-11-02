"""Capture local dependency versions and hashes for the actual model sources."""

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone

from pipeline_utils import configure_project_environment, project_path, require_local_venv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require_local_venv()
    configure_project_environment()
    source_records = {}
    for name in ("yolov13_src", "yolov12_src"):
        source = project_path(name, must_exist=True)
        files = {str(p.relative_to(source)).replace("\\", "/"): digest(p) for p in sorted(source.rglob("*.py")) if "__pycache__" not in p.parts}
        commit = None
        if (source / ".git").exists():
            result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            commit = result.stdout.strip()
        source_records[name] = {"git_commit": commit, "python_files_sha256": files}
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name")}, key=str.casefold)
    target = project_path("results/fullscale_setup/dependency_versions.txt")
    target.write_text("\n".join(packages) + "\n", encoding="utf-8")
    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sources": source_records,
        "pretrained_weights_sha256": {m: digest(project_path(f"weights/{m}.pt", must_exist=True)) for m in ("yolo11n", "yolov12n", "yolov13n", "yolov13s")},
        "dataset_manifest_sha256": digest(project_path("dataset/fullscale_person/manifest.json", must_exist=True)),
        "dependency_versions_file": str(target.relative_to(project_path("."))).replace("\\", "/"),
        "dependency_versions_sha256": digest(target),
        "limitations": "Package versions and source/weight hashes identify the local study environment; they are not an independent reproducibility rerun.",
    }
    project_path("results/fullscale_setup/provenance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Captured {len(packages)} package versions and {sum(len(r['python_files_sha256']) for r in source_records.values())} source-file hashes")


if __name__ == "__main__":
    main()
