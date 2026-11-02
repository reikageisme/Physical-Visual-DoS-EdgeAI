"""Shared project-local path, cache, configuration, and metric helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache"
DATASET_ROOT = PROJECT_ROOT / "dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def project_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a path and reject access outside this project directory."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must stay inside the project directory: {value}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Required project file or directory does not exist: {resolved}")
    return resolved


def configure_project_environment() -> None:
    """Keep Python, model, plotting, temporary, and pip caches inside the project."""
    dataset_cache = DATASET_ROOT / ".cache" / "huggingface"
    folders = {
        "PIP_CACHE_DIR": CACHE_ROOT / "pip",
        "TMP": CACHE_ROOT / "tmp",
        "TEMP": CACHE_ROOT / "tmp",
        "TORCH_HOME": CACHE_ROOT / "torch",
        "YOLO_CONFIG_DIR": PROJECT_ROOT / ".config",
        "MPLCONFIGDIR": CACHE_ROOT / "matplotlib",
        # Hugging Face caches may contain downloaded datasets, so keep them
        # inside the designated dataset boundary as well as the project.
        "HF_HOME": dataset_cache,
        "HF_HUB_CACHE": dataset_cache / "hub",
        "HF_DATASETS_CACHE": dataset_cache / "datasets",
        "XDG_CACHE_HOME": CACHE_ROOT,
    }
    for variable, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(folder)
    # Ultralytics 8.3.63 still calls np.trapz; NumPy 2.5 removed that alias.
    try:
        import numpy as np
        if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
            np.trapz = np.trapezoid
    except ImportError:
        pass
    os.chdir(PROJECT_ROOT)


def require_local_venv() -> None:
    expected = (PROJECT_ROOT / ".venv").resolve()
    actual = Path(sys.prefix).resolve()
    if actual != expected:
        raise RuntimeError(
            "Run this script with the project-local environment: "
            "activate .venv first, or call .venv/Scripts/python.exe directly."
        )


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = project_path(path, must_exist=True)
    with config_path.open("r", encoding="utf-8-sig") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return loaded


def validate_dataset_yaml(path: str | Path, *, require_test: bool = True) -> tuple[Path, dict[str, Any]]:
    """Check split directories and images while preserving the configured split."""
    yaml_path = project_path(path, must_exist=True)
    dataset = load_yaml(yaml_path)
    root_value = dataset.get("path", ".")
    dataset_root = Path(root_value)
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    dataset_root = project_path(dataset_root, must_exist=True)

    required_splits = ["train", "val"] + (["test"] if require_test else [])
    for split in required_splits:
        if split not in dataset:
            raise ValueError(f"{yaml_path} must define the existing {split!r} split.")
        split_value = Path(str(dataset[split]))
        if not split_value.is_absolute():
            split_value = dataset_root / split_value
        split_dir = project_path(split_value, must_exist=True)
        if not split_dir.is_dir():
            raise ValueError(f"The {split} split must be an image directory: {split_dir}")
        count = sum(1 for item in split_dir.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        if count == 0:
            raise ValueError(f"No images found in the configured {split} split: {split_dir}")
    if not dataset.get("names"):
        raise ValueError(f"Set class names in {yaml_path} before training/evaluation.")
    return yaml_path, dataset


def materialize_ultralytics_dataset_yaml(path: str | Path) -> Path:
    """Create a project-local runtime YAML with an absolute, project-safe root.

    The maintained YAML uses the requested project-relative ``path: ./dataset``
    form. Ultralytics resolves relative ``path`` values against its global
    ``datasets_dir`` setting, so passing that YAML unchanged would duplicate the
    dataset directory. This runtime copy preserves the split/class mapping and
    pins ``path`` to its absolute project-local location.
    """
    import yaml

    source = project_path(path, must_exist=True)
    with source.open("r", encoding="utf-8-sig") as handle:
        dataset = yaml.safe_load(handle) or {}
    if not isinstance(dataset, dict):
        raise ValueError(f"Expected a YAML mapping in {source}")
    raw_root = Path(str(dataset.get("path", ".")))
    root = raw_root if raw_root.is_absolute() else PROJECT_ROOT / raw_root
    root = project_path(root, must_exist=True)
    dataset["path"] = str(root)
    digest = hashlib.sha256(str(source.relative_to(PROJECT_ROOT)).encode("utf-8")).hexdigest()[:10]
    output = project_path(Path("dataset/.runtime") / f"{source.stem}-{digest}.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with project_path(path, must_exist=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str | Path, data: Any) -> Path:
    output = project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(output)
    return output


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    rows = list(rows)
    output = project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return output


def metric_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def source_revision() -> str | None:
    revision_file = project_path("results/source_revision.txt")
    if revision_file.is_file():
        value = revision_file.read_text(encoding="utf-8").strip()
        return value or None
    return None
