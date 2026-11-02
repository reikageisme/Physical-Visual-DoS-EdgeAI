"""Pin Ultralytics settings to project-local dataset, run, weight, and config paths."""

from __future__ import annotations

import json

from pipeline_utils import PROJECT_ROOT, configure_project_environment, project_path, require_local_venv, write_json

configure_project_environment()


def main() -> int:
    require_local_venv()
    # Import only after YOLO_CONFIG_DIR has been set by the project utility.
    from ultralytics.utils import SETTINGS, SETTINGS_FILE

    dataset_dir = project_path("dataset")
    runs_dir = project_path("runs")
    weights_dir = project_path("weights")
    for directory in (dataset_dir, runs_dir, weights_dir):
        directory.mkdir(parents=True, exist_ok=True)

    SETTINGS.update(
        {
            "datasets_dir": str(dataset_dir),
            "runs_dir": str(runs_dir),
            "weights_dir": str(weights_dir),
        }
    )

    settings_file = project_path(SETTINGS_FILE, must_exist=True)
    resolved = {
        key: str(project_path(SETTINGS[key]))
        for key in ("datasets_dir", "runs_dir", "weights_dir")
    }
    receipt = {
        "settings_file": str(settings_file.relative_to(PROJECT_ROOT)),
        "YOLO_CONFIG_DIR": str(project_path(".config")),
        "settings": resolved,
        "all_paths_inside_project": True,
    }
    write_json("results/yolo_path_config.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
