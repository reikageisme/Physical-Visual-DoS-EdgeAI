from __future__ import annotations
from pipeline_utils import PROJECT_ROOT, configure_project_environment, project_path, require_local_venv

configure_project_environment()


def main() -> None:
    require_local_venv()
    from huggingface_hub import snapshot_download

    destination = project_path('dataset/external/adv_tshirt_1k')
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id='xingjunm/AdvT-shirt',
        repo_type='dataset',
        local_dir=str(destination),
        cache_dir=str(project_path('dataset/.cache/huggingface/hub')),
        max_workers=8,
    )
    print(f'Downloaded dataset snapshot into {destination.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
