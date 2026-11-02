from __future__ import annotations
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, project_path, require_local_venv, write_json, utc_timestamp

configure_project_environment()

REVISION = '2abab7153a065fb2925e8088e9ca2b19016ab7d6'
WEIGHTS = {scale: f'yolov12{scale}.pt' for scale in ('n', 's', 'l')}


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'project-yolov13-research-pipeline'})
    with urllib.request.urlopen(request, timeout=120) as response, target.open('wb') as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    require_local_venv()
    repo_dir = project_path('yolov12_src')
    if not (repo_dir / 'ultralytics' / '__init__.py').is_file():
        archive = project_path('.cache/yolov12_source.zip')
        fetch(f'https://github.com/sunsmarterjie/yolov12/archive/{REVISION}.zip', archive)
        repo_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            prefix = f'yolov12-{REVISION}/'
            for member in zf.infolist():
                if not member.filename.startswith(prefix):
                    continue
                relative = Path(member.filename[len(prefix):])
                if not relative.parts:
                    continue
                target = (repo_dir / relative).resolve()
                target.relative_to(repo_dir.resolve())
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, target.open('wb') as dst:
                        dst.write(src.read())
        archive.unlink(missing_ok=True)
    weight_hashes = {}
    for scale, filename in WEIGHTS.items():
        path = project_path(Path('weights') / filename)
        if not path.is_file():
            fetch(f'https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/{filename}', path)
        weight_hashes[scale] = sha256(path)
    manifest = {
        'retrieved_at_utc': utc_timestamp(),
        'source_repository': 'https://github.com/sunsmarterjie/yolov12',
        'source_revision': REVISION,
        'source_license': 'AGPL-3.0',
        'weight_release': 'v1.0 (official YOLOv12 repository)',
        'weights': {scale: {'path': f'weights/{filename}', 'sha256': weight_hashes[scale]} for scale, filename in WEIGHTS.items()},
        'implementation_path': 'yolov12_src/',
    }
    write_json(project_path('results/yolov12_source_manifest.json'), manifest)
    print(f'Prepared official YOLOv12 source at revision {REVISION} and {len(WEIGHTS)} release weights inside the project.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
