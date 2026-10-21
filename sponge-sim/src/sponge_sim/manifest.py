"""manifest.json cho mỗi run: git, hash code/weights/config, phiên bản, phần cứng, affinity read-back."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil

from . import backends, hw
from .cohorts import REPO

SIM = REPO / 'sponge-sim'
WEIGHTS = REPO / 'yolov8n.pt'


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str:
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def build(config: dict, extra_files: list[Path] = ()) -> dict:
    code = sorted((SIM / 'src').rglob('*.py')) + sorted((SIM / 'experiments').rglob('*.py')) + list(extra_files)
    cores = hw.cpu_topology()
    return dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        git=dict(head=_git('rev-parse', 'HEAD'), branch=_git('branch', '--show-current'),
                 tags_at_head=_git('tag', '--points-at', 'HEAD').split(),
                 sponge_sim_dirty=bool(_git('status', '--porcelain', '--', 'sponge-sim/src', 'sponge-sim/experiments'))),
        code_sha256={str(Path(p).relative_to(REPO)).replace('\\', '/'): sha256(p) for p in code},
        weights=dict(path='yolov8n.pt', sha256=sha256(WEIGHTS)),
        config=config, config_sha256=config_hash(config),
        python=sys.version, platform=platform.platform(), backends=backends.metadata(),
        cpu_topology=cores, cpu_profiles=hw.profiles(cores), power=hw.power_snapshot(),
        affinity_now=sorted(psutil.Process().cpu_affinity()),
    )


def write(path: Path, manifest: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
