from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

from pipeline_utils import project_path, write_json, utc_timestamp


REVISION = "b1035f8"
URL = f"https://codeload.github.com/sunsmarterjie/yolov12/zip/{REVISION}"
DEST = project_path("yolov12_v1_src")


def main() -> None:
    request = urllib.request.Request(URL, headers={"User-Agent": "project-yolov13-research-pipeline"})
    with urllib.request.urlopen(request, timeout=90) as response:
        archive_bytes = response.read()
    digest = hashlib.sha256(archive_bytes).hexdigest()
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite existing source directory: {DEST}")
    DEST.mkdir(parents=True)
    prefix = f"yolov12-{REVISION}/"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith(prefix):
                continue
            relative = Path(entry.filename[len(prefix):])
            target = project_path(DEST / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
    manifest = {
        "retrieved_at_utc": utc_timestamp(),
        "source_repository": "https://github.com/sunsmarterjie/yolov12",
        "source_revision": REVISION,
        "revision_note": "YOLOv12-v1.0 release commit; AAttn uses qkv, matching the released v1.0 detector weights.",
        "archive_url": URL,
        "archive_sha256": digest,
        "implementation_path": "yolov12_v1_src/",
    }
    write_json("results/yolov12_v1_source_manifest.json", manifest)
    print(f"Retrieved {REVISION}: sha256={digest}; files installed under {DEST.relative_to(project_path('.'))}")


if __name__ == "__main__":
    main()
