"""Save host CPU/RAM and Python environment details for the experiment record."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, require_local_venv

configure_project_environment()


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def main() -> int:
    require_local_venv()
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    value = {
        "cpu_name": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "ram_gib": round(status.ullTotalPhys / 2**30, 2),
        "available_ram_gib_at_capture": round(status.ullAvailPhys / 2**30, 2),
        "platform": platform.platform(),
        "python": sys.version,
        "capture_scope": "inside project_yolov13/.venv",
    }
    output = PROJECT_ROOT / "results" / "fullscale_setup" / "hardware.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
