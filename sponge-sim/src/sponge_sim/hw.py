"""Phần cứng: đọc topology CPU (read-only), ghim affinity thật và xác minh bằng tải.

Không đổi power plan / turbo. Chỉ đọc và in checklist cho người dùng tự đặt.
"""
from __future__ import annotations

import ctypes
import platform
import sys
import time

import psutil

RELATION_PROCESSOR_CORE = 0
LTP_PC_SMT = 0x1


def cpu_topology() -> list[dict]:
    """Danh sách core vật lý: logical CPUs, EfficiencyClass, SMT. Đọc qua Win32 API.

    EfficiencyClass càng cao = core hiệu năng càng cao (P-core). Không suy diễn
    từ chỉ số logical CPU.
    """
    if sys.platform != 'win32':
        raise RuntimeError('cpu_topology chỉ hỗ trợ Windows trong harness này')
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    length = ctypes.c_ulong(0)
    kernel32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, None, ctypes.byref(length))
    buf = ctypes.create_string_buffer(length.value)
    if not kernel32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf, ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    raw, offset, cores = buf.raw, 0, []
    while offset < length.value:
        relationship = int.from_bytes(raw[offset:offset + 4], 'little')
        size = int.from_bytes(raw[offset + 4:offset + 8], 'little')
        if relationship == RELATION_PROCESSOR_CORE:
            flags, eff = raw[offset + 8], raw[offset + 9]
            group_count = int.from_bytes(raw[offset + 30:offset + 32], 'little')
            logical = []
            for g in range(group_count):
                base = offset + 32 + g * 16  # GROUP_AFFINITY: Mask(8) Group(2) Reserved(6)
                mask = int.from_bytes(raw[base:base + 8], 'little')
                group = int.from_bytes(raw[base + 8:base + 10], 'little')
                logical += [group * 64 + i for i in range(64) if mask >> i & 1]
            cores.append(dict(core_index=len(cores), logical_cpus=logical,
                              efficiency_class=eff, smt=bool(flags & LTP_PC_SMT)))
        offset += size
    max_eff = max(c['efficiency_class'] for c in cores)
    for c in cores:
        c['kind'] = 'P' if c['efficiency_class'] == max_eff and max_eff > 0 else 'E'
    return cores


def profiles(cores: list[dict]) -> dict[str, list[int]]:
    """Profile ghim: mỗi core vật lý lấy logical CPU đầu tiên (tránh HT anh em)."""
    p = [c['logical_cpus'][0] for c in cores if c['kind'] == 'P']
    e = [c['logical_cpus'][0] for c in cores if c['kind'] == 'E']
    out = {}
    if p:
        out['P1'] = p[:1]
    if e:
        out['E1'] = e[:1]
    if len(e) >= 4:
        out['E4'] = e[:4]
    return out


def pin(cpus: list[int]) -> list[int]:
    """Ghim process hiện tại; trả về affinity đọc lại (read-back)."""
    proc = psutil.Process()
    proc.cpu_affinity(list(cpus))
    return sorted(proc.cpu_affinity())


def verify_pin(cpus: list[int], seconds: float = 2.0, threads: int | None = None) -> dict:
    """Xác minh ghim bằng tải thật (torch matmul, nhả GIL) với `threads` intra-op threads.

    Hai tiêu chí (cả hai phải đạt):
    - effective_cores = CPU-time của process / wall ≤ len(cpus) + 0.15
      (process không thể dùng nhiều CPU hơn tập được phép);
    - mean cpu_percent trên các CPU được phép ≥ 70%.
    Mặc định threads = 2*len(cpus) để cố ý vượt tập được phép: nếu ghim không có
    hiệu lực, effective_cores sẽ > len(cpus).
    """
    import torch
    n = threads or 2 * len(cpus)
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    a = torch.randn(512, 512)
    proc = psutil.Process()
    psutil.cpu_percent(percpu=True)
    t0, c0 = time.perf_counter(), sum(proc.cpu_times()[:2])
    while time.perf_counter() - t0 < seconds:
        a = torch.tanh(a @ a)
    wall, cpu = time.perf_counter() - t0, sum(proc.cpu_times()[:2]) - c0
    per = psutil.cpu_percent(percpu=True)
    torch.set_num_threads(old)
    eff = cpu / wall
    inside = sum(per[i] for i in cpus) / len(cpus)
    return dict(allowed=sorted(cpus), threads=n, wall_s=wall, cpu_s=cpu,
                effective_cores=eff, mean_percent_allowed=inside, percpu=per,
                passed=eff <= len(cpus) + 0.15 and inside >= 70.0)


def power_snapshot() -> dict:
    """Chỉ đọc: pin/sạc, tần số. Không gọi lệnh đổi cấu hình."""
    bat = psutil.sensors_battery() if hasattr(psutil, 'sensors_battery') else None
    freq = psutil.cpu_freq()
    return dict(
        platform=platform.platform(), processor=platform.processor(),
        logical=psutil.cpu_count(), physical=psutil.cpu_count(logical=False),
        battery_percent=None if bat is None else bat.percent,
        power_plugged=None if bat is None else bat.power_plugged,
        cpu_freq_mhz=None if freq is None else dict(current=freq.current, min=freq.min, max=freq.max),
        power_plan='CHƯA ĐỌC: người dùng tự chạy `powercfg /getactivescheme` và ghi vào PROGRESS',
    )


CHECKLIST = """Checklist điện/nền (BẠN tự đặt; harness không đổi gì):
  [ ] Cắm sạc suốt phiên đo (power_plugged=True trong manifest).
  [ ] Power plan: ghi lại kết quả `powercfg /getactivescheme`; giữ NGUYÊN plan đó cho mọi lượt đo.
  [ ] Windows Power mode (Settings > System > Power): ghi lại mức (Best performance / Balanced ...), giữ nguyên.
  [ ] Turbo/boost: không đổi; nếu muốn cố định thì tự đặt và ghi vào DEVIATIONS/PROGRESS trước khi đo.
  [ ] Đóng trình duyệt, IDE indexer, OneDrive sync, Windows Update, antivirus quét theo lịch.
  [ ] Không dùng máy khi đang đo; tắt sleep/màn hình tắt trong thời gian đo.
  [ ] Để máy nghỉ ~5 phút trước khi đo (nhiệt độ ổn định).
"""


if __name__ == '__main__':
    import json
    cores = cpu_topology()
    print(json.dumps(dict(cores=cores, profiles=profiles(cores), power=power_snapshot()), indent=1, ensure_ascii=False))
    print(CHECKLIST)
