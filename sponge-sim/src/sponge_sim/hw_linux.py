"""Bản Linux/Android (proot) của hw.py, cho điện thoại. Cùng tên hàm với hw.py; `hw.py` tự chuyển sang module này khi không phải Windows.

Nguyên tắc (PREREG_E5_phone.md): KHÔNG giả định ghim core hoạt động như Linux gốc, KHÔNG tin số luồng torch tự báo.
Mọi kiểm tra bằng tải thật, không dựa vào /proc/stat (Android thường chặn). Không đọc gì ngoài tiến trình của chính mình, TRỪ khi
người dùng đã đồng ý bước A0 và đặt SPONGE_ALLOW_SYSFS=1 (khi đó mới đọc sysfs chỉ-đọc: cpu_capacity, tần số, nhiệt).

Profile: lấy từ biến môi trường SPONGE_PROFILES (JSON, vd '{"big1":[7],"little1":[0]}') do bước xác minh ghim tạo ra,
hoặc từ tệp pin_verify.json nếu SPONGE_PROFILES_FILE trỏ tới.
"""
from __future__ import annotations

import json
import os
import platform
import threading
import time

import psutil

ALLOW_SYSFS = os.environ.get('SPONGE_ALLOW_SYSFS') == '1'

CHECKLIST = """Checklist điện thoại (BẠN tự đặt; harness không đổi gì):
  [ ] Đang sạc, pin 30-80%, Không làm phiền BẬT, tiết kiệm pin TẮT.
  [ ] Mạng di động TẮT; Wi-Fi ở mức tối thiểu hoặc TẮT nếu SSH không cần liên tục (CLAUDE.md).
  [ ] Bật khóa thức (wake lock) của Termux; không dùng điện thoại trong lúc đo.
  [ ] Ghi lại trạng thái mạng bạn đã đặt vào tin nhắn xác nhận (không tự kiểm được).
"""


def cpu_topology() -> list[dict]:
    """Điện thoại: không có cách tin cậy để biết cụm từ bên trong proot. Trả về danh sách CPU mà tiến trình thấy;
    cụm (nhanh/chậm) chỉ được gán bởi `classify_clusters` sau khi đo tốc độ, hoặc từ cpu_capacity nếu được phép đọc."""
    cpus = sorted(os.sched_getaffinity(0))
    cores = [dict(core_index=i, logical_cpus=[c], efficiency_class=None, smt=False, kind='unknown') for i, c in enumerate(cpus)]
    if ALLOW_SYSFS:
        for c in cores:
            try:
                path = f'/sys/devices/system/cpu/cpu{c["logical_cpus"][0]}/cpu_capacity'
                c['cpu_capacity'] = int(open(path).read().strip())
            except Exception as e:  # SELinux/proot: ghi lại là không đọc được, không tìm đường vòng
                c['cpu_capacity'] = f'khong_doc_duoc:{type(e).__name__}'
    return cores


def profiles(cores=None) -> dict[str, list[int]]:
    if os.environ.get('SPONGE_PROFILES'):
        return {k: [int(x) for x in v] for k, v in json.loads(os.environ['SPONGE_PROFILES']).items()}
    f = os.environ.get('SPONGE_PROFILES_FILE')
    if f and os.path.exists(f):
        return {k: [int(x) for x in v] for k, v in json.load(open(f, encoding='utf-8')).items()}
    raise RuntimeError('Chưa có profile: chạy xác minh ghim (phone_pin_verify.py) rồi đặt SPONGE_PROFILES hoặc SPONGE_PROFILES_FILE')


def pin(cpus: list[int]) -> list[int]:
    """Ghim bằng sched_setaffinity; trả về affinity đọc lại. Nếu đọc lại khác yêu cầu thì ném lỗi (không giả vờ thành công)."""
    os.sched_setaffinity(0, set(cpus))
    back = sorted(os.sched_getaffinity(0))
    if back != sorted(cpus):
        raise RuntimeError(f'Ghim không có hiệu lực: yêu cầu {sorted(cpus)}, đọc lại {back}')
    return back


def _last_cpus_of_threads() -> list[int]:
    """CPU mà từng luồng của tiến trình chạy gần nhất (trường 39 của /proc/self/task/<tid>/stat). Chỉ đọc tiến trình của chính mình."""
    out = []
    for tid in os.listdir('/proc/self/task'):
        try:
            data = open(f'/proc/self/task/{tid}/stat').read()
            fields = data[data.rindex(')') + 2:].split()
            out.append(int(fields[36]))   # sau ')' trường 3 là chỉ số 0 -> trường 39 là chỉ số 36
        except Exception:
            pass
    return out


def verify_pin(cpus: list[int], seconds: float = 3.0, threads: int | None = None, min_effective: float | None = None) -> dict:
    """Kiểm bằng tải thật (torch matmul, 2×|cpus| luồng):
    1) effective_cores = CPU-time tiến trình / thời gian thực; <= |cpus|+0,15 và >= min_effective (nếu đặt);
    2) lấy mẫu mỗi 50 ms CPU chạy gần nhất của từng luồng; >= 99% mẫu thuộc `cpus`.
    Trả về dict có `passed`."""
    import torch
    n = threads or 2 * len(cpus)
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    a = torch.randn(512, 512)
    samples: list[int] = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            samples.extend(_last_cpus_of_threads())
            time.sleep(0.05)

    th = threading.Thread(target=sampler, daemon=True)
    proc = psutil.Process()
    t0, c0 = time.perf_counter(), sum(proc.cpu_times()[:2])
    th.start()
    while time.perf_counter() - t0 < seconds:
        a = torch.tanh(a @ a)
    wall, cpu = time.perf_counter() - t0, sum(proc.cpu_times()[:2]) - c0
    stop.set()
    th.join(1.0)
    torch.set_num_threads(old)
    eff = cpu / wall
    allowed = set(cpus)
    inside = sum(1 for c in samples if c in allowed) / max(len(samples), 1)
    upper_ok = eff <= len(cpus) + 0.15
    lower_ok = None if min_effective is None else eff >= min_effective
    share_ok = inside >= 0.99 and len(samples) >= 20
    return dict(allowed=sorted(cpus), threads=n, wall_s=wall, cpu_s=cpu, effective_cores=eff, samples=len(samples),
                share_samples_inside=inside, observed_cpus=sorted(set(samples)), upper_ok=upper_ok, lower_ok=lower_ok,
                min_effective=min_effective, share_ok=share_ok, passed=upper_ok and share_ok and (lower_ok is not False))


def speed_benchmark(cpu: int, repeats: int = 5) -> dict:
    """Ghim vào 1 CPU, chạy benchmark cố định (matmul 256x256 x20, 1 luồng), trả về trung vị ms và CV. Dùng để phân cụm nhanh/chậm."""
    import torch
    old_aff = sorted(os.sched_getaffinity(0))
    old_thr = torch.get_num_threads()
    try:
        pin([cpu])
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(0)
        m = torch.randn(256, 256, generator=g)
        def work():
            x = m
            for _ in range(20):
                x = torch.tanh(x @ m)
        work()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns(); work(); ts.append((time.perf_counter_ns() - t0) / 1e6)
        ts.sort()
        med = ts[len(ts) // 2]
        mean = sum(ts) / len(ts)
        sd = (sum((t - mean) ** 2 for t in ts) / (len(ts) - 1)) ** .5
        return dict(cpu=cpu, median_ms=med, cv=sd / mean, samples_ms=ts)
    finally:
        os.sched_setaffinity(0, set(old_aff))
        torch.set_num_threads(old_thr)


def classify_clusters(bench: list[dict], within: float = 0.15, between: float = 1.3) -> dict:
    """Phân cụm theo tốc độ: sắp theo median_ms; tách ở khoảng nhảy lớn nhất. Chấp nhận khi trong cụm chênh <= within
    và giữa hai cụm >= between lần; nếu không thì 'khong_xac_dinh'."""
    b = sorted(bench, key=lambda r: r['median_ms'])
    if len(b) < 2:
        return dict(status='khong_xac_dinh', reason='it_hon_2_cpu')
    gaps = [b[i + 1]['median_ms'] / b[i]['median_ms'] for i in range(len(b) - 1)]
    k = max(range(len(gaps)), key=lambda i: gaps[i])
    fast, slow = b[:k + 1], b[k + 1:]
    spread = lambda grp: max(r['median_ms'] for r in grp) / min(r['median_ms'] for r in grp) - 1
    ok = gaps[k] >= between and spread(fast) <= within and spread(slow) <= within
    return dict(status='ok' if ok else 'khong_xac_dinh', gap=gaps[k], fast_cpus=[r['cpu'] for r in fast],
                slow_cpus=[r['cpu'] for r in slow], spread_fast=spread(fast), spread_slow=spread(slow))


def power_snapshot() -> dict:
    """Chỉ thông tin của chính tiến trình/nền tảng; pin/nhiệt chỉ đọc khi được phép (SPONGE_ALLOW_SYSFS=1)."""
    snap = dict(platform=platform.platform(), machine=platform.machine(), logical=psutil.cpu_count(),
                physical=psutil.cpu_count(logical=False), affinity=sorted(os.sched_getaffinity(0)),
                network_state='NGUOI_DUNG_KHAI_BAO (khong tu kiem duoc)', power_plan='n/a (Android)')
    return snap


def read_thermal() -> dict:
    """Đọc mọi vùng nhiệt đọc được (chỉ khi người dùng đồng ý A0). Không đọc được -> trả {} và ghi lý do."""
    if not ALLOW_SYSFS:
        return {}
    out = {}
    base = '/sys/class/thermal'
    try:
        zones = sorted(z for z in os.listdir(base) if z.startswith('thermal_zone'))
    except Exception as e:
        return {'_loi': type(e).__name__}
    for z in zones:
        try:
            typ = open(f'{base}/{z}/type').read().strip()
            val = int(open(f'{base}/{z}/temp').read().strip())
            out[f'{z}:{typ}'] = val / 1000.0 if abs(val) > 1000 else float(val)
        except Exception:
            pass
    return out
