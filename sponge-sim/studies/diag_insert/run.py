"""Chẩn đoán (post hoc, 03/11/2026): thời gian của bước chèn patch `insert` mà phép đo toàn trình của E6/E6-A1 tính cho nhánh có patch
nhưng không tính cho nhánh sạch (studies/e6_remeasure/run.py, hàm full()). Đo trên cùng profile ghim lõi, cùng hàm insert, cùng kích thước.

    python run.py OUT_JSON
"""
import json
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))


def main():
    from sponge_sim import hw
    cpus = hw.profiles(hw.cpu_topology())['P1_cpu2']
    hw.pin(cpus)
    import torch
    torch.set_num_interop_threads(1)
    verify = hw.verify_pin(cpus, min_effective=0.5)
    torch.set_num_threads(1)
    from sponge_sim import attacks_provenance as ap_, timing
    out = dict(pin=dict(effective_cores=verify['effective_cores'], passed=verify['passed']), cases={})
    g = torch.Generator().manual_seed(0)
    for canvas, size in ((320, 64), (640, 128), (640, 64)):
        base = torch.rand(1, 3, canvas, canvas, generator=g)
        patch = torch.rand(3, size, size, generator=g)
        loc = (canvas // 3, canvas // 4)
        ns = [r['ns'] for r in timing.timed(ap_.insert, setup=lambda: (base, patch[None], [loc]), repeats=300, warmup=20)]
        ns.sort()
        out['cases'][f'{canvas}px_patch{size}'] = dict(median_ms=ns[len(ns) // 2] / 1e6, p10_ms=ns[len(ns) // 10] / 1e6, p90_ms=ns[9 * len(ns) // 10] / 1e6, n=len(ns))
    out['pin_end'] = hw.verify_pin(cpus, min_effective=0.5)['effective_cores']
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
