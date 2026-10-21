"""Đo thời gian CPU theo quy ước skill: perf_counter_ns, gc tắt trong vùng đo, warm-up bỏ,
timestamp từng phép đo, benchmark tham chiếu để bắt trôi xung nhịp, khối Latin square.
"""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch

DRIFT_FLAG = 0.03  # >3% so với tham chiếu đầu phiên -> gắn cờ


def timed(fn, setup=None, repeats: int = 1, warmup: int = 0) -> list[dict]:
    """Mỗi lần lặp: args = setup() (NGOÀI vùng đo), rồi đo fn(*args). Trả về list dict.
    gc tắt trong suốt vòng đo, bật lại sau."""
    setup = setup or (lambda: ())
    for _ in range(warmup):
        fn(*setup())
    out = []
    was = gc.isenabled()
    gc.disable()
    try:
        for r in range(repeats):
            args = setup()
            t0 = time.perf_counter_ns()
            result = fn(*args)
            t1 = time.perf_counter_ns()
            out.append(dict(rep=r, ns=t1 - t0, t_start_ns=t0, wall_utc=time.time(), result=result))
    finally:
        if was:
            gc.enable()
    return out


_REF_A = None


def reference_ms(repeats: int = 15) -> float:
    """Benchmark tham chiếu cố định (matmul 256×256, 20 lần) – trung vị ms. Dùng số thread hiện hành."""
    global _REF_A
    if _REF_A is None:
        g = torch.Generator().manual_seed(0)
        _REF_A = torch.randn(256, 256, generator=g)
    def work():
        a = _REF_A
        for _ in range(20):
            a = torch.tanh(a @ _REF_A)
        return None
    samples = sorted(x['ns'] for x in timed(work, repeats=repeats, warmup=3))
    return samples[len(samples) // 2] / 1e6


def drift(ref_now: float, ref_start: float) -> dict:
    d = ref_now / ref_start - 1
    return dict(ref_ms=ref_now, ref_start_ms=ref_start, rel_change=d, flagged=abs(d) > DRIFT_FLAG)


def latin_blocks(conditions: list, n_blocks: int, seed: int) -> list[list]:
    """Hình vuông Latin vòng (mỗi điều kiện xuất hiện ở mỗi vị trí như nhau qua k khối),
    thứ tự hàng và nhãn điều kiện được hoán vị ngẫu nhiên theo seed."""
    rng = random.Random(seed)
    k = len(conditions)
    labels = conditions[:]
    rng.shuffle(labels)
    rows = [[labels[(i + j) % k] for j in range(k)] for i in range(k)]
    blocks = []
    while len(blocks) < n_blocks:
        batch = rows[:]
        rng.shuffle(batch)
        blocks += batch
    return blocks[:n_blocks]


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open('a', encoding='utf-8')

    def write(self, row: dict):
        self.fh.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        self.fh.flush()

    def close(self):
        self.fh.close()
