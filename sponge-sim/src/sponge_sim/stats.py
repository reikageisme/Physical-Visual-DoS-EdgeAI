"""Thống kê theo quy ước skill sponge-research. Không import helper thống kê của repo cũ.

- bootstrap percentile 20.000 lần, seed RNG truyền vào và phải ghi lại;
- sign-flip chính xác (n <= 20), tolerance 1e-12 (giống receipt REVIEW/03 §6);
- d_z = mean / SD(ddof=1);
- bootstrap hai tầng (cụm ảnh × seed) cho ma trận [ảnh, seed];
- Holm step-down; MDE cặp (xấp xỉ t, 80% công suất).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats as sps

N_BOOT = 20_000


def bootstrap_ci(x, seed: int, n_boot: int = N_BOOT, level: float = .95, stat=np.mean):
    """CI percentile, đơn vị lấy mẫu lại = phần tử của x. Cách rút giống REVIEW/03 §6."""
    x = np.asarray(x, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))
    values = stat(x[draws], axis=1)
    a = (1 - level) / 2
    return np.quantile(values, [a, 1 - a])


def sign_flip_p(x, tol: float = 1e-12) -> float:
    """p hai phía chính xác của kiểm định đảo dấu. Với n=10, p nhỏ nhất = 2/1024 (mọi seed cùng dấu)."""
    x = np.asarray(x, dtype=float)
    if len(x) > 20:
        raise ValueError('sign_flip_p chính xác chỉ dùng cho n <= 20')
    obs = abs(x.mean())
    hits = sum(abs(np.mean(x * np.asarray(s))) >= obs - tol
               for s in itertools.product((-1, 1), repeat=len(x)))
    return hits / 2 ** len(x)


def dz(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1))


def paired_summary(x, seed: int) -> dict:
    x = np.asarray(x, dtype=float)
    lo, hi = bootstrap_ci(x, seed)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)), ci95=[float(lo), float(hi)],
                p_sign_flip=sign_flip_p(x) if len(x) <= 20 else None, dz=dz(x),
                all_same_sign=bool((x > 0).all() or (x < 0).all()), bootstrap_seed=seed, n_boot=N_BOOT)


def two_level_bootstrap(matrix, seed: int, n_boot: int = N_BOOT, level: float = .95, stat=np.nanmean):
    """matrix [ảnh, seed]: rút lại ảnh (cụm) rồi rút lại seed (cột), trả CI của stat toàn ma trận.
    Phủ cả độ bất định của ảnh lẫn seed (khác CI chỉ theo seed)."""
    m = np.asarray(matrix, dtype=float)
    rng = np.random.default_rng(seed)
    ni, ns = m.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        rows = rng.integers(0, ni, ni)
        cols = rng.integers(0, ns, ns)
        out[b] = stat(m[np.ix_(rows, cols)])
    a = (1 - level) / 2
    return np.quantile(out, [a, 1 - a])


def cluster_bootstrap(values, clusters, seed: int, n_boot: int = N_BOOT, level: float = .95, fn=None):
    """Bootstrap cụm: rút lại cụm (vd. ảnh), giữ nguyên mọi quan sát trong cụm.
    fn(idx) -> thống kê trên chỉ số quan sát; mặc định là mean(values)."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in uniq]
    fn = fn or (lambda idx: values[idx].mean())
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[b] = fn(np.concatenate([members[k] for k in pick]))
    a = (1 - level) / 2
    return np.quantile(out, [a, 1 - a])


def spearman(x, y) -> float:
    return float(sps.spearmanr(x, y).statistic)


def holm(pvalues) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind='stable')
    adj, prev = np.empty(len(p)), 0.
    for rank, i in enumerate(order):
        prev = max(prev, min(1., (len(p) - rank) * p[i]))
        adj[i] = prev
    return adj.tolist()


def mde_paired(sd: float, n: int, alpha: float = .05, power: float = .8) -> float:
    """MDE hai phía cho trung bình chênh lệch ghép cặp: (t_{1-a/2,n-1} + t_{power,n-1}) * sd / sqrt(n).
    Xấp xỉ t trung tâm; ghi rõ là xấp xỉ khi báo."""
    df = n - 1
    return float((sps.t.ppf(1 - alpha / 2, df) + sps.t.ppf(power, df)) * sd / np.sqrt(n))
