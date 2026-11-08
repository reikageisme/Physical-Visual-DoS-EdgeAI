"""E5-phone, phần A: xác minh ghim core qua proot bằng tải thật. Xem prereg/PREREG_E5_phone.md.

Chạy trong Ubuntu proot trên điện thoại, trong ~/sponge-sim:
    python sponge-sim/studies/e5_phone/pin_verify.py --out results/phone_a36/pin_<id>
Việc làm: (1) cơ chế ghim: sched_setaffinity từng CPU, đọc lại; (2) benchmark tốc độ từng CPU -> phân cụm nhanh/chậm;
(3) với mỗi profile ứng viên, verify_pin bằng tải thật (effective_cores + CPU chạy gần nhất của từng luồng).
Ghi pin_verify.json và profiles.json (chỉ gồm profile ĐẠT). KHÔNG đọc gì ngoài tiến trình của chính mình, trừ khi SPONGE_ALLOW_SYSFS=1
(bước A0 đã được người dùng đồng ý). Nếu ghim không có hiệu lực: ghi nhận, không tạo profile, thoát mã 2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=5, help='số lần benchmark mỗi CPU')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from sponge_sim import hw_linux as hw
    import torch
    res = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), torch=torch.__version__,
               torch_default_threads=torch.get_num_threads(), cpus_seen=sorted(os.sched_getaffinity(0)),
               allow_sysfs=hw.ALLOW_SYSFS, topology=hw.cpu_topology())
    # 1) cơ chế ghim
    mech = {}
    for c in res['cpus_seen']:
        try:
            mech[c] = dict(ok=True, readback=hw.pin([c]))
        except Exception as e:
            mech[c] = dict(ok=False, error=f'{type(e).__name__}: {e}')
        finally:
            os.sched_setaffinity(0, set(res['cpus_seen']))
    res['pin_mechanism'] = mech
    if not any(m['ok'] for m in mech.values()):
        res['conclusion'] = 'KHONG_GHIM_DUOC: sched_setaffinity khong co hieu luc; khong tao profile ghim; khong ket luan ve so core'
        (out / 'pin_verify.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
        print(res['conclusion'])
        raise SystemExit(2)
    # 2) benchmark từng CPU + phân cụm
    bench = [hw.speed_benchmark(c, a.repeats) for c in res['cpus_seen'] if mech[c]['ok']]
    res['speed_benchmark'] = bench
    res['clusters'] = hw.classify_clusters(bench)
    # 3) profile ứng viên: 1 CPU ổn định nhất (CV nhỏ nhất) mỗi cụm; cụm 4 CPU chỉ thêm nếu song song đạt
    cand = {}
    cl = res['clusters']
    by_cpu = {b['cpu']: b for b in bench}
    if cl['status'] == 'ok':
        for name, group in (('big', cl['fast_cpus']), ('little', cl['slow_cpus'])):
            best = min(group, key=lambda c: by_cpu[c]['cv'])
            cand[f'{name}1'] = [best]
            if len(group) >= 4:
                cand[f'{name}4'] = sorted(group)[:4]
    else:
        # cụm không xác định: chỉ dùng từng CPU riêng lẻ (nhanh nhất và chậm nhất), gọi tên theo CPU, không gọi là "lõi hiệu năng"
        fast = min(bench, key=lambda b: b['median_ms'])['cpu']
        slow = max(bench, key=lambda b: b['median_ms'])['cpu']
        cand[f'cpu{fast}'] = [fast]
        if slow != fast:
            cand[f'cpu{slow}'] = [slow]
    verified = {}
    res['verify'] = {}
    for name, cpus in cand.items():
        hw.pin(cpus)
        v = hw.verify_pin(cpus, seconds=3.0, min_effective=0.5 if len(cpus) == 1 else 0.7 * len(cpus))
        os.sched_setaffinity(0, set(res['cpus_seen']))
        res['verify'][name] = v
        if v['passed']:
            verified[name] = cpus
    res['profiles_passed'] = verified
    res['conclusion'] = 'ok' if verified else 'KHONG_CO_PROFILE_DAT'
    (out / 'pin_verify.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    (out / 'profiles.json').write_text(json.dumps(verified), encoding='utf-8')
    print(json.dumps(dict(clusters=res['clusters'], profiles_passed=verified,
                          verify={k: (round(v['effective_cores'], 2), round(v['share_samples_inside'], 3), v['passed'])
                                  for k, v in res['verify'].items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
