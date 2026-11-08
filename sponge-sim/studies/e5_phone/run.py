"""E5-phone, phần B: đo A/A theo ĐỢT NGẮN, nghỉ giữa đợt, XEN KẼ profile. Xem prereg/PREREG_E5_phone.md.

Điều phối (một lệnh, chạy nền, log trong ~/sponge-sim):
    SPONGE_PROFILES_FILE=results/phone_a36/pin_<id>/profiles.json SPONGE_REPO_ROOT=$PWD/phone_root \\
    python sponge-sim/studies/e5_phone/run.py --run-dir results/phone_a36/<id> --network-state "di dong TAT; Wi-Fi TAT"
Mỗi đợt = 10 ảnh (1 warm-up + 20 lần A/B ABBA/BAAB mỗi ảnh); sau đợt nghỉ --rest-s giây (và chờ nhiệt hạ nếu đọc được).
Thứ tự profile trong từng đợt theo hình vuông Latin (seed 5031). Mỗi worker (một profile, một đợt) là một tiến trình riêng, ghim trước
khi torch tạo thread pool. Tham chiếu xung nhịp (matmul 256x256 x20) ở đầu và cuối mỗi đợt; cờ nếu lệch > 3% so với tham chiếu đầu phiên.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
sys.path.insert(0, str(SIM / 'studies' / 'e5_protocol'))

CONFIG = dict(experiment='E5-phone', burst_images=10, repeats=10, warmup_per_image=1, n_images=100,
              tau=.01, cap=None, iou=.45, interop_threads=1, order_seed=5031, label_order_seed=5032,
              min_effective={'default': 0.5}, drift_flag=0.03, rest_s=60, rest_max_s=180, thermal_period_s=5,
              stop_temp_c=70.0, stop_resume_c=55.0)


def worker(profile: str, burst: int, run_dir: Path, cfg: dict):
    from run import IMAGE_IDS            # cùng 100 ảnh với E5 laptop (studies/e5_protocol/run.py)
    from sponge_sim import hw
    cpus = hw.profiles(hw.cpu_topology())[profile]
    readback = hw.pin(cpus)
    import torch
    torch.set_num_interop_threads(cfg['interop_threads'])
    torch.set_num_threads(len(cpus))
    from ultralytics import YOLO
    from sponge_sim import provenance as pv, timing
    from sponge_sim.backends import BACKENDS
    from sponge_sim.cohorts import REPO
    data = REPO / 'outputs/datasets/coco_val2017_extended'
    names = {im['id']: im['file_name'] for im in
             json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
    model = YOLO(str(REPO / 'yolov8n.pt')).model.eval().float()
    ids = IMAGE_IDS[:cfg['n_images']][burst * cfg['burst_images']:(burst + 1) * cfg['burst_images']]
    w = timing.JsonlWriter(run_dir / f'raw_{profile}.jsonl')
    ref_file = run_dir / f'ref0_{profile}.json'
    ref_now = timing.reference_ms()
    if not ref_file.exists():
        ref_file.write_text(json.dumps(dict(ref_ms=ref_now, wall_utc=time.time())), encoding='utf-8')
    ref0 = json.loads(ref_file.read_text(encoding='utf-8'))['ref_ms']
    w.write(dict(kind='burst_start', profile=profile, burst=burst, allowed=cpus, readback=readback,
                 torch_threads=torch.get_num_threads(), wall_utc=time.time(),
                 **timing.drift(ref_now, ref0)))
    tp, tv = BACKENDS['torch_pure'], BACKENDS['tv_nms']
    rng = random.Random(cfg['label_order_seed'] + burst * 101 + cpus[0])

    def total(p):
        im = pv.load_image(p)
        r = model(im)
        r = r[0] if isinstance(r, (tuple, list)) else r
        act, _ = pv.tau_filter(pv.decode(r[0]), cfg['tau'], cfg['cap'])
        return tp.run((act[:, :4], act[:, 4]), cfg['iou'])

    def one_rep(path):
        out = {}
        def t(name, fn, *args):
            r = timing.timed(fn, setup=lambda: args, repeats=1)[0]
            out[name] = r['ns']
            return r['result']
        with torch.no_grad():
            image = t('pre', pv.load_image, path)
            raw = t('forward', model, image)
            raw = raw[0] if isinstance(raw, (tuple, list)) else raw
            decoded = t('decode', pv.decode, raw[0])
            active, n_tau = t('tau', pv.tau_filter, decoded, cfg['tau'], cfg['cap'])
            boxes, scores, classes = active[:, :4], active[:, 4], active[:, 5]
            t('nms_torch_pure', tp.run, tp.prepare(boxes, scores, classes), cfg['iou'])
            t('nms_tv_nms', tv.run, tv.prepare(boxes, scores, classes), cfg['iou'])
            t('total', total, path)
        return out, n_tau

    for image_id in ids:
        path = data / 'images' / names[image_id]
        for _ in range(cfg['warmup_per_image']):
            one_rep(path)
        seq = ['A', 'B', 'B', 'A'] if rng.random() < .5 else ['B', 'A', 'A', 'B']
        seq = (seq * (2 * cfg['repeats'] // 4 + 1))[:2 * cfg['repeats']]
        for k, label in enumerate(seq):
            stages, n_tau = one_rep(path)
            w.write(dict(kind='measure', profile=profile, burst=burst, image_id=image_id, label=label, seq_index=k,
                         n_tau=n_tau, stages_ns=stages, wall_utc=time.time()))
    w.write(dict(kind='burst_end', profile=profile, burst=burst, wall_utc=time.time(),
                 **timing.drift(timing.reference_ms(), ref0)))
    w.close()


class ThermalLogger:
    """Ghi nhiệt mỗi cfg['thermal_period_s'] giây khi đọc được (SPONGE_ALLOW_SYSFS=1); giữ lần đọc gần nhất."""
    def __init__(self, path: Path, period: float):
        from sponge_sim import hw
        self.read = getattr(hw, 'read_thermal', None) or __import__('sponge_sim.hw_linux', fromlist=['x']).read_thermal
        self.path, self.period, self.last, self.stop = path, period, {}, threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        with open(self.path, 'a', encoding='utf-8') as f:
            while not self.stop.is_set():
                self.last = self.read()
                f.write(json.dumps(dict(wall_utc=time.time(), temps=self.last)) + '\n')
                f.flush()
                self.stop.wait(self.period)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--profiles', default='', help='mặc định: mọi profile trong SPONGE_PROFILES(_FILE)')
    ap.add_argument('--network-state', default='CHUA_KHAI_BAO')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--worker')
    ap.add_argument('--burst', type=int, default=0)
    ap.add_argument('--rest-s', type=int, default=0)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = dict(CONFIG)
    if a.smoke:
        cfg.update(n_images=4, burst_images=2, repeats=2, rest_s=2, rest_max_s=4)
    if a.rest_s:
        cfg['rest_s'] = a.rest_s
    if a.worker:
        worker(a.worker, a.burst, run_dir, cfg)
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    from sponge_sim import hw, manifest, timing
    all_profiles = hw.profiles(hw.cpu_topology())
    names = [p for p in (a.profiles.split(',') if a.profiles else list(all_profiles)) if p]
    n_bursts = cfg['n_images'] // cfg['burst_images']
    order = timing.latin_blocks(names, n_bursts, cfg['order_seed'])
    manifest.write(run_dir / 'manifest.json', manifest.build(dict(cfg, smoke=a.smoke, profiles=names, burst_order=order,
                                                                    network_state_declared_by_user=a.network_state)))
    th = ThermalLogger(run_dir / 'thermal.jsonl', cfg['thermal_period_s'])
    th.start()
    t0_temp = None
    for b in range(n_bursts):
        for prof in order[b]:
            cmd = [sys.executable, '-X', 'utf8', __file__, '--run-dir', str(run_dir), '--worker', prof, '--burst', str(b)]
            if a.smoke:
                cmd.append('--smoke')
            t1 = time.time()
            subprocess.run(cmd, check=True)
            print(f'[E5-phone] đợt {b} profile {prof}: {time.time() - t1:.0f}s', flush=True)
            # nghỉ giữa đợt
            if t0_temp is None and th.last:
                t0_temp = max(v for v in th.last.values() if isinstance(v, (int, float)))
            rested = 0
            while rested < cfg['rest_s']:
                time.sleep(1); rested += 1
            if th.last and t0_temp is not None:
                while rested < cfg['rest_max_s']:
                    cur = max(v for v in th.last.values() if isinstance(v, (int, float)))
                    if cur <= t0_temp + 1.0:
                        break
                    time.sleep(2); rested += 2
                cur = max(v for v in th.last.values() if isinstance(v, (int, float)))
                with open(run_dir / 'rests.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(burst=b, profile=prof, rested_s=rested, max_temp=cur, t0_temp=t0_temp,
                                            not_cooled=cur > t0_temp + 1.0)) + '\n')
                if cur >= cfg['stop_temp_c']:
                    print('[E5-phone] DỪNG AN TOÀN: nhiệt', cur, '>=', cfg['stop_temp_c'], flush=True)
                    while max(v for v in th.last.values() if isinstance(v, (int, float))) > cfg['stop_resume_c']:
                        time.sleep(10)
    th.close()
    print('[E5-phone] DONE', flush=True)


if __name__ == '__main__':
    main()
