"""Finite forward-only diagnostic; no attack, NMS or mask measurements.

Reuses source-bound process/QoS/pinning/timer/PDH helpers from the laptop
harness. Each cadence gets exactly one fresh worker. No quality promotion,
adaptive warmup, replacement worker or automatic retry is implemented.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from run import REPO, SIM, load_lock, power_mode, sha


def validate(cfg):
    expected = dict(blocks=12, samples=15, warmup=3, pin_seconds=3, settle_s=60,
                    worker_timeout_s=600, seed=20261107,
                    pauses_s={'continuous': 0, 'pause2': 2})
    if cfg != expected:
        raise ValueError('exact finite diagnostic configuration required')
    order = list(expected['pauses_s'])
    random.Random(cfg['seed']).shuffle(order)
    return order


def trace_blocks(cfg, condition, work, measure, collect, check, emit, flush, wait):
    """Observers run outside model clocks; every observation is preserved."""
    validate(cfg)
    if condition not in cfg['pauses_s']:
        raise ValueError('unknown cadence')
    baseline = None
    for block in range(cfg['blocks']):
        check()
        for _ in range(cfg['warmup']):
            work()
        emit(kind='pdh_block_initial', block=block, sample=collect())
        samples = []
        for sample in range(cfg['samples']):
            ns, start, _, cpu_ns = measure(work, capture_cpu=True, keep_output=False)
            pdh = collect()
            samples.append(ns)
            emit(kind='timing', block=block, sample=sample, ns=ns,
                 t_start_ns=start, process_cpu_ns=cpu_ns, pdh_after=pdh)
        median = statistics.median(samples)
        if baseline is None:
            baseline = median
        emit(kind='reference', block=block, median_ns=median, initial_ratio=median/baseline,
             diagnostic_flag_3pct=abs(median/baseline-1) > .03)
        flush()
        if block+1 < cfg['blocks']:
            wait(cfg['pauses_s'][condition])


def worker(args):
    lock = load_lock(args.lock)
    cfg = lock['config']
    validate(cfg)
    import os
    import psutil
    proc = psutil.Process()
    proc.cpu_affinity([2])
    if proc.cpu_affinity() != [2]:
        raise RuntimeError('initial affinity readback failed')
    proc.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    import windows_qos as qos
    change = qos.set_high()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO
    from output_lifetime import measure
    from pdh_cpu import Query
    sys.path.insert(0, str(SIM/'src'))
    from sponge_sim import hw
    if (torch.__version__, torchvision.__version__, ultralytics.__version__) != tuple(
            lock['versions'][k] for k in ('torch', 'torchvision', 'ultralytics')):
        raise RuntimeError('runtime version mismatch')
    torch.manual_seed(20261107)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (torch.get_num_threads(), torch.get_num_interop_threads()) != (1, 1):
        raise RuntimeError('thread count readback failed')
    def check():
        if (hw.power_snapshot()['power_plugged'] is not True or power_mode() != lock['power_mode']
                or proc.cpu_affinity() != [2] or proc.nice() != psutil.ABOVE_NORMAL_PRIORITY_CLASS
                or not qos.high_readback_ok(qos.snapshot())):
            raise RuntimeError('AC/power/affinity/priority/QoS condition changed')
    check()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(condition=args.condition, config=cfg, cpu_set=[2], lock_sha256=sha(args.lock),
                    tag=args.tag, prereg_commit=args.commit, versions=lock['versions'],
                    power_start=hw.power_snapshot(), power_mode_start=power_mode(),
                    qos_change=change, priority_start=int(proc.nice()),
                    torch_threads=1, interop_threads=1, parallel_info=torch.__config__.parallel_info(),
                    hardware=hw.cpu_topology(), thermal_telemetry='unavailable',
                    inputs=lock['files'], endpoint='net(clean_tensor), FP32 unfused, output release after clocks')
    (args.run_dir/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    net = YOLO(str(REPO/lock['weights_path'])).model.eval().float().cpu()
    for p in net.parameters():
        p.requires_grad_(False)
    image = torch.load(REPO/lock['input_path'], map_location='cpu', weights_only=True)['clean'].contiguous()
    if (tuple(image.shape) != (1, 3, 640, 640) or image.dtype != torch.float32
            or not torch.isfinite(image).all()):
        raise RuntimeError('clean input shape/dtype/finite check failed')
    fh = (args.run_dir/'raw.jsonl').open('x', encoding='utf-8')
    pending = []
    def emit(**r):pending.append(dict(wall_utc=time.time(), **r))
    def flush():
        for row in pending:
            fh.write(json.dumps(row, allow_nan=False)+'\n')
        fh.flush()
        pending.clear()
    def verify(kind):
        result = hw.verify_pin([2], seconds=cfg['pin_seconds'], threads=2, min_effective=.5)
        torch.set_num_threads(1)
        emit(kind=kind, verify=result, affinity=proc.cpu_affinity())
        flush()
        if not result['passed']:
            raise RuntimeError('pin-load gate failed')
    pdh = None
    try:
        with torch.inference_mode():
            verify('pin_start')
            pdh = Query(2)
            trace_blocks(cfg, args.condition, lambda:net(image), measure, pdh.collect,
                         check, emit, flush, time.sleep)
            verify('pin_end')
        check()
        emit(kind='done', affinity=proc.cpu_affinity(), power_end=hw.power_snapshot(),
             power_mode_end=power_mode(), qos_end=qos.snapshot(), priority_end=int(proc.nice()))
    except BaseException as e:
        emit(kind='aborted', error=type(e).__name__, message=str(e))
        raise
    finally:
        flush()
        fh.close()
        if pdh is not None:pdh.close()


def campaign(args):
    from separate_app import Separation
    lock = load_lock(args.lock)
    order = validate(lock['config'])
    # Verify the registration identity before affinity changes or workers.
    tagged = subprocess.check_output(['git', 'rev-parse', args.tag+'^{commit}'], cwd=REPO, text=True).strip()
    if tagged != args.commit:
        raise RuntimeError('tag/commit identity mismatch')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir/'campaign.json').write_text(json.dumps(dict(
        tag=args.tag, commit=args.commit, lock_sha256=sha(args.lock), order=order,
        user_idle=args.user_idle), indent=2), encoding='utf-8')
    sep = Separation(args.run_dir.with_name(args.run_dir.name+'_affinity.jsonl'))
    statuses = []
    try:
        sep.apply()
        for condition in order:
            time.sleep(lock['config']['settle_s'])
            cmd = [sys.executable, __file__, '--lock', str(args.lock), '--run-dir', str(args.run_dir/condition),
                   '--tag', args.tag, '--commit', args.commit, '--condition', condition]
            with (args.run_dir/f'{condition}.log').open('x', encoding='utf-8') as log:
                try:
                    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=lock['config']['worker_timeout_s'])
                    statuses.append(dict(condition=condition, returncode=r.returncode))
                except subprocess.TimeoutExpired:
                    statuses.append(dict(condition=condition, timeout=True))
            if statuses[-1].get('returncode') != 0:
                break  # infrastructure/environment failure, never a replacement worker
    finally:
        sep.restore()
    (args.run_dir/'campaign_done.json').write_text(json.dumps(statuses, indent=2), encoding='utf-8')
    print('REFERENCE_TRACE_FINISHED', json.dumps(statuses), 'APP_AFFINITY_RESTORED', flush=True)
    if len(statuses) != 2 or any(s.get('returncode') != 0 for s in statuses):
        raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lock', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--condition', choices=['continuous', 'pause2'])
    ap.add_argument('--user-idle')
    args = ap.parse_args()
    if args.condition:worker(args)
    elif args.user_idle:campaign(args)
    else:ap.error('standing idle declaration required')
