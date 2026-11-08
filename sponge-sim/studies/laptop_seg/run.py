"""CPU-only smoke/A-A, fresh process per profile. No attack optimization.

Design derived from studies/stability_pipeline/run.py and m2_confirm/run.py;
original file SHA256s are bound in lock.json. Old runners/raw are untouched.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

SIM = Path(__file__).resolve().parents[2]
REPO = SIM.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def power_mode():
    out = {}
    dll = ctypes.WinDLL('powrprof')
    for name in ('PowerGetUserConfiguredACPowerMode', 'PowerGetEffectiveOverlayScheme'):
        buf = (ctypes.c_ubyte*16)()
        fn = getattr(dll, name)
        fn.argtypes, fn.restype = [ctypes.c_void_p], ctypes.c_ulong
        code = fn(ctypes.byref(buf))
        if code:
            raise RuntimeError(f'{name} failed: {code}')
        out[name] = str(uuid.UUID(bytes_le=bytes(buf)))
    return out


def aa_order(rep, image_index):
    stages = ['forward', 'total']
    if (rep//2+image_index) % 2:
        stages.reverse()
    return [(s, label) for s in stages for label in
            (['A', 'B'] if (rep+image_index+stages.index(s)) % 2 == 0 else ['B', 'A'])]


def load_lock(path):
    lock = json.loads(path.read_text())
    for p, expected in lock['source_sha256'].items():
        if sha(REPO/p) != expected:
            raise RuntimeError(f'locked source changed: {p}')
    for row in lock['files']:
        if sha(REPO/row['path']) != row['sha256']:
            raise RuntimeError(f'locked input changed: {row["path"]}')
    return lock


def worker(args):
    lock = load_lock(args.lock)
    cfg = lock['config']
    if cfg.get('isolated_stage') and cfg.get('defense_comparison'):
        raise RuntimeError('component and defense modes are mutually exclusive')
    cpus = lock['profiles'][args.profile]
    import psutil
    proc = psutil.Process()
    proc.cpu_affinity(cpus)              # before torch imports / thread pools
    assert sorted(proc.cpu_affinity()) == cpus
    priority_before = int(proc.nice())
    if cfg.get('priority') == 'above_normal':
        proc.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        if proc.nice() != psutil.ABOVE_NORMAL_PRIORITY_CLASS:
            raise RuntimeError('Above-normal priority readback failed')
    qos = None
    qos_change = None
    if cfg.get('qos') == 'high':
        import windows_qos as qos
        qos_change = qos.set_high()
    os.environ['OMP_NUM_THREADS'] = str(len(cpus))
    os.environ['MKL_NUM_THREADS'] = str(len(cpus))
    os.environ['OPENBLAS_NUM_THREADS'] = str(len(cpus))
    import torch
    torch.manual_seed(20261107)
    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    import torchvision                 # fixed compiled NMS, before Ultralytics
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import nms, ops
    import numpy as np
    sys.path.insert(0, str(SIM/'src'))
    from sponge_sim import hw
    if (torch.__version__, torchvision.__version__) != tuple(lock['versions'][k] for k in ('torch', 'torchvision')):
        raise RuntimeError('runtime version changed')
    if ultralytics.__version__ != lock['versions']['ultralytics']:
        raise RuntimeError('Ultralytics version changed')
    if sha(nms.__file__) != lock['nms_source_sha256'] or sha(ops.__file__) != lock['ops_source_sha256']:
        raise RuntimeError('runtime source changed')
    assert 'torchvision.ops.nms' in Path(nms.__file__).read_text(encoding='utf-8')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    power0 = hw.power_snapshot()
    if power0['power_plugged'] is not True:
        raise RuntimeError('AC power required')
    if power_mode() != lock['power_mode']:
        raise RuntimeError('power mode changed since lock')
    manifest = dict(experiment=cfg.get('experiment', 'LAPTOP-SEG-SMOKE'),
                    smoke=cfg.get('smoke', True), profile=args.profile,
                    cpu_set=cpus, config=cfg, lock_sha256=sha(args.lock),
                    priority_before=priority_before,priority_start=int(proc.nice()),
                    tag=args.tag, prereg_commit=args.commit, hardware=hw.cpu_topology(),
                    power_start=power0, versions=lock['versions'],
                    power_mode_start=power_mode(),
                    torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                    parallel_info=torch.__config__.parallel_info(),
                    qos_change=qos_change, qos_start=None if qos is None else qos.snapshot(),
                    process_cpu_clock=vars(time.get_clock_info('process_time')),
                    inputs=[r for r in lock['files'] if r['path'].endswith('_inputs.pt')],
                    thermal_telemetry='unavailable; no inference about actual CPU temperature',
                    backend='torchvision.ops.nms inside locked Ultralytics wrapper',
                    omp={k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    (args.run_dir/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    fh = (args.run_dir/'raw.jsonl').open('x', encoding='utf-8')
    pending = []
    def write(**row):
        pending.append(dict(wall_utc=time.time(), **row))
    def flush():
        for row in pending:
            fh.write(json.dumps(row, allow_nan=False)+'\n')
        fh.flush()
        pending.clear()
    def timed(fn):
        if cfg.get('release_output',False):
            from output_lifetime import measure
            return measure(fn,cfg.get('capture_process_cpu',False),keep_output=False)
        enabled = gc.isenabled()
        gc.disable()
        try:
            c0 = time.process_time_ns() if cfg.get('capture_process_cpu',False) else None
            t0 = time.perf_counter_ns()
            result = fn()
            ns = time.perf_counter_ns()-t0
            cpu_ns = None if c0 is None else time.process_time_ns()-c0
            return ns, t0, result, cpu_ns
        finally:
            if enabled:
                gc.enable()
    def verify(label):
        v = hw.verify_pin(cpus, seconds=cfg['pin_seconds'], threads=2*len(cpus),
                          min_effective=.5 if len(cpus)==1 else .7*len(cpus))
        torch.set_num_threads(len(cpus))
        write(kind=label, verify=v, affinity=proc.cpu_affinity())
        flush()
        if not v['passed']:
            raise RuntimeError('pin load gate failed')
    net = YOLO(str(REPO/lock['weights_path'])).model.eval().float().cpu()
    for p in net.parameters():
        p.requires_grad_(False)
    paths = [REPO/p for p in lock['input_paths'][:cfg['n_images']]]
    frozen = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    images = [p['clean'].contiguous() for p in frozen]
    attacked = None
    if cfg.get('attack_replay',False):
        attacked=[p['attacked'].contiguous() for p in frozen]
        for clean,adv in zip(images,attacked):
            if (adv.shape!=clean.shape or adv.dtype!=torch.float32 or not torch.isfinite(adv).all()
                    or adv.min()<0 or adv.max()>1 or (adv-clean).abs().max()>16/255+1e-6):
                raise RuntimeError('frozen input budget/shape/range check failed')
    for image in images:
        assert image.shape == (1, 3, 640, 640) and image.dtype == torch.float32 and torch.isfinite(image).all()
    pdh = None
    if cfg.get('capture_pdh',False):
        from pdh_cpu import Query
        pdh = Query(cpus[0])
    def tensors(raw):
        if torch.is_tensor(raw):
            yield raw
        elif isinstance(raw, (tuple, list)):
            for x in raw:
                yield from tensors(x)
        elif isinstance(raw, dict):
            for x in raw.values():
                yield from tensors(x)
    def pred_proto(raw):
        ts = list(tensors(raw))
        pred = next(t for t in ts if t.ndim == 3 and t.shape[1] == 116)
        proto = next(t for t in ts if t.ndim == 4 and t.shape[1] == 32)
        return pred, proto[0]
    def suppress(pred, max_det=300, max_nms=30000):
        return nms.non_max_suppression(pred, conf_thres=.25, iou_thres=.45,
                                       nc=80, max_det=max_det, max_nms=max_nms)[0]
    def full(image, max_det=300, max_nms=30000):
        pred, proto = pred_proto(net(image))
        det = suppress(pred, max_det=max_det, max_nms=max_nms)
        if not len(det):
            return None
        return ops.process_mask_native(proto, det[:, 6:], det[:, :4], (640, 640))
    reference0 = None
    def reference(block):
        nonlocal reference0
        for _ in range(cfg['reference_warmup']):
            net(images[0])
        samples, cpu_samples = [], []
        for _ in range(cfg['reference_samples']):
            ns, _, result, cpu_ns = timed(lambda: net(images[0]))
            samples.append(ns)
            cpu_samples.append(cpu_ns)
            del result
        median = float(np.median(samples))
        reference0 = median if reference0 is None else reference0
        write(kind='reference', block=block, samples_ns=samples,
              cpu_samples_ns=cpu_samples, ratio=median/reference0,
              flagged=abs(median/reference0-1) > .03)
    try:
        with torch.inference_mode():
            verify('pin_start')
            warm_stage=None
            if cfg.get('isolated_stage'):
                from stage_replay import jobs as stage_jobs
                warm_stage,warm_check=stage_jobs(net,images[0],attacked[0],pred_proto,suppress,ops,
                                                 torchvision,lambda **row:None,paths[0].stem,cfg)
            for _ in range(cfg['global_warmup']):
                if warm_stage is not None:
                    warm_stage[0][2]();warm_stage[2][2]()
                elif attacked is None:
                    net(images[0])
                elif cfg.get('defense_comparison'):
                    setting=cfg['defense_comparison']
                    full(attacked[0]);full(attacked[0],max_det=setting['max_det'],max_nms=setting['max_nms'])
                else:
                    full(images[0]);full(attacked[0])
            if warm_stage is not None:
                warm_check()
                del warm_stage,warm_check
            if pdh is not None:
                write(kind='pdh_initial',sample=pdh.collect())
            # Same-input duplicate labels are A/A, not an attack comparison.
            reference('start')
            for ix, (path, image) in enumerate(zip(paths, images)):
                if hw.power_snapshot()['power_plugged'] is not True:
                    raise RuntimeError('AC power lost')
                if sorted(proc.cpu_affinity()) != cpus:
                    raise RuntimeError('affinity changed')
                if power_mode() != lock['power_mode']:
                    raise RuntimeError('power mode changed in run')
                if qos is not None and not qos.high_readback_ok(qos.snapshot()):
                    raise RuntimeError('process QoS changed in run')
                if cfg.get('priority')=='above_normal' and proc.nice()!=psutil.ABOVE_NORMAL_PRIORITY_CLASS:
                    raise RuntimeError('priority changed in run')
                reference(ix)
                if attacked is not None:
                    from attack_replay import jobs,balanced_orders,total_orders,carryover_orders
                    total_only=cfg.get('total_only',False)
                    stage_check=None
                    if cfg.get('isolated_stage'):
                        work,stage_check=stage_jobs(net,image,attacked[ix],pred_proto,suppress,ops,
                                                   torchvision,write,path.stem,cfg)
                    elif cfg.get('defense_comparison'):
                        from defense_replay import jobs as defense_jobs
                        work=defense_jobs(net,attacked[ix],pred_proto,suppress,full,torchvision,
                                          write,path.stem,cfg['defense_comparison'])
                    else:
                        work=jobs(net,image,attacked[ix],pred_proto,suppress,full,ops,write,path.stem,total_only)
                    for _,_,fn,setup in work:
                        for _ in range(cfg['warmup_per_image']):fn(*setup())
                    if cfg.get('balance_carryover',False):
                        first_arm_index=0 if ix%2==0 else 2
                        burn=work[first_arm_index][2]()
                        del burn
                        orders=carryover_orders(cfg['repeats'],ix,20261107+100*ix)
                    else:
                        orders=total_orders(cfg['repeats'],ix) if total_only else balanced_orders(work,cfg['repeats'],20261107+100*ix)
                    for rep,order in enumerate(orders):
                        for pos,j in enumerate(order):
                            arm,stage,fn,setup=work[j]
                            arguments=setup()  # NMS clone outside BOTH clocks
                            ns,start,_,cpu_ns=timed(lambda f=fn,a=arguments:f(*a))
                            pdh_sample=None if pdh is None else pdh.collect()
                            write(kind='timing',image=path.stem,block=ix,arm=arm,stage=stage,label=arm,
                                  rep=rep,position=pos,ns=ns,t_start_ns=start,process_cpu_ns=cpu_ns,pdh_after=pdh_sample)
                    if stage_check is not None:stage_check()
                    write(kind='memory',image=path.stem,info=proc.memory_info()._asdict())
                    flush()
                    continue
                pred, proto = pred_proto(net(image))
                det = suppress(pred)
                write(kind='counts', image=path.stem, n_tau=int((pred[:, 4:84].amax(1)>.25).sum()),
                      k=len(det), proto_shape=list(proto.shape))
                for _ in range(cfg['warmup_per_image']):
                    net(image); full(image)
                for rep in range(cfg['repeats']):
                    for pos, (stage, label) in enumerate(aa_order(rep, ix)):
                        fn = (lambda: net(image)) if stage == 'forward' else (lambda: full(image))
                        ns, start, _, cpu_ns = timed(fn)
                        pdh_sample = None if pdh is None else pdh.collect()
                        write(kind='timing', image=path.stem, block=ix, stage=stage,
                              label=label, rep=rep, position=pos, ns=ns, t_start_ns=start,
                              process_cpu_ns=cpu_ns,pdh_after=pdh_sample)
                flush()
            reference('end')
            verify('pin_end')
        power_end = hw.power_snapshot()
        if power_end['power_plugged'] is not True or power_mode() != lock['power_mode']:
            raise RuntimeError('power condition changed at end')
        qos_end = None if qos is None else qos.snapshot()
        if qos is not None and not qos.high_readback_ok(qos_end):
            raise RuntimeError('process QoS changed at end')
        if cfg.get('priority')=='above_normal' and proc.nice()!=psutil.ABOVE_NORMAL_PRIORITY_CLASS:
            raise RuntimeError('priority changed at end')
        write(kind='done', affinity=proc.cpu_affinity(), power_end=power_end,
              power_mode_end=power_mode(), qos_end=qos_end)
    except BaseException as e:
        write(kind='aborted', error=type(e).__name__, message=str(e))
        raise
    finally:
        flush()
        fh.close()
        if pdh is not None:
            pdh.close()


def campaign(args):
    lock = load_lock(args.lock)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    # One timing worker at a time. Parent blocks without reading progress.
    with (args.run_dir/'campaign.json').open('x', encoding='utf-8') as f:
        json.dump(dict(tag=args.tag, commit=args.commit, lock_sha256=sha(args.lock),
                       user_idle=args.user_idle, smoke=lock['config'].get('smoke', True)), f, indent=2)
    time.sleep(lock['config']['settle_s'])
    statuses = []
    for ix, profile in enumerate(lock['profiles']):
        if ix:
            time.sleep(lock['config']['between_profiles_s'])
        cmd = [sys.executable, __file__, '--lock', str(args.lock), '--run-dir', str(args.run_dir/profile),
               '--tag', args.tag, '--commit', args.commit, '--profile', profile]
        with (args.run_dir/f'{profile}.log').open('x', encoding='utf-8') as log:
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=lock['config']['worker_timeout_s'])
                statuses.append(dict(profile=profile, returncode=result.returncode))
            except subprocess.TimeoutExpired:
                statuses.append(dict(profile=profile, timeout=True))
        # Failure is retained; the fixed second profile is not a retry.
    with (args.run_dir/'campaign_done.json').open('x', encoding='utf-8') as f:
        json.dump(statuses, f, indent=2)
    print('CAMPAIGN_FINISHED', json.dumps(statuses), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lock', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--profile', choices=['P1_cpu2', 'P2'])
    ap.add_argument('--user-idle', default='')
    args = ap.parse_args()
    if args.profile:
        worker(args)
    elif args.user_idle:
        campaign(args)
    else:
        ap.error('user idle declaration required for campaign')
