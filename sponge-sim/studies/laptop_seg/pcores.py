"""Fixed two-physical-P-core replay using the unchanged registered worker.

Own-app transaction derives from separate_app.py:46-70 (source SHA in lock),
extending exclusion/readback to BOTH P cores and their SMT siblings. Original
runner/separation files remain unchanged for historical source audits.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import psutil

import run
import separate_app

CPUS = [0, 2]
EXCLUDED = [0, 1, 2, 3]


def verify_topology(cores):
    pairs = sorted(sorted(c['logical_cpus']) for c in cores if c['kind'] == 'P')
    if pairs != [[0, 1], [2, 3]]:
        raise RuntimeError('registered two physical P-core mapping changed')
    return CPUS[:], EXCLUDED[:]


class PcoreSeparation(separate_app.Separation):
    def apply(self):
        for attempt in range(3):
            for p in [self.root, *self.root.children(recursive=True)]:
                try:
                    key = (p.pid, p.create_time())
                    current = sorted(p.cpu_affinity())
                    if key in self.records:
                        if current != self.records[key]['new']:
                            raise RuntimeError('affinity changed during preparation')
                        continue
                    new = separate_app.restricted(current, excluded=EXCLUDED)
                    row = dict(kind='intent', pid=p.pid, created=key[1], name=p.name(), old=current, new=new)
                    self.write(row)
                    self.records[key] = row
                    p.cpu_affinity(new)
                    if sorted(p.cpu_affinity()) != new:
                        raise RuntimeError('helper affinity readback failed')
                    self.write(dict(kind='applied', pid=p.pid, created=key[1]))
                except psutil.NoSuchProcess:
                    continue
            live = [self.root, *self.root.children(recursive=True)]
            if all(not set(p.cpu_affinity()) & set(EXCLUDED) for p in live if p.is_running()):
                self.write(dict(kind='ready', processes=len(self.records), attempt=attempt,
                                excluded=EXCLUDED, profile_cpu_set=CPUS))
                return
        raise RuntimeError('process tree changed too quickly; no benchmark')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lock', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--user-idle', required=True)
    args = ap.parse_args()
    lock = run.load_lock(args.lock)
    if lock['profiles'] != {'P2': CPUS}:
        raise RuntimeError('exact fixed P2 profile required')
    commit = subprocess.check_output(['git', 'rev-parse', args.tag+'^{commit}'], cwd=run.REPO, text=True).strip()
    if commit != args.commit:
        raise RuntimeError('tag identity mismatch')
    sys.path.insert(0, str(run.SIM/'src'))
    from sponge_sim import hw
    topology = hw.cpu_topology()
    verify_topology(topology)
    if topology != lock['pcores_topology']:
        raise RuntimeError('topology changed since source/config lock')
    sep = PcoreSeparation(args.run_dir.with_name(args.run_dir.name+'_affinity.jsonl'))
    try:
        sep.write(dict(kind='profile_topology', cpu_set=CPUS, excluded=EXCLUDED, topology=topology))
        sep.apply()
        run.campaign(args)  # original fresh worker pins [0,2], 2 intra-op / 1 inter-op
    finally:
        sep.restore()
    print('PCORES_APP_AFFINITY_RESTORE_FINISHED', flush=True)


if __name__ == '__main__':
    main()
