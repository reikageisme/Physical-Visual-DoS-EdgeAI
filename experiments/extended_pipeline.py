"""Sequential local research queue; never edits papers or publishes unfinished data.

Waits for the already-running optimizer, then launches one phase at a time
using this same interpreter. Source hashes are frozen before launch. Failure
stops the queue and is recorded; there is no random-result fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASES = [('evaluation','extended_evaluation.py'),('controls','extended_controls.py'),
          ('backend_qos','extended_backend_qos.py'),('transfer','extended_transfer.py'),
          ('reference','extended_reference.py'),
          ('analysis','extended_analysis.py'),('figures','extended_figures.py')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='outputs/extended_q1_20261002')
    args = parser.parse_args()
    run = (ROOT/args.run_dir).resolve()
    if not run.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Research output must remain inside task outputs')
    plan = dict(interpreter=sys.executable,run_dir=str(run),
                script_sha256=digest(__file__),phases=[dict(name=n,script=f,sha256=digest(ROOT/'experiments'/f)) for n,f in PHASES],
                excluded_work='laptop-only image experiments; webcam optical tests and second physical hardware are outside this software queue',
                does_not_write_or_upload_manuscripts=True)
    destination = run/'pipeline.json'
    if destination.exists():
        old = json.loads(destination.read_text())
        if old['plan'] != plan:
            raise RuntimeError('Changed queued sources; inspect existing state before new queue')
        if old.get('finished_utc'):
            print('PIPELINE ALREADY COMPLETE',flush=True)
            return
        state = old
    else:
        state = dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,status='waiting_for_optimizer',completed_phases=[])
        save(destination,state)
    last_count = None
    while True:
        optimizer = json.loads((run/'manifest.json').read_text())
        if 'optimization_finished_utc' in optimizer:
            break
        count = len(list((run/'optimization').glob('*/complete.json')))
        if count != last_count:
            print(f'WAITING FOR OPTIMIZER: {count}/160 completed cases',flush=True)
            last_count = count
        time.sleep(30)
    for phase in plan['phases']:
        name = phase['name']
        if name in state['completed_phases']:
            continue
        script = ROOT/'experiments'/phase['script']
        if digest(script) != phase['sha256']:
            state.update(status='source_changed',failed_phase=name)
            save(destination,state)
            raise RuntimeError('Queued source changed: '+str(script))
        state.update(status='running',current_phase=name,phase_started_utc=datetime.now(timezone.utc).isoformat())
        save(destination,state)
        print('STARTING PHASE: '+name,flush=True)
        command = [sys.executable,'-X','utf8','-u',str(script),'--run-dir',str(run)]
        with (run/f'pipeline_{name}.log').open('a',encoding='utf-8') as log:
            result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            state.update(status='failed',failed_phase=name,returncode=result.returncode,
                         failed_utc=datetime.now(timezone.utc).isoformat())
            save(destination,state)
            raise RuntimeError(f'Phase {name} failed; inspect pipeline_{name}.log')
        state['completed_phases'].append(name)
        save(destination,state)
        print('FINISHED PHASE: '+name,flush=True)
    state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    state.pop('current_phase',None)
    save(destination,state)
    print('SOFTWARE PIPELINE COMPLETE; real hardware work and manuscript integration are separate',flush=True)


if __name__ == '__main__':
    main()
