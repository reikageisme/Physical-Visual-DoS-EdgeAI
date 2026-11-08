"""Saved-stage audit only. Infrastructure checks follow sibling audit_defense.

No inference or new timing. Does not promote historical failed measurements.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from analyze_attack import analyze
from attack_replay import carryover_orders

REPO=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root,lockpath):
    run=root/'P1_cpu2';lock=json.loads(lockpath.read_text());cfg=lock['config']
    saved=json.loads((run/'stage_summary.json').read_text())
    rows=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    manifest=json.loads((run/'manifest.json').read_text());tag=manifest['tag']
    assert sha(lockpath)==manifest['lock_sha256']
    assert subprocess.check_output(['git','rev-parse',tag],text=True).strip()==manifest['prereg_commit']
    assert subprocess.check_output(['git','show',f'{tag}:{lockpath.relative_to(REPO).as_posix()}'])==lockpath.read_bytes()
    differences=[]
    for name,expected in lock['source_sha256'].items():
        assert sha(REPO/name)==expected
        blob=subprocess.check_output(['git','show',f'{tag}:{name}'])
        if hashlib.sha256(blob).hexdigest()!=expected:
            assert name=='sponge-sim/src/sponge_sim/hw.py'
            assert blob.replace(b'\r\n',b'\n')==(REPO/name).read_bytes().replace(b'\r\n',b'\n')
            differences.append(dict(path=name,tag_sha256=hashlib.sha256(blob).hexdigest(),working_sha256=expected))
    for entry in lock['files']:assert sha(REPO/entry['path'])==entry['sha256']
    assert all(saved[k]==v for k,v in analyze(run).items())
    events=[json.loads(s) for s in root.with_name(root.name+'_affinity.jsonl').read_text().splitlines()]
    assert any(e['kind']=='ready' for e in events)
    assert not any(e['kind'].startswith('restore_') for e in events)
    assert all(e['readback']==e['expected'] for e in events if e['kind']=='restored')
    assert {e['pid'] for e in events if e['kind']=='applied'}<={e['pid'] for e in events if e['kind'] in ('restored','exited')}
    end=rows[-1];assert end['kind']=='done' and end['affinity']==manifest['cpu_set']==[2]
    assert manifest['power_start']['power_plugged'] and end['power_end']['power_plugged']
    assert manifest['power_mode_start']==end['power_mode_end']==lock['power_mode']
    for q in (manifest['qos_start'],end['qos_end']):
        assert q['control_mask'] & 1 and not q['state_mask'] & 1 and q['priority_class']==32768
    ids=[Path(p).stem for p in lock['input_paths'][:cfg['n_images']]]
    expected_counts=json.loads((REPO/'sponge-sim/results/laptop_session_confirm_20261107/full_01/P1_cpu2/attack_summary.json').read_text())['counts']
    expected_counts={(r['image'],r['arm']):r for r in expected_counts}
    names=[(a,s) for a in ('clean','attacked') for s in ('aa_A','aa_B')]
    for ix,image in enumerate(ids):
        trials=[r for r in rows if r['kind']=='timing' and r['image']==image]
        order=[[names.index((r['arm'],r['stage'])) for r in trials if r['rep']==rep] for rep in range(48)]
        assert order==carryover_orders(48,ix,20261107+100*ix)
        for arm in ('clean','attacked'):
            counts=[r for r in saved['counts'] if r['image']==image and r['arm']==arm]
            assert len(counts)==1;c=counts[0];previous=expected_counts[image,arm]
            for key in ('n_tau','k','prediction_sha256','proto_shape'):assert c[key]==previous[key]
            assert c['kernel_n']==c['n_tau'] and c['k']==min(c['kernel_kept'],300) and c['kernel_iou']==.45
            checks=[r for r in saved['stage_input_checks'] if r['image']==image and r['arm']==arm]
            assert len(checks)==1 and checks[0]['before_sha256']==checks[0]['after_sha256'] and checks[0]['unchanged']
    receipt=dict(status='OFFLINE_STAGE_AUDIT_NO_NEW_INFERENCE',tag=tag,commit=manifest['prereg_commit'],
                 lock_sha256=sha(lockpath),raw_sha256=sha(run/'raw.jsonl'),summary_sha256=sha(run/'stage_summary.json'),
                 auditor_sha256=sha(Path(__file__)),quality_pass=saved['quality_pass'],feasibility_pass=saved['feasibility_pass'],
                 endpoint=cfg['isolated_stage'],calls_per_trial=cfg['calls_per_trial'],
                 n_trials=sum(r['kind']=='timing' for r in rows),
                 actual_timed_operation_calls=sum(r['kind']=='timing' for r in rows)*cfg['calls_per_trial'],
                 restoration=dict(Counter(e['kind'] for e in events)),tag_line_ending_differences=differences,
                 duration_raw_s=end['wall_utc']-rows[0]['wall_utc'],
                 checks=['source/tag/input','original predictions/counts','captured fixed-kernel boundary',
                         'immutable prepared features','exact carryover order','recomputed estimator/gates','environment/app restore'])
    with (root/'verification_receipt.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({k:receipt[k] for k in ('status','endpoint','quality_pass','feasibility_pass','raw_sha256','n_trials','restoration')},indent=2))


if __name__=='__main__':main(*(Path(p).resolve() for p in sys.argv[1:]))
