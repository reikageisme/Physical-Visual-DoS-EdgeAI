"""Audit saved defense measurements offline; no model loads or new timing."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from analyze_attack import analyze
from attack_replay import carryover_orders

REPO = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, lockpath):
    run=root/'P1_cpu2'
    saved=json.loads((run/'defense_summary.json').read_text())
    rows=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    manifest=json.loads((run/'manifest.json').read_text())
    lock=json.loads(lockpath.read_text());cfg=lock['config'];tag=manifest['tag']
    assert sha(lockpath)==manifest['lock_sha256']
    assert subprocess.check_output(['git','rev-parse',tag],text=True).strip()==manifest['prereg_commit']
    assert subprocess.check_output(['git','show',f'{tag}:{lockpath.relative_to(REPO).as_posix()}'])==lockpath.read_bytes()
    line_endings=[]
    for name,expected in lock['source_sha256'].items():
        assert sha(REPO/name)==expected
        tagged=subprocess.check_output(['git','show',f'{tag}:{name}'])
        if hashlib.sha256(tagged).hexdigest()!=expected:
            assert name=='sponge-sim/src/sponge_sim/hw.py'
            assert tagged.replace(b'\r\n',b'\n')==(REPO/name).read_bytes().replace(b'\r\n',b'\n')
            line_endings.append(dict(path=name,tag_sha256=hashlib.sha256(tagged).hexdigest(),working_sha256=expected))
    for item in lock['files']:assert sha(REPO/item['path'])==item['sha256']
    assert all(saved[k]==v for k,v in analyze(run).items())
    journal=root.with_name(root.name+'_affinity.jsonl')
    events=[json.loads(s) for s in journal.read_text().splitlines()]
    assert any(e['kind']=='ready' for e in events)
    assert not any(e['kind'].startswith('restore_') for e in events)
    assert all(e['readback']==e['expected'] for e in events if e['kind']=='restored')
    assert {e['pid'] for e in events if e['kind']=='applied'} <= {e['pid'] for e in events if e['kind'] in ('restored','exited')}
    done=rows[-1];assert done['kind']=='done' and done['affinity']==manifest['cpu_set']==[2]
    assert manifest['power_start']['power_plugged'] and done['power_end']['power_plugged']
    assert manifest['power_mode_start']==done['power_mode_end']==lock['power_mode']
    for qos in (manifest['qos_start'],done['qos_end']):
        assert qos['control_mask'] & 1 and not qos['state_mask'] & 1 and qos['priority_class']==32768
    previous=json.loads((REPO/'sponge-sim/results/laptop_session_confirm_20261107/full_01/P1_cpu2/attack_summary.json').read_text())
    previous={r['image']:r for r in previous['counts'] if r['arm']=='attacked'}
    ids=[Path(p).stem for p in lock['input_paths'][:cfg['n_images']]]
    job_names=[(a,s) for a in cfg['arms'] for s in ('aa_A','aa_B')]
    for ix,image in enumerate(ids):
        trials=[r for r in rows if r['kind']=='timing' and r['image']==image]
        actual=[[job_names.index((r['arm'],r['stage'])) for r in trials if r['rep']==rep] for rep in range(48)]
        assert actual==carryover_orders(48,ix,20261107+100*ix)
        counts=[r for r in saved['counts'] if r['image']==image]
        assert [r['arm'] for r in counts]==cfg['arms']
        for r in counts:
            for key in ('n_tau','prediction_sha256'):assert r[key]==previous[image][key]
            options=dict(max_det=300,max_nms=30000) if r['arm']=='baseline' else cfg['defense_comparison']
            assert r['options']==options and r['kernel_n']==min(r['n_tau'],options['max_nms'])
            assert r['k']<=options['max_det'] and r['input_kind']=='same_frozen_attacked'
            if r['arm']=='baseline':assert r['k']==previous[image]['k']
    receipt=dict(status='OFFLINE_VERIFIED_NO_NEW_INFERENCE',tag=tag,commit=manifest['prereg_commit'],
                 raw_sha256=sha(run/'raw.jsonl'),lock_sha256=sha(lockpath),
                 summary_sha256=sha(run/'defense_summary.json'),audit_source_sha256=sha(Path(__file__)),
                 tag_line_ending_differences=line_endings,quality_pass=saved['quality_pass'],
                 feasibility_pass=saved['feasibility_pass'],n_trials=sum(r['kind']=='timing' for r in rows),
                 restored_events=dict(Counter(e['kind'] for e in events)),
                 duration_raw_s=done['wall_utc']-rows[0]['wall_utc'],
                 checks=['source/tag/input hashes','same attacked prediction as accepted prior session',
                         'exact carryover schedule','observed kernel and retained output bounds',
                         'saved summary recomputed','power/QoS/affinity/restore'])
    with (root/'verification_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main(*(Path(p).resolve() for p in sys.argv[1:]))
