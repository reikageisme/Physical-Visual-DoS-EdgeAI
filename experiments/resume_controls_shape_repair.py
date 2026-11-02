"""One explicitly bounded repair of a failed, not-yet-measured control phase.

Keeps optimizer/evaluation data and the failed source/log/manifest snapshot.
No source hash is silently relaxed. No finished scientific case is rewritten.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/extended_q1_20261002'
SNAPSHOT = RUN/'repairs/20261003_controls_shape'
OLD_CONTROL_HASH = '476e5cd51c44e00538dc7f645db67984744d04d7fde5a1000399cf129ba9d73e'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    temporary = path.with_suffix('.repair.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    pipeline_file = RUN/'pipeline.json'
    controls_file = RUN/'controls/manifest.json'
    stream_file = RUN/'stream_qos/manifest.json'
    state = json.loads(pipeline_file.read_text(encoding='utf-8'))
    manifest = json.loads(controls_file.read_text(encoding='utf-8'))
    stream = json.loads(stream_file.read_text(encoding='utf-8'))
    before = (SNAPSHOT/'extended_controls.before.py').read_bytes()
    after = (ROOT/'experiments/extended_controls.py').read_bytes()
    assert digest(SNAPSHOT/'extended_controls.before.py') == OLD_CONTROL_HASH
    assert after.decode('utf-8').splitlines() == before.decode('utf-8').replace(
        'locations = find_locations(list(train),64,',
        'locations = find_locations(list(train.split(1,dim=0)),64,',1).splitlines()
    new_hash = digest(ROOT/'experiments/extended_controls.py')
    if state.get('repair_history') and state['repair_history'][-1].get('new_control_sha256') == new_hash:
        print('CONTROL SHAPE REPAIR ALREADY RECORDED')
        return
    assert state['status'] == 'failed' and state['failed_phase'] == 'controls'
    assert state['completed_phases'] == ['evaluation']
    assert not list((RUN/'controls').glob('*/evaluation.json'))
    assert not list((RUN/'controls').glob('*/optimization.json'))
    assert not list((RUN/'controls').glob('*/resume.pt'))
    assert manifest['protocol']['script_sha256'] == OLD_CONTROL_HASH
    assert stream['status'] == 'failed' and stream['error_message'] == 'Preceding pipeline stopped: failed'
    assert digest(ROOT/'experiments/extended_stream_qos.py') == stream['protocol']['script_sha256']
    assert len(list((RUN/'stream_qos').glob('*.json'))) == 1
    assert json.loads((RUN/'evaluation/manifest.json').read_text(encoding='utf-8'))['evaluated_cases'] == 250
    assert digest(ROOT/'experiments/extended_pipeline.py') == state['plan']['script_sha256']
    for phase in state['plan']['phases']:
        if phase['name'] == 'controls':
            assert phase['sha256'] == OLD_CONTROL_HASH
        else:
            assert digest(ROOT/'experiments'/phase['script']) == phase['sha256']
    record = dict(repaired_utc=datetime.now(timezone.utc).isoformat(),old_control_sha256=OLD_CONTROL_HASH,
                  new_control_sha256=new_hash,reason='preserve batch dimension for each training image in location search',
                  scientific_configuration_unchanged=True,completed_optimizer_cases_preserved=160,
                  completed_evaluation_cases_preserved=250,completed_control_cases_before_repair=0,
                  snapshot_directory=str(SNAPSHOT.relative_to(ROOT)),repair_script_sha256=digest(__file__))
    manifest['protocol']['script_sha256'] = new_hash
    manifest.setdefault('repair_history',[]).append(record)
    for phase in state['plan']['phases']:
        if phase['name'] == 'controls':
            phase['sha256'] = new_hash
    state.setdefault('repair_history',[]).append(record)
    state['status'] = 'ready_to_resume'
    for field in ['failed_phase','returncode','failed_utc','current_phase','phase_started_utc']:
        state.pop(field,None)
    stream.setdefault('previous_failures',[]).append({key:stream[key] for key in
        ['failed_utc','error_type','error_message']})
    for field in ['failed_utc','error_type','error_message']:
        stream.pop(field,None)
    stream.update(status='waiting',upstream_repaired_utc=record['repaired_utc'])
    save(stream_file,stream)
    save(controls_file,manifest)
    save(pipeline_file,state)
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
