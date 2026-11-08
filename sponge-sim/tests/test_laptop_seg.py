"""Synthetic fixtures, no inference/timing benchmark."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('laptop_'+name, SIM/'studies/laptop_seg'/f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('repeats',[8,16])
def test_order_balances_stage_position_and_labels(repeats):
    run = load('run')
    counts = Counter()
    for image in (0,1):
        for rep in range(repeats):
            order = run.aa_order(rep,image)
            assert sorted(order) == [('forward','A'),('forward','B'),('total','A'),('total','B')]
            for pos, (s,l) in enumerate(order):
                counts[image,s,l,pos] += 1
    assert set(counts.values()) == {repeats//4}


def fixture(path, n=2):
    (path/'manifest.json').write_text(json.dumps({'config':{'n_images':n,'repeats':8}}))
    rows = [dict(kind='pin_start',verify={'passed':True})]
    rows += [dict(kind='reference',flagged=False,ratio=1.0) for _ in range(n+2)]
    rows += [dict(kind='timing',image=i,stage=s,label=l,ns=1000000,rep=r)
             for i in (str(k) for k in range(n)) for s in ('forward','total') for l in ('A','B') for r in range(8)]
    rows += [dict(kind='pin_end',verify={'passed':True}),dict(kind='done')]
    return rows


def save(path, rows):
    (path/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_complete_aa_is_smoke_only(tmp_path):
    rows = fixture(tmp_path)
    save(tmp_path,rows)
    r = load('analyze').analyze(tmp_path)
    assert r['feasibility_pass'] and not r['inference_allowed']
    assert r['stages']['total']['aa_mde_fraction'] == 0


@pytest.mark.parametrize('fault', ['repeat','duplicate_rep','flag','pin','incomplete'])
def test_faults_cannot_be_promoted(tmp_path,fault):
    rows = fixture(tmp_path)
    if fault == 'repeat':
        del rows[5]
    if fault == 'duplicate_rep':
        rows[5]['rep'] = 1
    if fault == 'flag':
        rows[1]['flagged'] = True
    if fault == 'pin':
        rows[0]['verify']['passed'] = False
    if fault == 'incomplete':
        rows.pop()
    save(tmp_path,rows)
    assert not load('analyze').analyze(tmp_path)['feasibility_pass']


@pytest.mark.parametrize('noisy', [False,True])
def test_full_controls_require_mde_gate_too(tmp_path,noisy):
    rows = fixture(tmp_path,20)
    m = json.loads((tmp_path/'manifest.json').read_text())
    m['config'].update(smoke=False,max_reference_flags=2,max_mde_fraction=.01)
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    if noisy:
        for row in rows:
            if row['kind']=='timing' and row['stage']=='total' and row['label']=='A':
                row['ns'] *= 1.1 if int(row['image'])%2 else .9
    save(tmp_path,rows)
    r = load('analyze').analyze(tmp_path)
    assert r['feasibility_pass'] and r['quality_pass'] is (not noisy)
    assert not r['inference_allowed']


def test_high_qos_requires_control_and_disabled_throttling():
    qos=load('windows_qos')
    assert qos.high_readback_ok(dict(control_mask=1,state_mask=0))
    assert not qos.high_readback_ok(dict(control_mask=0,state_mask=0))
    assert not qos.high_readback_ok(dict(control_mask=1,state_mask=1))


def test_cpu_diagnostic_retains_high_fractions_and_detects_wait_like_spike():
    m=load('analyze_cpu')
    rows=[dict(kind='timing',image='i',stage='forward',label='A',rep=i,ns=wall,process_cpu_ns=cpu)
          for i,(wall,cpu) in enumerate([(100,110),(100,100),(100,100),(1000,100)])]
    result=m.summarize(rows)['stages']['forward']
    assert result['n_highwall']==1 and result['highwall_low_cpu_share']==1
    assert result['highwall_trials'][0]['cpu_over_wall']==.1
    assert result['median_cpu_over_wall']==1  # >1 trial retained
    rows[0]['process_cpu_ns']=None
    with pytest.raises(ValueError):
        m.summarize(rows)


def test_pdh_missing_is_not_zero_and_over100_is_retained():
    m=load('pdh_cpu')
    assert m.valid_value(0,0,250.0)==250.0
    assert m.valid_value(0,1,0.0)==0.0
    assert m.valid_value(0,5,0.0) is None
    assert m.valid_value(5,0,0.0) is None


def test_pdh_descriptive_correlations_preserve_undefined_and_missing():
    import sys
    sys.path.insert(0,str(SIM/'studies/laptop_seg'))
    m=load('analyze_pdh')
    assert m.correlation([(1,2),(2,2)])['spearman'] is None
    assert m.correlation([(1,None),(2,3)])['n']==1
    assert m.correlation([(1,3),(2,2),(3,1)])['spearman']==-1
    assert m.describe([None,0,200])==dict(valid=2,minimum=0,median=100,maximum=200)


def test_app_separation_never_empty_mask():
    m=load('separate_app')
    assert m.restricted([0,1,2,3,4])==[0,1,4]
    with pytest.raises(RuntimeError):m.restricted([2,3])


def test_app_restore_handles_new_children_and_preserves_manual_changes(monkeypatch):
    from types import SimpleNamespace
    m=load('separate_app')
    class P:
        def __init__(self,pid,mask,parent=None):self.pid,self.mask,self.parent=pid,mask,parent
        def create_time(self):return self.pid*10
        def name(self):return 'helper'
        def cpu_affinity(self,mask=None):
            if mask is not None:self.mask=mask
            return self.mask
        def parents(self):return [] if self.parent is None else [self.parent]
        def children(self,recursive):return [new,changed]
    root=P(1,[0,1,4]);new=P(2,[0,1,4],root);changed=P(3,[0],root)
    procs={p.pid:p for p in [root,new,changed]}
    monkeypatch.setattr(m.psutil,'Process',lambda pid:procs[pid])
    sep=m.Separation.__new__(m.Separation);sep.root=root;sep.file=SimpleNamespace(close=lambda:None)
    sep.records={(1,10):dict(pid=1,created=10,old=[0,1,2,3,4],new=[0,1,4]),
                 (3,30):dict(pid=3,created=30,old=[0,1,2,3,4],new=[0,1,4])}
    events=[];sep.write=events.append;sep.restore()
    assert root.mask==new.mask==[0,1,2,3,4]
    assert changed.mask==[0]
    assert any(r['kind']=='restore_skipped_changed_mask' for r in events)


def test_attack_stage_and_arm_positions_balanced():
    from collections import Counter
    m=load('attack_replay');jobs=[(a,s) for a in ['clean','attacked'] for s in range(6)]
    orders=m.balanced_orders(jobs,12,20261107);pos=Counter();first=Counter()
    for order in orders:
        assert sorted(order)==list(range(12))
        for j in range(0,12,2):
            a,b=jobs[order[j]],jobs[order[j+1]]
            assert a[1]==b[1] and a[0]!=b[0]
            pos[a[1],j//2]+=1;first[a]+=1
    assert set(pos.values())=={2} and set(first.values())=={6}


@pytest.mark.parametrize('total_only',[False,True])
def test_attack_analysis_known_doubling_and_missing_repetition(tmp_path,total_only):
    m=load('analyze_attack')
    repeats=48 if total_only else 12
    (tmp_path/'manifest.json').write_text(json.dumps({'config':dict(n_images=20,repeats=repeats,total_only=total_only,smoke=False,max_reference_flags=2,max_mde_fraction=.01)}))
    rows=[dict(kind='reference',flagged=False) for _ in range(22)]
    rows += [dict(kind='timing',image=f'i{i}',arm=a,stage=s,rep=rep,ns=(100+i)*(1 if a=='clean' else 2))
             for i in range(20) for a in ('clean','attacked') for s in (('aa_A','aa_B') if total_only else m.STAGES) for rep in range(repeats)]
    rows += [dict(kind='pin_start',verify={'passed':True}),dict(kind='pin_end',verify={'passed':True}),dict(kind='done')]
    save(tmp_path,rows);r=m.analyze(tmp_path)
    assert r['quality_pass'] and r['contrasts']['total']['geometric_ratio']==2
    assert r['contrasts']['total']['ci95']==[2,2]
    rows.pop(22);save(tmp_path,rows)
    assert not m.analyze(tmp_path)['quality_pass']


def test_total_only_order_balances_every_position_and_adjacent_duplicates():
    m=load('attack_replay');counts=Counter()
    for order in m.total_orders(48,0):
        assert sorted(order)==[0,1,2,3]
        assert order[0]//2==order[1]//2 and order[2]//2==order[3]//2
        for pos,j in enumerate(order):counts[j,pos]+=1
    assert set(counts.values())=={12}


@pytest.mark.parametrize('image_index',[0,1])
def test_carryover_labels_have_identical_contexts_and_positions(image_index):
    m=load('attack_replay');orders=m.carryover_orders(48,image_index,20261107)
    previous=0 if image_index%2==0 else 1;contexts=Counter();positions=Counter()
    for order in orders:
        assert sorted(order)==[0,1,2,3]
        for pos,j in enumerate(order):
            arm=j//2;label=j%2
            contexts[arm,label,previous]+=1;positions[j,pos]+=1;previous=arm
    for arm in (0,1):
        for prev in (0,1):assert contexts[arm,0,prev]==contexts[arm,1,prev]
    assert set(positions.values())=={12}
    assert orders==m.carryover_orders(48,image_index,20261107)


def test_output_release_finishes_computation_and_prevents_cross_call_retention():
    import weakref
    m=load('output_lifetime');alive=[]
    class Buffer:pass
    def fn():
        obj=Buffer();alive.append(weakref.ref(obj));return obj
    old=m.measure(fn,capture_cpu=True,keep_output=True)
    assert alive[-1]() is old[2]
    del old
    assert alive[-1]() is None
    for _ in range(3):
        result=m.measure(fn,capture_cpu=True,keep_output=False)
        assert result[2] is None and alive[-1]() is None
        assert result[0]>0 and result[3]>=0


def test_defense_kernel_probe_restores_original_on_exception():
    from types import SimpleNamespace
    m=load('defense_replay')
    original=lambda *a:None
    tv=SimpleNamespace(ops=SimpleNamespace(nms=original))
    class Pred:
        def clone(self):return self
    def broken(pred,**kwargs):raise ValueError('probe failure')
    with pytest.raises(ValueError):m.observed_suppression(Pred(),broken,tv,{})
    assert tv.ops.nms is original


@pytest.mark.parametrize('setting',[dict(max_det=300,max_nms=300),dict(max_det=50,max_nms=30000)])
def test_defense_jobs_share_input_but_count_different_boundaries(setting):
    from types import SimpleNamespace
    import numpy as np
    m=load('defense_replay')
    class Tensor(np.ndarray):
        def numpy(self):return np.asarray(self)
        def clone(self):return self.copy()
        def amax(self,axis):return self.max(axis=axis)
    pred=np.zeros((1,116,700),dtype=np.float32).view(Tensor);pred[:,4:84]=.5
    original=lambda boxes,scores,iou:np.arange(len(scores))
    tv=SimpleNamespace(ops=SimpleNamespace(nms=original))
    def suppress(p,max_det,max_nms):
        kernel=min(700,max_nms)
        tv.ops.nms(np.zeros((kernel,4)),np.zeros(kernel),.45)
        return np.zeros((min(max_det,kernel//4),38),dtype=np.float32).view(Tensor)
    calls=[];rows=[];attacked=object()
    def full(image,**kwargs):calls.append((image,kwargs));return None
    jobs=m.jobs(lambda x:pred,attacked,lambda p:(p,None),suppress,full,tv,
                lambda **r:rows.append(r),'image',setting)
    assert tv.ops.nms is original
    assert jobs[0][2] is jobs[1][2] and jobs[2][2] is jobs[3][2]
    assert rows[0]['prediction_sha256']==rows[1]['prediction_sha256']
    assert rows[0]['kernel_n']==700 and rows[0]['k']==175
    assert rows[1]['kernel_n']==min(700,setting['max_nms'])
    assert rows[1]['k']==min(setting['max_det'],rows[1]['kernel_n']//4)
    for job in jobs:job[2]()
    assert all(c[0] is attacked for c in calls)
    assert calls[0][1]==dict(max_det=300,max_nms=30000) and calls[-1][1]==setting


def test_defense_analysis_ratio_is_defended_over_same_attacked_baseline(tmp_path):
    m=load('analyze_attack')
    cfg=dict(n_images=20,repeats=48,total_only=True,smoke=False,max_reference_flags=2,
             max_mde_fraction=.01,arms=['baseline','defended'],defense_comparison={'max_det':300,'max_nms':300})
    (tmp_path/'manifest.json').write_text(json.dumps({'config':cfg}))
    rows=[dict(kind='reference',flagged=False) for _ in range(22)]
    rows += [dict(kind='timing',image=f'i{i}',arm=a,stage=s,rep=rep,ns=(100+i)*(2 if a=='baseline' else 1))
             for i in range(20) for a in cfg['arms'] for s in ('aa_A','aa_B') for rep in range(48)]
    rows += [dict(kind='pin_start',verify={'passed':True}),dict(kind='pin_end',verify={'passed':True}),dict(kind='done')]
    save(tmp_path,rows);r=m.analyze(tmp_path)
    assert r['quality_pass'] and r['contrasts']['total']['geometric_ratio']==.5
    assert r['contrasts']['total']['ci95']==[.5,.5]
    assert 'baseline_medians_ms' in r['contrasts']['total'] and 'clean_medians_ms' not in r['contrasts']['total']
    assert r['aa']['baseline']['mde_fraction']==r['aa']['defended']['mde_fraction']==0
    rows.pop(22);save(tmp_path,rows)
    assert not m.analyze(tmp_path)['quality_pass']


def test_prespecified_defense_signflip_and_holm():
    m=load('defense_stats')
    assert m.exact_signflip([1,2,3])==.25
    assert m.exact_signflip([1,1,-1])==1
    assert m.exact_signflip([0,0,0])==1
    assert m.holm([.03,.04])==[.06,.06]
    assert m.holm([1,.001])==[1,.002]
