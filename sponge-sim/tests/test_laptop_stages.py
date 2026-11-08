"""Prepared-stage contracts using tiny fake tensors, not model inference."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

STUDY=Path(__file__).resolve().parents[1]/'studies/laptop_seg'
sys.path.insert(0,str(STUDY))
import stage_replay as stage
from analyze_attack import analyze


class Tensor(np.ndarray):
    def numpy(self):return np.asarray(self)
    def clone(self):return self.copy()
    def contiguous(self):return self.copy()
    def amax(self,axis):return self.max(axis=axis)


def tensor(value):return np.asarray(value).view(Tensor)


def fixture():
    pred=tensor(np.zeros((1,116,4),dtype=np.float32));pred[:,4:84]=.5
    proto=tensor(np.ones((32,2,2),dtype=np.float32))
    boxes=tensor(np.ones((4,4),dtype=np.float32));scores=tensor(np.ones(4,dtype=np.float32))
    indices=tensor([0,2]);det=tensor(np.ones((2,38),dtype=np.float32));calls=[]
    def kernel(b,s,iou):calls.append((b.copy(),s.copy(),iou));return indices.copy()
    tv=SimpleNamespace(ops=SimpleNamespace(nms=kernel))
    def suppress(p):tv.ops.nms(boxes,scores,.45);return det
    ops=SimpleNamespace(process_mask_native=lambda *args:tensor(np.zeros((2,2,2),dtype=np.uint8)))
    return pred,proto,boxes,scores,det,tv,suppress,ops,calls,kernel


@pytest.mark.parametrize('endpoint,ncalls',[('nms_kernel',64),('mask_native',1)])
def test_fixed_probe_restore_aa_identity_and_post_inputs(endpoint,ncalls):
    pred,proto,b,s,det,tv,suppress,ops,calls,kernel=fixture();rows=[]
    work,verify=stage.jobs(lambda im:None,object(),object(),lambda raw:(pred,proto),suppress,
                           ops,tv,lambda **r:rows.append(r),'i',dict(isolated_stage=endpoint,calls_per_trial=ncalls))
    assert tv.ops.nms is kernel
    assert work[0][2] is work[1][2] and work[2][2] is work[3][2]
    before=len(calls);work[0][2]()
    assert len(calls)-before==(64 if endpoint=='nms_kernel' else 0)
    verify()
    assert len([r for r in rows if r['kind']=='counts'])==2
    assert all(r['unchanged'] for r in rows if r['kind']=='stage_input_check')


def test_capture_restores_kernel_after_wrapper_failure():
    pred,_,_,_,_,tv,_,_,_,kernel=fixture()
    def bad(p):raise ValueError('untimed wrapper failed')
    with pytest.raises(ValueError):stage.capture_kernel(pred,bad,tv)
    assert tv.ops.nms is kernel


def test_native_probe_cannot_mutate_prepared_inputs():
    pred,proto,_,_,det,tv,suppress,ops,_,_=fixture()
    def mutate(pr,*args):pr[0,0,0]+=1;return tensor(np.ones((2,2,2)))
    ops.process_mask_native=mutate
    with pytest.raises(RuntimeError,match='probe mutated'):
        stage.jobs(lambda im:None,None,None,lambda raw:(pred,proto),suppress,ops,tv,
                   lambda **r:None,'i',dict(isolated_stage='mask_native',calls_per_trial=1))


def test_analysis_normalizes_block_absolute_cost_but_not_aa_ratio(tmp_path):
    cfg=dict(n_images=20,repeats=48,total_only=True,smoke=False,max_reference_flags=2,
             max_mde_fraction=.01,isolated_stage='nms_kernel',calls_per_trial=64)
    (tmp_path/'manifest.json').write_text(json.dumps({'config':cfg}))
    rows=[dict(kind='reference',flagged=False) for _ in range(22)]
    rows += [dict(kind='timing',image=f'i{i}',arm=a,stage=s,rep=rep,ns=64*(100+i)*(1 if a=='clean' else 2))
             for i in range(20) for a in ('clean','attacked') for s in ('aa_A','aa_B') for rep in range(48)]
    rows += [dict(kind='stage_input_check',image=f'i{i}',arm=a,unchanged=True)
             for i in range(20) for a in ('clean','attacked')]
    rows += [dict(kind='pin_start',verify={'passed':True}),dict(kind='pin_end',verify={'passed':True}),dict(kind='done')]
    def save():(tmp_path/'raw.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    save();r=analyze(tmp_path);t=r['contrasts']['nms_kernel']
    assert r['quality_pass'] and t['geometric_ratio']==2 and t['ci95']==[2,2]
    assert t['clean_medians_ms'][0]==.0001 and r['aa']['clean']['mde_fraction']==0
    assert 'total' not in r['contrasts'] and r['endpoint']['raw_ns_units']=='whole block'
    rows.pop(-5);save()
    assert not analyze(tmp_path)['quality_pass']
