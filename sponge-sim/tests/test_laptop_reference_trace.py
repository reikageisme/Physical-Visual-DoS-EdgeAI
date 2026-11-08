"""Finite-cadence schedule and observer placement without Torch/model work."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'studies/laptop_seg'))
from reference_trace import trace_blocks, validate


def config():
    return dict(blocks=12, samples=15, warmup=3, pin_seconds=3, settle_s=60,
                worker_timeout_s=600, seed=20261107, pauses_s={'continuous':0, 'pause2':2})


def test_fixed_schedule_and_observers_outside_model_timer():
    events=[];rows=[];sleeps=[]
    def work():events.append('work')
    def measure(fn, **kw):
        assert kw==dict(capture_cpu=True,keep_output=False)
        events.append('timer_start');fn();events.append('timer_end')
        return 100, 200, None, 90
    def collect():events.append('pdh');return {'saved':True}
    trace_blocks(config(),'pause2',work,measure,collect,lambda:events.append('check'),
                 lambda **r:rows.append(r),lambda:events.append('flush'),sleeps.append)
    assert events.count('work')==216
    assert len([r for r in rows if r['kind']=='timing'])==180
    assert len([r for r in rows if r['kind']=='reference'])==12
    assert len([r for r in rows if r['kind']=='pdh_block_initial'])==12
    assert sleeps==[2]*11
    for i,e in enumerate(events):
        if e=='timer_start':assert events[i:i+4]==['timer_start','work','timer_end','pdh']
    assert all(r['initial_ratio']==1 for r in rows if r['kind']=='reference')


def test_no_automatic_worker_retry_or_partial_block_replacement():
    calls=[];rows=[]
    def broken(*a,**k):calls.append('measure');raise RuntimeError('fixture observer failure')
    with pytest.raises(RuntimeError):
        trace_blocks(config(),'continuous',lambda:None,broken,lambda:{},lambda:None,
                     lambda **r:rows.append(r),lambda:None,lambda _:None)
    assert calls==['measure']
    assert [r['kind'] for r in rows]==['pdh_block_initial']
    altered=config();altered['warmup']=40
    with pytest.raises(ValueError):validate(altered)
    assert sorted(validate(config()))==['continuous','pause2']
