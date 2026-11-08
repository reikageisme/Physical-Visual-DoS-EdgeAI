"""Synthetic snapshots only; no thread/load/device measurements."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies/arm_preflight_v2'))
from pin_analysis import analyze


def fixtures():
    frames=[]
    for i in range(21):
        frames.append(dict(started_s=i*.01,finished_s=i*.01+.001,complete=True,errors=[],threads={
            1:dict(starttime=1,ticks=i*10,last_cpu=1,mask=[1],state='R'),
            2:dict(starttime=2,ticks=0,last_cpu=7,mask=[1],state='S')}))
    return frames


def run(frames=None,cpu=.2,**kw):
    return analyze(fixtures() if frames is None else frames,[1],0,.201,cpu,1000,
                   max_accounting_slack_fraction=kw.get('slack',.1))


def test_idle_cpu_artifact_not_load_pin_failure_when_all_other_gates_met():
    r=run()
    assert r['passed'] and r['decision']=='PASS'
    assert r['active_samples']==20 and r['active_share']==1
    assert r['all_tid_last_cpu_share_same_snapshots']==.5


def test_wide_idle_mask_still_fails_even_with_perfect_active_share():
    f=fixtures()
    for frame in f:frame['threads'][2]['mask']=[1,7]
    r=run(f)
    assert r['active_share']==1 and not r['masks_ok'] and r['decision']=='FAIL'


def test_upper_cpu_violation_independent_of_sampler_is_still_fail():
    r=run(cpu=.4)
    assert r['active_share']==1 and not r['upper_ok'] and r['decision']=='FAIL'


def test_missing_accounting_cannot_be_imputed_to_success():
    r=run(cpu=.22)  # CPU not present in per-TID counters, outside bounded slack
    assert r['upper_ok'] and r['lower_ok'] and not r['accounting']['passed']
    assert r['decision']=='INCONCLUSIVE' and not r['passed']


def test_too_few_active_samples_and_tid_reuse_are_inconclusive():
    f=fixtures()
    for frame in f[1:]:frame['threads'][1]['ticks']=10
    assert run(f,cpu=.11)['decision']=='INCONCLUSIVE'
    f=fixtures();f[-1]['threads'][1]['starttime']=10
    assert run(f)['decision']=='INCONCLUSIVE'


def test_partial_or_malformed_snapshots_never_become_pass():
    f=fixtures();f[2]['complete']=False
    assert run(f)['decision']=='INCONCLUSIVE'
    f=fixtures();del f[2]['threads'][1]['mask']
    assert run(f)['decision']=='INCONCLUSIVE'


def test_huge_accounting_tolerance_cannot_hide_missing_coverage():
    f=fixtures();f[0]['finished_s']=.009;f[-1]['started_s']=.191
    r=run(f)
    assert r['accounting']['passed'] is False and r['decision']=='INCONCLUSIVE'


def test_boolean_timestamps_or_thread_counters_are_not_valid_raw():
    f=fixtures();f[0]['started_s']=False
    assert run(f)['decision']=='INCONCLUSIVE'
    f=fixtures();f[2]['threads'][1]['ticks']=True
    assert run(f)['decision']=='INCONCLUSIVE'
