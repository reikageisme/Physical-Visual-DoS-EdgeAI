"""Filesystem fixtures and fake scheduler only; never reads /proc or a phone."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies/arm_preflight_v2'))
import linux_observe as obs
from checks import Policy,thread_interval


def stat(tid,ticks=10,start=99,cpu=1):
    fields=['0']*37
    fields[0]='S';fields[11]=str(ticks);fields[12]='0';fields[19]=str(start);fields[36]=str(cpu)
    return f'{tid} (pool ) thread) '+ ' '.join(fields)


def task(root,tid,mask='1',**kw):
    p=root/str(tid);p.mkdir(exist_ok=True)
    (p/'stat').write_text(stat(tid,**kw));(p/'status').write_text('Name:\tpool\nCpus_allowed_list:\t'+mask+'\n')


def setup(monkeypatch,tmp_path,masks):
    monkeypatch.setattr(obs,'PROC_TASK',tmp_path)
    monkeypatch.setattr(obs.os,'sched_getaffinity',lambda tid:set(masks[tid]),raising=False)
    for tid,mask in masks.items():task(tmp_path,tid,','.join(map(str,sorted(mask))))


@pytest.mark.parametrize('text',['','1,,2','2-1','-1','1-2-3','x'])
def test_reject_malformed_cpu_list(text):
    with pytest.raises(ValueError):obs.cpu_list(text)


def test_cpu_list_ranges():
    assert obs.cpu_list('0-2,2,5,7-8')==[0,1,2,5,7,8]


def test_snapshot_keeps_idle_cpu_and_full_raw(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path,{101:{1},102:{1}})
    task(tmp_path,102,cpu=7)
    before=obs.collect_threads(clock=lambda:0)
    task(tmp_path,101,ticks=15)
    after=obs.collect_threads(clock=lambda:1)
    assert before['complete'] and after['complete']
    assert before['threads'][102]['last_cpu']==7
    assert 'Cpus_allowed_list' in before['threads'][102]['status']
    interval=thread_interval(before['threads'],after['threads'],{1})
    assert interval['complete'] and interval['active_share']==1 and interval['active_samples']==1


def test_status_scheduler_disagreement_is_not_complete(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path,{101:{1,2}})
    task(tmp_path,101,mask='1')
    frame=obs.collect_threads()
    assert not frame['complete'] and frame['errors']
    assert frame['threads'][101]['mask']==[1,2]  # retains evidence


def test_tid_reuse_during_snapshot_is_incomplete(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path,{101:{1}})
    def affinity(tid):
        task(tmp_path,tid,start=100)
        return {1}
    monkeypatch.setattr(obs.os,'sched_getaffinity',affinity,raising=False)
    frame=obs.collect_threads()
    assert not frame['complete']
    assert 'reused' in frame['errors'][0]['message']


def test_thread_born_during_snapshot_is_not_silently_dropped(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path,{101:{1}})
    def affinity(tid):
        task(tmp_path,102)
        return {1}
    monkeypatch.setattr(obs.os,'sched_getaffinity',affinity,raising=False)
    frame=obs.collect_threads()
    assert not frame['complete'] and frame['last_tids']==[101,102]


def test_all_tid_readback_success_is_not_a_load_pass(monkeypatch,tmp_path):
    masks={101:{0,1},102:{0,1}};setup(monkeypatch,tmp_path,masks)
    def set_mask(tid,mask):
        masks[tid]=set(mask);task(tmp_path,tid,mask='1')
    monkeypatch.setattr(obs.os,'sched_setaffinity',set_mask,raising=False)
    frame=obs.pin_all_current([1])
    assert frame['readback_complete']
    assert 'passed' not in frame  # no CPU-time/residency evidence from a mask readback


def test_all_tid_pin_one_pass_and_reports_partial_permission_failure(monkeypatch,tmp_path):
    masks={101:{0,1},102:{0,1}};setup(monkeypatch,tmp_path,masks);calls=[]
    def set_mask(tid,mask):
        calls.append(tid)
        if tid==102:raise PermissionError('denied')
        masks[tid]=set(mask);task(tmp_path,tid,mask='1')
    monkeypatch.setattr(obs.os,'sched_setaffinity',set_mask,raising=False)
    r=obs.pin_all_current([1])
    assert calls==[101,102] and not r['readback_complete']
    assert r['changes'][0]['readback']==[1] and r['changes'][1]['error']=='PermissionError'


def test_all_tid_pin_records_thread_churn_without_retry(monkeypatch,tmp_path):
    masks={101:{0,1}};setup(monkeypatch,tmp_path,masks);calls=[]
    def set_mask(tid,mask):
        calls.append(tid);masks[tid]=set(mask);task(tmp_path,tid,mask='1')
        masks[102]={1};task(tmp_path,102)
    monkeypatch.setattr(obs.os,'sched_setaffinity',set_mask,raising=False)
    r=obs.pin_all_current([1])
    assert calls==[101] and not r['readback_complete']
    assert r['after']['first_tids']==[101,102]


def sensors():
    return dict(zones=[dict(id='cpu',path='cpu/temp',type_path='cpu/type',type_value='cpu-thermal',unit='milli_celsius')],
                battery=dict(id='battery',path='battery/temp',type_path='battery/type',type_value='Battery',unit='deci_celsius'),
                soc_path='soc',status_path='status',charging_values=['Charging'],
                known_status_values=['Charging','Discharging','Full'])


def values():return {'cpu/temp':'50000','cpu/type':'cpu-thermal','battery/temp':'350',
                     'battery/type':'Battery','soc':'50','status':'Charging'}


def test_explicit_units_and_timestamp_at_start():
    data=values();ticks=iter([0,10])
    frame=obs.sensor_frame(sensors(),data.__getitem__,lambda:next(ticks))
    assert not frame['errors'] and frame['reading'].zones_c=={'cpu':50}
    assert frame['reading'].battery_c==35 and frame['reading'].sampled_s==0
    assert Policy(frozenset({'cpu'})).classify(frame['reading'],10)=='invalid_telemetry'


@pytest.mark.parametrize('key,value',[('cpu/temp','NaN'),('cpu/type','renamed'),
                                     ('status','Unknown'),('soc','101'),('battery/temp','-9999')])
def test_invalid_sensor_or_unknown_power_cannot_be_safe(key,value):
    data=values();data[key]=value
    frame=obs.sensor_frame(sensors(),data.__getitem__,lambda:0)
    assert frame['errors'] and frame['reading'] is None
    assert Policy(frozenset({'cpu'})).classify(frame['reading'],0)=='invalid_telemetry'


def test_missing_battery_does_not_fall_back_to_cpu():
    data=values();del data['battery/temp']
    frame=obs.sensor_frame(sensors(),data.__getitem__,lambda:0)
    assert frame['reading'] is None and frame['errors']


def test_known_discharging_is_power_gate():
    data=values();data['status']='Discharging'
    frame=obs.sensor_frame(sensors(),data.__getitem__,lambda:0)
    assert not frame['errors']
    assert Policy(frozenset({'cpu'})).classify(frame['reading'],0)=='power_gate'
