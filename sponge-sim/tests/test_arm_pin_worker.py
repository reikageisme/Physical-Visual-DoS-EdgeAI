"""No Torch import/load: fake runtime plus clocks validate worker boundaries."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies/arm_preflight_v2'))
import pin_worker as worker


def config(order='pin_before_pool',scope='all'):
    return dict(cell_id='fixture',allowed_cpus=[1],pool_order=order,pin_scope=scope,torch_threads=2,
                matrix_size=256,warmup_matmuls=2,intervals=20,interval_s=.1,
                max_accounting_slack_fraction=.1,expected_torch_version='fake',expected_torch_init_sha256='a'*64)


def fake_runtime(log):
    tensor=SimpleNamespace(sum=lambda:SimpleNamespace(item=lambda:1))
    generator=SimpleNamespace(manual_seed=lambda seed:None)
    return SimpleNamespace(__version__='fake',__file__='fixture-only',float32='float32',
        __config__=SimpleNamespace(parallel_info=lambda:'fake'),
        Generator=lambda **k:generator,rand=lambda *a,**k:tensor,empty_like=lambda a:tensor,
        inference_mode=nullcontext,set_num_threads=lambda n:log.append('threads'),
        set_num_interop_threads=lambda n:None,get_num_threads=lambda:2,get_num_interop_threads=lambda:1,
        mm=lambda *a,**k:log.append('warmup'))


@pytest.mark.parametrize('order',['pin_before_pool','pool_before_pin'])
@pytest.mark.parametrize('scope',['main','all'])
def test_pin_import_warmup_boundary_and_no_repin_in_capture(monkeypatch,order,scope):
    log=[];events=[]
    monkeypatch.delitem(sys.modules,'torch',raising=False)
    monkeypatch.setattr(worker.os,'sched_getaffinity',lambda tid:{0,1},raising=False)
    monkeypatch.setattr(worker.os,'sysconf',lambda key:100,raising=False)
    monkeypatch.setattr(worker,'collect_threads',lambda **kw:dict(complete=True))
    monkeypatch.setattr(worker,'pin_current',lambda cpus:log.append('pin_main'))
    monkeypatch.setattr(worker,'pin_all_current',lambda cpus:(log.append('pin_all') or dict(readback_complete=True)))
    monkeypatch.setattr(worker,'load_torch',lambda cfg:(log.append('import') or fake_runtime(log)))
    def capture(*args,**kwargs):
        log.append('capture')
        return [],dict(window_start_s=0,window_end_s=1,process_cpu_s=1,ticks_hz=100)
    monkeypatch.setattr(worker,'capture_window',capture)
    monkeypatch.setattr(worker,'analyze',lambda *a,**kw:dict(passed=False,decision='INCONCLUSIVE'))
    r=worker.execute_cell(config(order,scope),events.append)
    if order=='pin_before_pool':assert log.index('pin_main')<log.index('import')<log.index('warmup')
    else:assert log.index('import')<log.index('warmup')<log.index('pin_main')
    if scope=='all':assert log.index('warmup')<log.index('pin_all')<log.index('capture')
    else:assert 'pin_all' not in log
    assert log.count('pin_main')==1 and log[-1]=='capture'
    assert r['status']=='diagnostic_recorded' and not r['profile_qualified']


def test_already_imported_torch_rejected_before_any_load(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',object())
    with pytest.raises(RuntimeError,match='fresh process'):
        worker.execute_cell(config(),lambda e:None)


def test_runtime_fingerprint_mismatch_fails_before_work(monkeypatch,tmp_path):
    path=tmp_path/'init.py';path.write_text('fixture')
    fake=SimpleNamespace(__file__=str(path),__version__='fake')
    monkeypatch.setattr(worker.importlib,'import_module',lambda name:fake)
    with pytest.raises(RuntimeError,match='changed since lock'):worker.load_torch(config())


@pytest.mark.parametrize('key,value',[('torch_threads',1),('allowed_cpus',[0,1]),
                                      ('interval_s',float('nan')),('intervals',0),
                                      ('expected_torch_init_sha256','UNLOCKED')])
def test_invalid_or_unlocked_config_cannot_start(key,value):
    cfg=config();cfg[key]=value
    with pytest.raises(ValueError):worker.validate(cfg)


def test_capture_brackets_frames_and_persists_each_interval():
    t=[0];events=[];operations=[]
    def wall():return t[0]
    def cpu():return t[0]//2
    def work():t[0]+=60_000_000;operations.append(True)
    def observe(clock):
        start=clock();t[0]+=1_000_000
        return dict(started_s=start,finished_s=clock())
    cfg=config();cfg['intervals']=2
    frames,r=worker.capture_window(work,cfg,observe,events.append,wall_ns=wall,cpu_ns=cpu,ticks_hz=100)
    assert r['matmul_counts']==[2,2] and len(frames)==3 and len(operations)==4
    assert r['window_start_s']<=frames[0]['started_s']<frames[-1]['finished_s']<=r['window_end_s']
    assert r['process_cpu_s']==(r['process_cpu_end_ns']-r['process_cpu_start_ns'])/1e9
    assert [e['index'] for e in events if e['kind']=='thread_snapshot']==[0,1,2]
    assert r['snapshot_and_logging_overhead_included']


def test_partial_raw_is_retained_and_existing_run_is_never_overwritten(monkeypatch,tmp_path):
    def failed(cfg,emit):
        emit(dict(kind='thread_snapshot',index=0));raise RuntimeError('interrupted fixture')
    monkeypatch.setattr(worker,'execute_cell',failed)
    raw=tmp_path/'raw.jsonl'
    with pytest.raises(RuntimeError):worker.execute_to_raw({},raw)
    rows=[json.loads(s) for s in raw.read_text().splitlines()]
    assert [r['kind'] for r in rows]==['thread_snapshot','cell_exception']
    before=raw.read_bytes()
    with pytest.raises(FileExistsError):worker.execute_to_raw({},raw)
    assert raw.read_bytes()==before


def test_nonfinite_output_remains_json_serializable_diagnostic_failure(monkeypatch,tmp_path):
    monkeypatch.delitem(sys.modules,'torch',raising=False)
    monkeypatch.setattr(worker.os,'sched_getaffinity',lambda tid:{0,1},raising=False)
    monkeypatch.setattr(worker.os,'sysconf',lambda key:100,raising=False)
    monkeypatch.setattr(worker,'collect_threads',lambda **kw:dict(complete=True))
    monkeypatch.setattr(worker,'pin_current',lambda cpus:None)
    monkeypatch.setattr(worker,'pin_all_current',lambda cpus:dict(readback_complete=True))
    runtime=fake_runtime([])
    out=runtime.empty_like(None);out.sum=lambda:SimpleNamespace(item=lambda:float('nan'))
    runtime.empty_like=lambda a:out
    monkeypatch.setattr(worker,'load_torch',lambda cfg:runtime)
    monkeypatch.setattr(worker,'capture_window',lambda *a,**k:([],dict(window_start_s=0,window_end_s=1,process_cpu_s=1,ticks_hz=100)))
    monkeypatch.setattr(worker,'analyze',lambda *a,**k:dict(passed=True,decision='PASS'))
    raw=tmp_path/'nonfinite.jsonl'
    result=worker.execute_to_raw(config(),raw)
    assert result['qa']['decision']=='FAIL' and result['qa']['reason']=='nonfinite_matrix_output'
    assert result['checksum'] is None and result['output_finite'] is False
    assert json.loads(raw.read_text().splitlines()[-1])['kind']=='cell_done'
    assert worker.child_exit_code(result)==0  # completed diagnostic FAIL, never profile PASS
