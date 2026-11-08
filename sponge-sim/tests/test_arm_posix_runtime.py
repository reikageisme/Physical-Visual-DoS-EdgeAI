"""Mock process groups/helpers; no real OS termination or phone access."""
import sys
import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies/arm_preflight_v2'))
import posix_runtime as rt
from checks import Guard,Policy,Reading,supervise_once


def mock_prctl(monkeypatch,parents=(123,123),set_result=0,get_result=0,readback=9):
    calls=[];ids=iter(parents)
    class Prctl:
        def __call__(self,option,value,*zeros):
            calls.append((option,value,*zeros))
            if option==1:return set_result
            ctypes.c_int.from_address(value).value=readback
            return get_result
    monkeypatch.setattr(rt.sys,'platform','linux')
    monkeypatch.setattr(rt.os,'getppid',lambda:next(ids))
    monkeypatch.setattr(rt.signal,'SIGKILL',9,raising=False)
    monkeypatch.setattr(ctypes,'CDLL',lambda *a,**k:SimpleNamespace(prctl=Prctl()))
    return calls


def test_parent_death_set_readback_and_both_parent_checks(monkeypatch):
    calls=mock_prctl(monkeypatch)
    assert rt.bind_parent_death(123) is None
    assert [call[0] for call in calls]==[1,2] and calls[0][1]==9


@pytest.mark.parametrize('parents',[(456,),(123,456)])
def test_parent_death_rejects_race_before_or_after_setup(monkeypatch,parents):
    calls=mock_prctl(monkeypatch,parents=parents)
    with pytest.raises(RuntimeError,match='parent changed'):rt.bind_parent_death(123)
    assert len(calls)==(0 if len(parents)==1 else 2)


@pytest.mark.parametrize('failure',[dict(set_result=-1),dict(get_result=-1)])
def test_parent_death_syscall_failure_has_no_fallback(monkeypatch,failure):
    mock_prctl(monkeypatch,**failure)
    with pytest.raises(OSError,match='PDEATHSIG failed'):rt.bind_parent_death(123)


def test_parent_death_wrong_readback_rejects(monkeypatch):
    mock_prctl(monkeypatch,readback=0)
    with pytest.raises(RuntimeError,match='readback'):rt.bind_parent_death(123)


@pytest.mark.parametrize('parent',[True,0,1,-1,'123'])
def test_parent_death_requires_captured_parent_identity(monkeypatch,parent):
    mock_prctl(monkeypatch)
    with pytest.raises(ValueError):rt.bind_parent_death(parent)


def test_own_group_launches_bootstrap_with_same_stdio_and_fresh_session(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(rt.sys,'platform','linux')
    def popen(args,**options):
        calls.append((args,options));return SimpleNamespace(pid=321)
    monkeypatch.setattr(rt.subprocess,'Popen',popen)
    child=rt.OwnGroup(['fixture-binary','--fixture'],tmp_path,stdout='FIXTURE_FD')
    args,options=calls[0]
    assert args[:3]==[sys.executable,str(Path(rt.__file__).resolve()),'--exec-bound']
    assert args[3:]==[str(rt.os.getpid()),'fixture-binary','--fixture']
    assert options['start_new_session'] and options['stdout']=='FIXTURE_FD'
    assert options['cwd']==tmp_path and child.pgid==321


def test_bootstrap_arms_before_exec_without_reading_stdin(monkeypatch):
    calls=[]
    monkeypatch.setattr(rt.sys,'argv',['runtime','--exec-bound','123','fixture-binary','--fixture'])
    monkeypatch.setattr(rt.sys,'stdin',object())
    monkeypatch.setattr(rt,'bind_parent_death',lambda p:calls.append(('armed',p)))
    def execvp(name,args):
        calls.append(('exec',name,args));raise InterruptedError('fixture exec')
    monkeypatch.setattr(rt.os,'execvp',execvp)
    with pytest.raises(InterruptedError,match='fixture exec'):rt.internal_main()
    assert calls==[('armed',123),('exec','fixture-binary',['fixture-binary','--fixture'])]


def test_failed_bootstrap_never_executes_payload(monkeypatch):
    monkeypatch.setattr(rt.sys,'argv',['runtime','--exec-bound','123','fixture-binary'])
    def fail(*a):raise RuntimeError('fixture setup failure')
    def forbidden(*a):raise AssertionError('payload must not run')
    monkeypatch.setattr(rt,'bind_parent_death',fail)
    monkeypatch.setattr(rt.os,'execvp',forbidden)
    with pytest.raises(RuntimeError,match='setup failure'):rt.internal_main()


def test_incomplete_bootstrap_never_reads_stdin(monkeypatch):
    monkeypatch.setattr(rt.sys,'argv',['runtime','--exec-bound','123'])
    monkeypatch.setattr(rt.sys,'stdin',object())
    with pytest.raises(ValueError,match='explicit command'):rt.internal_main()


def test_exited_leader_with_live_group_is_not_success():
    g=rt.OwnGroup.__new__(rt.OwnGroup)
    g.process=SimpleNamespace(poll=lambda:0);g.group_live=lambda:True
    with pytest.raises(RuntimeError,match='members remain'):g.poll()


def test_dead_group_is_never_signalled_again(monkeypatch):
    g=rt.OwnGroup.__new__(rt.OwnGroup);g.gone=True;g.pgid=123
    monkeypatch.setattr(rt.signal,'SIGKILL',9,raising=False)  # POSIX constant absent on test host Windows
    def forbidden(*a):raise AssertionError('must not address old PGID')
    monkeypatch.setattr(rt.os,'killpg',forbidden,raising=False)
    g.terminate();g.kill()


def test_timeout_exception_translation():
    g=rt.OwnGroup.__new__(rt.OwnGroup)
    def expired(timeout):raise rt.subprocess.TimeoutExpired('child',timeout)
    g.process=SimpleNamespace(wait=expired)
    with pytest.raises(TimeoutError,match='leader did not exit'):g.wait(.1)


def test_helper_timeout_stops_owned_group_and_closes_streams(monkeypatch,tmp_path):
    closed=[];stopped=[]
    class Child:
        def __init__(self,*a,**kw):
            stream=SimpleNamespace(close=lambda:closed.append(True))
            self.process=SimpleNamespace(stdin=stream,stdout=stream,stderr=stream,communicate=self.communicate)
        def communicate(self,*a,**kw):raise rt.subprocess.TimeoutExpired('helper',kw['timeout'])
    monkeypatch.setattr(rt,'OwnGroup',Child)
    monkeypatch.setattr(rt,'stop_group',lambda c:stopped.append(c))
    with pytest.raises(rt.subprocess.TimeoutExpired):rt.helper('--sensor',[],{},tmp_path,1)
    assert len(stopped)==1 and len(closed)==3


def test_event_ack_mismatch_and_exclusive_event_sequence(monkeypatch,tmp_path):
    requests=[]
    def helper(mode,args,payload,cwd,timeout):
        requests.append((mode,args,payload))
        return {'sha256':'wrong'}
    monkeypatch.setattr(rt,'helper',helper)
    emit=rt.BoundedEvents(tmp_path/'events',tmp_path,1)
    with pytest.raises(RuntimeError,match='ACK'):emit.emit({'kind':'gate'})
    assert emit.sequence==1 and requests[0][1][0].name=='000001.json'
    with pytest.raises(FileExistsError):rt.BoundedEvents(tmp_path/'events',tmp_path,1)


def test_sensor_error_keeps_raw_and_never_reuses_last_good_frame(monkeypatch,tmp_path):
    frames=iter([dict(errors=[],reading=dict(sampled_s=0,zones_c={'cpu':50},battery_c=35,battery_pct=50,charging=True)),
                 dict(errors=[{'error':'PermissionError'}],reading=None,raw={'sensor':'denied'})])
    def helper(mode,args,payload,*rest):
        frame=next(frames)
        return dict(frame=frame,capture_sha256=rt.json_sha(dict(sequence=payload['sequence'],config_sha256='sha',frame=frame)))
    monkeypatch.setattr(rt,'helper',helper)
    reader=rt.BoundedSensors(tmp_path/'spec.json','sha',tmp_path,1,tmp_path/'frames')
    assert reader.read().battery_c==35
    with pytest.raises(RuntimeError,match='invalid sensor'):reader.read()
    assert reader.last_frame['reading'] is None and reader.last_frame['raw']['sensor']=='denied'
    with pytest.raises(RuntimeError,match='new session required'):reader.read()


def test_sensor_timeout_clears_latest_and_poison_prevents_retry(monkeypatch,tmp_path):
    reader=rt.BoundedSensors(tmp_path/'spec.json','sha',tmp_path,1,tmp_path/'frames')
    reader.last_frame={'old':'good'}
    def fail(*a):raise rt.subprocess.TimeoutExpired('sensor',1)
    monkeypatch.setattr(rt,'helper',fail)
    with pytest.raises(rt.subprocess.TimeoutExpired):reader.read()
    assert reader.last_frame is None and reader.sequence==1 and reader.failed
    with pytest.raises(RuntimeError,match='new session'):reader.read()
    assert reader.sequence==1


def test_sensor_capture_bad_ack_is_fatal(monkeypatch,tmp_path):
    reader=rt.BoundedSensors(tmp_path/'spec.json','sha',tmp_path,1,tmp_path/'frames')
    monkeypatch.setattr(rt,'helper',lambda *a:dict(frame={'errors':[]},capture_sha256='wrong'))
    with pytest.raises(RuntimeError,match='capture ACK'):reader.read()
    assert reader.failed


def test_poll_exception_still_terminates_measurement_child():
    stopped=[]
    child=SimpleNamespace(poll=lambda:(_ for _ in ()).throw(RuntimeError('poll failed')),
                          terminate=lambda:stopped.append(True),wait=lambda timeout:None)
    g=Guard(Policy(frozenset({'cpu'})))
    read=lambda:Reading(0,{'cpu':50},35,50,True)
    with pytest.raises(RuntimeError,match='poll failed'):
        supervise_once(g,lambda:child,read,lambda e:None,lambda:0,lambda t:None)
    assert stopped==[True] and g.state=='END'
