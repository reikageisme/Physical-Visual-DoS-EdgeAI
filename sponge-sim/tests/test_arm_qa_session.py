"""Supervised QA contracts with fake children/clocks; no phone or Torch load."""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import pytest

SOURCE=Path(__file__).parents[1]/'studies/arm_preflight_v2'
sys.path.insert(0,str(SOURCE))
from checks import Guard,Policy,Reading
import pin_worker
import qa_session as qa
from session import SessionConfig


def cells():
    pairs=[(p,s) for p in ('pin_before_pool','pool_before_pin') for s in ('main','all')]
    random.Random(20261107).shuffle(pairs)
    return [dict(cell_id=f'{i}_{p}_{s}',allowed_cpus=[0],pool_order=p,pin_scope=s,torch_threads=2,
                 matrix_size=256,warmup_matmuls=20,intervals=32,interval_s=.1,
                 max_accounting_slack_fraction=.1,expected_torch_version='fake',
                 expected_torch_init_sha256='a'*64) for i,(p,s) in enumerate(pairs,1)]


def diagnostic(decision):
    return dict(status='diagnostic_recorded',profile_qualified=False,qa=dict(decision=decision,passed=decision=='PASS'),
                output_finite=True,checksum=1.)


class Clock:
    def __init__(self):self.t=0
    def __call__(self):return self.t
    def wait(self,t):self.t+=t


class Child:
    def __init__(self,code=0):self.code=code;self.gone=False;self.signals=[];self.pgid=123
    def poll(self):
        if self.code is not None:self.gone=True
        return self.code
    def terminate(self):
        if not self.gone:self.signals.append('TERM');self.code=-15;self.gone=True
    def kill(self):
        if not self.gone:self.signals.append('KILL');self.code=-9;self.gone=True
    def wait(self,timeout):return self.code


def simulate(codes,fail=None):
    clock=Clock();created=[];events=[];reads=[0]
    def spawn(cfg):
        child=Child(codes[len(created)]);created.append(child);return child
    def read():
        reads[0]+=1
        if fail=='sensor' and created:raise TimeoutError('fixture sensor deadline')
        return Reading(clock(),{'cpu':45},35,50,True)
    def emit(event):
        if fail=='logger' and event['kind']=='worker_started':raise TimeoutError('fixture logger deadline')
        if fail=='terminal' and event['kind']=='qa_terminal':raise TimeoutError('fixture terminal ACK')
        events.append(copy.deepcopy(event))
    guard=Guard(Policy(frozenset({'cpu'})))
    result=qa.run_cells(cells(),guard,spawn,read,emit,clock,clock.wait,SessionConfig(2,1))
    return result,created,events,guard


@pytest.mark.parametrize('decision',['PASS','FAIL','INCONCLUSIVE'])
def test_complete_diagnostic_exit_is_distinct_from_profile_qualification(decision):
    result=diagnostic(decision)
    assert pin_worker.child_exit_code(result)==0
    assert result['profile_qualified'] is False


def test_preparation_failure_and_malformed_success_are_nonzero():
    assert pin_worker.child_exit_code(dict(status='preparation_failed',profile_qualified=False))==65
    assert pin_worker.child_exit_code({})==65
    bad=diagnostic('PASS');bad['qa']['passed']=False
    assert pin_worker.child_exit_code(bad)==65
    bad=diagnostic('FAIL');bad['profile_qualified']=True
    assert pin_worker.child_exit_code(bad)==65


def test_execute_child_retains_preparation_failure_raw_and_nonzero_exit(monkeypatch,tmp_path):
    def failed(cfg,emit):
        result=dict(status='preparation_failed',reason='fixture_churn',profile_qualified=False)
        emit(dict(kind='cell_failed',result=result));return result
    monkeypatch.setattr(pin_worker,'execute_cell',failed)
    raw=tmp_path/'partial.jsonl'
    assert pin_worker.execute_child({},raw)==65
    assert json.loads(raw.read_text())['result']['status']=='preparation_failed'


def test_diagnostic_fail_data_proceeds_through_fixed_plan_without_promotion():
    result,created,events,guard=simulate([pin_worker.child_exit_code(diagnostic('FAIL')),0,0,0])
    assert len(created)==4 and result['status']=='control_flow_completed'
    assert all(b['rest']['elapsed_s']==60 for b in result['bursts'])
    assert result['terminal_journal_ack'] and result['qa_terminal_journal_ack']
    assert result['profile_qualified'] is False and result['study_quality']=='NOT_ASSESSED'
    assert events[-1]['kind']=='qa_terminal' and not any(c.signals for c in created)


def test_preparation_failure_stops_all_later_cells_without_retry():
    result,created,events,guard=simulate([65,0,0,0])
    assert len(created)==1 and guard.state=='END'
    assert result['bursts'][0]['status']=='aborted'
    assert [b['status'] for b in result['bursts'][1:]]==['not_started']*3
    assert result['reason']=='worker_error' and result['status']=='stopped_on_guard'


@pytest.mark.parametrize('failure',['sensor','logger'])
def test_reader_or_logger_failure_stops_owned_live_child_and_no_next_cell(failure):
    result,created,events,guard=simulate([None,0,0,0],failure)
    assert len(created)==1 and created[0].signals==['TERM'] and guard.state=='END'
    assert result['status']=='runtime_error' and result['profile_qualified'] is False


def test_post_cleanup_ack_failure_cannot_use_earlier_coordinator_ack_as_success():
    result,created,events,guard=simulate([0,0,0,0],'terminal')
    assert result['terminal_journal_ack'] and not result['qa_terminal_journal_ack']
    assert result['status']=='qa_journal_error' and guard.state=='END'


def test_unresolved_owned_group_cleanup_is_kept_as_error_not_success():
    clock=Clock();events=[];created=[]
    class Unstoppable(Child):
        def terminate(self):raise RuntimeError('fixture cleanup unavailable')
    def spawn(cfg):
        child=Unstoppable(None);created.append(child);return child
    def read():
        if created:raise TimeoutError('fixture reader')
        return Reading(clock(),{'cpu':45},35,50,True)
    guard=Guard(Policy(frozenset({'cpu'})))
    result=qa.run_cells(cells(),guard,spawn,read,events.append,clock,clock.wait,SessionConfig(2,1))
    assert len(created)==1 and result['status']=='qa_cleanup_error' and guard.state=='END'
    assert result['cleanup_errors'][0]['error']=='RuntimeError'
    assert events[-1]['kind']=='qa_terminal' and events[-1]['result']['cleanup_errors']


def test_reordered_or_mismatched_cells_rejected_before_any_spawn():
    plan=cells();plan.reverse()
    with pytest.raises(ValueError,match='order'):qa.validate_cells(plan)
    plan=cells();plan[1]['allowed_cpus']=[1]
    with pytest.raises(ValueError,match='match across'):qa.validate_cells(plan)


def sources(tmp_path):
    source=tmp_path/'source';source.mkdir()
    for name in qa.SOURCE_FILES:shutil.copyfile(SOURCE/name,source/name)
    return source,{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in qa.SOURCE_FILES}


def test_spawner_uses_exit_contract_and_locked_files_exactly_once(monkeypatch,tmp_path):
    source,hashes=sources(tmp_path);commands=[]
    def own(args,cwd,stdout):commands.append(args);return Child(0)
    monkeypatch.setattr(qa,'OwnGroup',own)
    plan=cells();clock=Clock()
    with qa.CellSpawner(plan,tmp_path,tmp_path/'run',source,hashes) as spawner:
        guard=Guard(Policy(frozenset({'cpu'})))
        result=qa.run_cells(plan,guard,spawner.spawn,lambda:Reading(clock(),{'cpu':45},35,50,True),
                            lambda e:None,clock,clock.wait,SessionConfig(2,1))
        assert len(commands)==4 and result['qa_terminal_journal_ack']
        assert all('SystemExit(execute_child' in cmd[2] for cmd in commands)
        with pytest.raises(RuntimeError,match='no retry'):spawner.spawn(plan[0])
    assert all(row['log'].closed for row in spawner.rows.values())


@pytest.mark.parametrize('tamper',['source','config'])
def test_source_or_cell_change_prevents_load_and_preserves_attempt(monkeypatch,tmp_path,tamper):
    source,hashes=sources(tmp_path);started=[]
    monkeypatch.setattr(qa,'OwnGroup',lambda *a,**k:started.append(True))
    plan=cells()
    with qa.CellSpawner(plan,tmp_path,tmp_path/'run',source,hashes) as spawner:
        target=source/'pin_worker.py' if tamper=='source' else spawner.rows[plan[0]['cell_id']]['config']
        target.write_text('changed',encoding='utf-8')
        with pytest.raises(ValueError):spawner.spawn(plan[0])
        with pytest.raises(RuntimeError,match='no retry'):spawner.spawn(plan[0])
    assert not started


def test_spawner_refuses_outside_workspace_and_old_run_directory(tmp_path):
    source,hashes=sources(tmp_path)
    with pytest.raises(ValueError,match='inside workspace'):
        qa.CellSpawner(cells(),tmp_path,tmp_path.parent/'outside_fixture',source,hashes)
    with qa.CellSpawner(cells(),tmp_path,tmp_path/'run',source,hashes):pass
    with pytest.raises(FileExistsError):qa.CellSpawner(cells(),tmp_path,tmp_path/'run',source,hashes)
