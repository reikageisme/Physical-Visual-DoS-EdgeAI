"""Pure virtual-clock session tests: no real waits, process or sensor access."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies/arm_preflight_v2'))
from checks import Guard,Policy,Reading
from session import SessionConfig,run_session


class Clock:
    def __init__(self):self.t=0
    def __call__(self):return self.t
    def wait(self,seconds):self.t+=seconds


class Child:
    def __init__(self,code=0):self.code=code;self.stopped=False
    def poll(self):return self.code
    def terminate(self):self.stopped=True;self.code=-1
    def kill(self):self.stopped=True;self.code=-9
    def wait(self,timeout):return self.code


def simulate(bursts,read=None,emit=None,policy=None):
    clock=Clock();events=[]
    def safe():return Reading(clock(),{'a':40,'b':50},35,50,True)
    reader=(lambda:read(clock)) if read else safe
    sink=(lambda event:emit(event,events)) if emit else (lambda event:events.append(copy.deepcopy(event)))
    guard=Guard(policy or Policy(frozenset({'a','b'})))
    result=run_session(guard,bursts,reader,sink,clock,clock.wait,SessionConfig(2,1))
    return result,events,clock,guard


def test_safe_plan_preload_baseline_and_sixty_second_rest():
    spawned=[]
    def spawn():spawned.append(True);return Child()
    result,events,clock,_=simulate([('A',spawn),('B',spawn)])
    assert len(spawned)==2 and result['status']=='control_flow_completed'
    assert result['terminal_journal_ack'] and result['study_quality']=='NOT_ASSESSED'
    assert result['stable_power_observations']==3
    assert result['baseline']['sampled_s']==2
    assert all(b['rest']['elapsed_s']==60 and not b['rest']['not_cooled'] for b in result['bursts'])
    assert events[-1]['kind']=='session_terminal' and 'terminal_journal_ack' not in events[-1]['result']
    assert clock()==122


def test_rest_checks_each_sensor_instead_of_global_maximum():
    loaded=[False]
    def spawn():loaded[0]=True;return Child()
    def read(clock):
        # max temperature dropped, but sensor a is above its OWN baseline+1.
        return Reading(clock(),{'a':45 if loaded[0] else 40,'b':40 if loaded[0] else 50},35,50,True)
    result,_,_,_=simulate([('A',spawn)],read)
    assert result['bursts'][0]['rest']['elapsed_s']==180
    assert result['bursts'][0]['rest']['not_cooled']


def test_charge_drops_during_stability_window_no_worker_spawn():
    def forbidden():raise AssertionError('must not spawn')
    def read(clock):return Reading(clock(),{'a':40,'b':50},35,50,clock()<1)
    result,_,_,_=simulate([('A',forbidden)],read)
    assert result['status']=='preflight_rejected' and result['reason']=='power_gate'
    assert result['bursts'][0]['status']=='not_started'


def test_slow_first_read_is_not_an_entire_stable_charging_window():
    spawned=[]
    def read(clock):
        if not spawned and clock()<7:clock.t+=3
        return Reading(clock(),{'a':40,'b':50},35,50,True)
    def spawn():spawned.append(True);return Child()
    result,_,_,_=simulate([('A',spawn)],read)
    assert result['stable_power_observations']>=2
    assert result['baseline']['sampled_s']>=7


def test_long_gap_in_charging_observations_is_not_stable_power():
    calls=[0]
    def read(clock):
        calls[0]+=1
        if calls[0]==2:clock.t+=7
        return Reading(clock(),{'a':40,'b':50},35,50,True)
    result,_,_,_=simulate([('A',lambda:Child())],read)
    assert result['status']=='preflight_rejected' and result['reason']=='charging_observation_gap'
    assert result['bursts'][0]['status']=='not_started'


def test_initial_hot_is_not_retried_until_it_looks_cool():
    calls=[]
    def read(clock):calls.append(clock());return Reading(clock(),{'a':70,'b':50},35,50,True)
    result,_,_,_=simulate([('A',lambda:Child())],read)
    assert result['status']=='preflight_rejected' and result['reason']=='hot'
    assert len(calls)==1


def test_thermal_abort_keeps_partial_and_moves_to_next_fixed_id_after_rest():
    state={'spawned':0,'heated':False};children=[]
    def spawn():
        state['spawned']+=1;c=Child(None if state['spawned']==1 else 0);children.append(c);return c
    def read(clock):
        hot=state['spawned']==1 and not state['heated']
        if hot:state['heated']=True
        return Reading(clock(),{'a':70 if hot else 40,'b':50},35,50,True)
    result,_,_,guard=simulate([('A',spawn),('B',spawn)],read)
    assert state['spawned']==2 and children[0].stopped
    assert [b['status'] for b in result['bursts']]==['aborted','completed']
    assert result['bursts'][0]['reason']=='thermal_stop'
    assert result['status']=='control_flow_finished_with_incomplete_bursts'
    assert guard.stops==0  # only second completed burst reset it


def test_third_consecutive_interruption_stops_without_retries():
    state={'spawned':0,'hot_for':0};children=[]
    def spawn():
        state['spawned']+=1;c=Child(None);children.append(c);return c
    def read(clock):
        hot=state['spawned']>state['hot_for']
        if hot:state['hot_for']=state['spawned']
        return Reading(clock(),{'a':70 if hot else 40,'b':50},35,50,True)
    result,_,_,guard=simulate([(str(i),spawn) for i in range(4)],read)
    assert state['spawned']==3 and all(c.stopped for c in children)
    assert [b['status'] for b in result['bursts']]==['aborted','aborted','aborted','not_started']
    assert result['reason']=='third_thermal_stop' and guard.stops==3


def test_ordinary_rest_deadline_does_not_override_safety_cooldown():
    loaded=[False]
    def spawn():loaded[0]=True;return Child()
    def read(clock):return Reading(clock(),{'a':70 if loaded[0] else 40,'b':50},35,50,True)
    result,_,_,_=simulate([('A',spawn)],read)
    assert result['status']=='stopped_on_guard' and result['reason']=='cooldown_timeout'
    assert result['bursts'][0]['status']=='aborted'


def test_logger_failure_after_spawn_stops_owned_child():
    child=Child(None)
    def emit(event,events):
        if event['kind']=='worker_started':raise OSError('logger unavailable')
        events.append(copy.deepcopy(event))
    result,_,_,_=simulate([('A',lambda:child)],emit=emit)
    assert child.stopped and result['status']=='runtime_error'
    assert result['bursts'][0]['status']=='aborted'


def test_terminal_log_failure_cannot_return_control_success():
    def emit(event,events):
        if event['kind']=='session_terminal':raise OSError('no ACK')
        events.append(copy.deepcopy(event))
    result,_,_,_=simulate([('A',lambda:Child())],emit=emit)
    assert result['status']=='journal_error' and not result['terminal_journal_ack']
    assert result['bursts'][0]['status']=='completed'


def test_completed_burst_not_relabelled_aborted_by_later_rest_reader_error():
    done=[False]
    class Completed(Child):
        def poll(self):done[0]=True;return 0
    def read(clock):
        if done[0]:raise OSError('rest sensor failed')
        return Reading(clock(),{'a':40,'b':50},35,50,True)
    result,_,_,_=simulate([('A',lambda:Completed())],read)
    assert result['status']=='runtime_error' and result['bursts'][0]['status']=='completed'
    assert result['bursts'][0]['post_burst_error']['error']=='OSError'


def test_duplicate_burst_ids_rejected_before_any_reads():
    g=Guard(Policy(frozenset({'a'})));clock=Clock()
    def forbidden():raise AssertionError('must not read')
    with pytest.raises(ValueError,match='unique'):
        run_session(g,[('A',lambda:Child()),('A',lambda:Child())],forbidden,lambda e:None,clock,clock.wait,SessionConfig(2,1))


def test_successful_guard_cannot_be_reused_to_start_another_session():
    _,_,clock,guard=simulate([('A',lambda:Child())])
    def forbidden():raise AssertionError('must not read or spawn')
    with pytest.raises(ValueError,match='fresh guard'):
        run_session(guard,[('A',forbidden)],forbidden,lambda e:None,clock,clock.wait,SessionConfig(2,1))


def test_new_full100_profile_journals_policy_and_preserves_not_charging_status():
    def read(clock):
        return Reading(clock(),{'a':40,'b':50},35,100,False,
                       power_source='PLUGGED_USB',power_status='FULL')
    result,events,_,_=simulate([('A',lambda:Child())],read,
                              policy=Policy(frozenset({'a','b'}),power_profile='full_100_external'))
    assert result['status']=='control_flow_completed' and result['study_quality']=='NOT_ASSESSED'
    assert events[0]['policy']['power_profile']=='full_100_external'
    assert events[0]['policy']['expected_zones']==['a','b']
    assert result['baseline']['charging'] is False and result['baseline']['power_status']=='FULL'
    json.dumps(events,allow_nan=False)  # no frozenset in durable event payload


def test_new_full100_source_loss_stops_worker_and_no_later_burst():
    active=[False];child=Child(None)
    def spawn():active[0]=True;return child
    def read(clock):
        return Reading(clock(),{'a':40,'b':50},35,100,False,
                       power_source='UNPLUGGED' if active[0] else 'PLUGGED_USB',power_status='FULL')
    result,_,_,guard=simulate([('A',spawn),('B',lambda:Child())],read,
                             policy=Policy(frozenset({'a','b'}),power_profile='full_100_external'))
    assert child.stopped and guard.state=='END' and result['reason']=='power_gate'
    assert [r['status'] for r in result['bursts']]==['aborted','not_started']
