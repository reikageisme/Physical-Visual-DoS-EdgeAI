"""Synthetic control-flow tests only. No device, torch, load or timing data."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'studies/arm_preflight_v2'))
from checks import Guard, Policy, Reading, parse_stat, supervise_once, thread_interval


def reading(t=0, cpu=50, bat=35, soc=50, charging=True):
    return Reading(t, {'cpu0': cpu}, bat, soc, charging)


def guard(**kw):
    return Guard(Policy(frozenset({'cpu0'}), **kw))


def test_stat_handles_spaces_and_parentheses():
    f = ['0']*37
    f[0], f[11], f[12], f[19], f[36] = 'S', '12', '7', '99', '6'
    assert parse_stat('42 (pool ) thread) '+' '.join(f)) == dict(state='S', ticks=19, starttime=99, last_cpu=6)


def thread(ticks, cpu, mask=(5,), start=99):
    return dict(ticks=ticks, last_cpu=cpu, mask=mask, starttime=start, state='S')


def test_idle_last_cpu_is_not_active_residency():
    a = {1: thread(10, 5), 2: thread(10, 0)}
    b = {1: thread(15, 5), 2: thread(10, 0)}
    r = thread_interval(a, b, {5})
    assert r['active_share'] == 1 and r['masks_ok'] and r['complete']
    assert r['active_samples'] == 1  # old sampler would report 1/2 inside


def test_idle_thread_with_bad_mask_still_fails():
    a = {1: thread(10, 5), 2: thread(10, 0, (0, 5))}
    b = {1: thread(15, 5), 2: thread(10, 0, (0, 5))}
    assert not thread_interval(a, b, {5})['masks_ok']


def test_churn_and_tid_reuse_are_inconclusive():
    a = {1: thread(10, 5)}
    assert not thread_interval(a, {2: thread(10, 5)}, {5})['complete']
    assert not thread_interval(a, {1: thread(11, 5, start=100)}, {5})['complete']
    assert thread_interval(a, a, {5})['active_share'] is None


@pytest.mark.parametrize('cpu,bat', [(70, 35), (50, 40), (70, 40)])
def test_hot_gate_before_start(cpu, bat):
    g = guard()
    assert g.observe(reading(cpu=cpu, bat=bat), 0) == 'thermal_stop'


def test_hysteresis_and_three_consecutive_stops():
    g = guard()
    for stop in range(1, 4):
        assert g.observe(reading(cpu=70), 0) == ('third_thermal_stop' if stop == 3 else 'thermal_stop')
        assert g.stops == stop
        if stop < 3:
            assert g.observe(reading(cpu=55, bat=38), 0) == 'cooling'
            assert g.observe(reading(cpu=54.9, bat=37.9), 0) == 'cooled'
    assert g.state == 'END'


def test_only_safe_completed_burst_resets_stops():
    g = guard()
    g.observe(reading(cpu=70), 0)
    with pytest.raises(RuntimeError):
        g.completed()
    g.observe(reading(), 0)
    assert g.stops == 1
    g.completed()
    assert g.stops == 0


@pytest.mark.parametrize('r,now', [(None, 0), (reading(cpu=float('nan')), 0),
                                  (reading(), 7), (reading(t=1), 0),
                                  (Reading(0, {}, 35, 50, True), 0)])
def test_missing_stale_nonfinite_data_ends_session(r, now):
    g = guard()
    assert g.observe(r, now) == 'invalid_telemetry'
    assert g.state == 'END'


@pytest.mark.parametrize('soc,charging', [(29.99, True), (80.01, True), (50, False)])
def test_power_gate(soc, charging):
    assert guard().observe(reading(soc=soc, charging=charging), 0) == 'power_gate'


def test_cooldown_has_deadline():
    g = guard(max_cooldown_s=10)
    g.observe(reading(cpu=70), 0)
    assert g.observe(reading(t=10, cpu=60), 10) == 'cooldown_timeout'


class Child:
    def __init__(self, stuck=False, code=None):
        self.terminated = self.killed = False
        self.stuck, self.code = stuck, code

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        if self.stuck and not self.killed:
            raise TimeoutError()
        self.code = -1

    def kill(self):
        self.killed = True


def simulate(g, samples, child=None, emit=None):
    events, spawned = [], []
    t = [0]
    rs = iter(samples)
    def spawn():
        spawned.append(True)
        return child
    def wait(_):
        t[0] += 1
    result = supervise_once(g, spawn, lambda: next(rs), emit or events.append, lambda: t[0], wait)
    return result, events, spawned


def test_prestart_hot_does_not_spawn():
    r, _, spawned = simulate(guard(), [reading(cpu=70)])
    assert r == 'thermal_stop' and not spawned


def test_battery_stop_interrupts_active_child_and_keeps_abort_event():
    c = Child(stuck=True)
    r, events, _ = simulate(guard(), [reading(), reading(), reading(t=1, bat=40)], c)
    assert r == 'thermal_stop' and c.terminated and c.killed
    assert events[-1]['kind'] == 'aborted'


def test_timeout_and_child_error_are_not_success():
    c = Child()
    r, _, _ = simulate(guard(worker_timeout_s=1), [reading(), reading(), reading(t=1)], c)
    assert r == 'worker_timeout' and c.terminated
    r, _, _ = simulate(guard(), [reading(), reading()], Child(code=7))
    assert r == 'worker_error'


def test_telemetry_exception_terminates_child():
    c = Child()
    with pytest.raises(StopIteration):
        simulate(guard(), [reading()], c)
    assert c.terminated


def test_safe_child_success_is_explicit():
    r, events, _ = simulate(guard(), [reading(), reading()], Child(code=0))
    assert r == 'completed' and events[-1]['kind'] == 'completed'
