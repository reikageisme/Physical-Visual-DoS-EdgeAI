"""Pure parser/guard checks with synthetic bytes; no live device or load."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'studies/arm_preflight_v2'))
from checks import Guard, Policy
from termux_power import combine_live, parse_power


def raw(**updates):
    data = dict(present=True, health='GOOD', plugged='PLUGGED_USB',
                status='CHARGING', temperature=36.1, percentage=50,
                current=-763000, level=50, scale=100, energy=None)
    data.update(updates)
    return json.dumps(data, allow_nan=True).encode()


def live(power=None, **updates):
    args = dict(power_started_s=10, power_finished_s=11, zones_c={'cpu': 50},
                zones_started_s=11, zones_finished_s=12,
                policy=Policy(frozenset({'cpu'})), now_s=12)
    args.update(updates)
    return combine_live(power or parse_power(raw()), **args)


def test_celsius_is_not_divided_and_parsing_does_not_create_live_timestamp():
    data = raw()
    power = parse_power(data)
    assert power.battery_c == 36.1 and power.battery_pct == 50
    assert power.charging  # current sign cannot override known status
    assert power.raw_sha256 == hashlib.sha256(data).hexdigest()
    assert not hasattr(power, 'sampled_s')


@pytest.mark.parametrize('status', ['DISCHARGING', 'FULL', 'NOT_CHARGING'])
def test_plugged_but_not_charging_fails_existing_power_gate(status):
    power = parse_power(raw(status=status))
    assert not power.charging
    assert Policy(frozenset({'cpu'})).classify(live(power), 12) == 'power_gate'


def test_unplugged_low_soc_stays_observable_but_cannot_start_worker():
    power = parse_power(raw(status='DISCHARGING', plugged='UNPLUGGED', percentage=24))
    guard = Guard(Policy(frozenset({'cpu'})))
    assert guard.observe(live(power), 12) == 'power_gate'
    assert guard.state == 'END'


@pytest.mark.parametrize('changes', [
    {'present': False}, {'present': 1}, {'present': None}, {'health': 'UNKNOWN'},
    {'health': 'OVERHEAT'}, {'health': None}, {'status': 'UNKNOWN'}, {'status': []},
    {'plugged': 'PLUGGED_-1'}, {'plugged': []},
    {'plugged': 'UNPLUGGED', 'status': 'CHARGING'},
    {'percentage': -1}, {'percentage': 101}, {'percentage': None},
    {'percentage': True}, {'percentage': '50'}, {'temperature': '36,1'},
    {'temperature': '36.1'}, {'temperature': False}, {'temperature': None},
    {'temperature': -214748364.8}, {'temperature': float('nan')},
    {'temperature': float('inf')}, {'current': float('nan')},
])
def test_invalid_ambiguous_or_unhealthy_payload_rejects(changes):
    with pytest.raises(ValueError):
        parse_power(raw(**changes))


def test_missing_percentage_never_falls_back_to_level_scale():
    data = json.loads(raw()); del data['percentage']
    with pytest.raises(ValueError, match='missing'):
        parse_power(json.dumps(data).encode())


@pytest.mark.parametrize('data', [
    b'', b'[]', b'null', b'{"percentage":50,"percentage":25}', b'{', b'\xff',
    b'{"x":{"a":1,"a":2}}', b'x'*65537, '[not bytes]',
], ids=['empty', 'array', 'null', 'duplicate', 'partial', 'bad_utf8',
        'nested_duplicate', 'oversize_64k', 'not_bytes'])
def test_malformed_oversize_duplicate_or_nonbytes_rejects(data):
    with pytest.raises(ValueError):
        parse_power(data)


def test_huge_number_and_excessive_nesting_reject():
    with pytest.raises(ValueError):
        parse_power(raw(temperature=10**1000))
    with pytest.raises(ValueError):
        parse_power(b'['*2000 + b'0' + b']'*2000)


def test_exponent_overflow_in_unused_field_cannot_hide_nonfinite_json():
    with pytest.raises(ValueError):
        parse_power(raw().replace(b'"current": -763000', b'"current": 1e999'))


@pytest.mark.parametrize('battery,zones,expected', [
    (39.9, {'cpu': 69.9}, 'ok'), (40, {'cpu': 50}, 'hot'),
    (36.1, {'cpu': 70}, 'hot'), (361, {'cpu': 50}, 'hot'),
])
def test_existing_thresholds_apply_without_scaling_or_capping(battery, zones, expected):
    reading = live(parse_power(raw(temperature=battery)), zones_c=zones)
    assert Policy(frozenset({'cpu'})).classify(reading, 12) == expected


def test_freshness_keeps_earliest_start_and_does_not_alias_zone_dictionary():
    zones = {'cpu': 50}
    reading = live(zones_c=zones)
    zones['cpu'] = 90
    assert reading.zones_c == {'cpu': 50} and reading.sampled_s == 10
    assert Policy(frozenset({'cpu'})).classify(reading, 17) == 'invalid_telemetry'


@pytest.mark.parametrize('changes', [
    {'power_started_s': 0}, {'zones_started_s': 0},  # stale independently
    {'power_finished_s': 13}, {'zones_finished_s': 13},  # future
    {'power_started_s': 11.5}, {'zones_started_s': 12.5},  # reversed
    {'power_started_s': -1}, {'power_started_s': True}, {'now_s': float('nan')},
    {'zones_c': {}}, {'zones_c': {'gpu': 50}},
    {'zones_c': {'cpu': 50, 'gpu': 50}}, {'zones_c': {'cpu': float('nan')}},
    {'zones_c': {'cpu': -999}}, {'zones_c': {'cpu': '50'}},
])
def test_stale_future_missing_or_invalid_thermal_read_cannot_be_safe(changes):
    with pytest.raises(ValueError):
        live(**changes)


def test_battery_only_cannot_supply_missing_thermal_registry():
    with pytest.raises(ValueError):
        live(zones_c=None)
    with pytest.raises(ValueError):
        live(power=object())


def full_policy():
    return Policy(frozenset({'cpu'}), power_profile='full_100_external')


@pytest.mark.parametrize('status', ['CHARGING', 'FULL', 'NOT_CHARGING'])
def test_explicit_new_full100_profile_keeps_raw_status_and_legacy_still_rejects(status):
    power = parse_power(raw(percentage=100, status=status))
    reading = live(power)
    assert reading.power_status == status and reading.power_source == 'PLUGGED_USB'
    assert reading.charging == (status == 'CHARGING')
    assert full_policy().classify(reading, 12) == 'ok'
    assert Policy(frozenset({'cpu'})).classify(reading, 12) == 'power_gate'


@pytest.mark.parametrize('changes', [
    {'percentage': 99}, {'percentage': 80},
    {'status': 'DISCHARGING'}, {'status': 'FULL', 'plugged': 'UNPLUGGED'},
])
def test_new_full100_profile_rejects_drop_or_power_loss(changes):
    data = dict(percentage=100, status='FULL'); data.update(changes)
    assert full_policy().classify(live(parse_power(raw(**data))), 12) == 'power_gate'


@pytest.mark.parametrize('battery,cpu', [(40,50), (36,70)])
def test_full100_profile_does_not_bypass_thermal_thresholds(battery, cpu):
    reading = live(parse_power(raw(percentage=100, status='FULL', temperature=battery)),
                   zones_c={'cpu':cpu})
    assert full_policy().classify(reading, 12) == 'hot'


def test_full100_does_not_accept_legacy_missing_status_or_assumed_charging():
    from dataclasses import replace
    reading = live(parse_power(raw(percentage=100, status='FULL')))
    assert full_policy().classify(replace(reading,power_source=None), 12) == 'invalid_telemetry'
    assert full_policy().classify(replace(reading,power_status=None), 12) == 'invalid_telemetry'
    assert full_policy().classify(replace(reading,charging=True), 12) == 'invalid_telemetry'


def test_unknown_power_profile_is_not_a_fallback():
    with pytest.raises(ValueError, match='power profile'):
        Policy(frozenset({'cpu'}), power_profile='auto')
