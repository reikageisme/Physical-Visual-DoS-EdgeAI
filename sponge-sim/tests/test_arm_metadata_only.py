"""Fixture metadata only, no host sysfs/package inspection or Torch import."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies/arm_preflight_v2'))
import metadata_only as meta


def test_metadata_never_imports_torch_or_converts_temperature(monkeypatch,tmp_path):
    monkeypatch.delitem(sys.modules,'torch',raising=False)
    package=tmp_path/'init.py';package.write_text('fixture')
    monkeypatch.setattr(meta.os,'sched_getaffinity',lambda tid:{1},raising=False)
    monkeypatch.setattr(meta.os,'sysconf',lambda key:100,raising=False)
    monkeypatch.setattr(meta,'collect_threads',lambda:dict(complete=True))
    monkeypatch.setattr(meta.importlib.metadata,'version',lambda name:'fixture')
    monkeypatch.setattr(meta.importlib.util,'find_spec',lambda name:SimpleNamespace(origin=str(package)))
    monkeypatch.setattr(meta,'inspect_sysfs',lambda:(_ for _ in ()).throw(AssertionError('not opted in')))
    result=meta.collect(False)
    assert not result['torch_imported'] and not result['sysfs_requested'] and 'torch' not in sys.modules
    assert len(result['packages']['torch']['init_sha256'])==64
    assert result['status']=='METADATA_ONLY_NOT_LOAD_AUTHORIZATION'


def test_sysfs_raw_keeps_denied_missing_and_does_not_guess_units(monkeypatch,tmp_path):
    thermal=tmp_path/'thermal';power=tmp_path/'power';thermal.mkdir();power.mkdir()
    zone=thermal/'thermal_zone0';zone.mkdir();(zone/'type').write_text('fixture-cpu');(zone/'temp').write_text('50000')
    battery=power/'battery';battery.mkdir();(battery/'type').write_text('Battery')
    monkeypatch.setattr(meta,'THERMAL',thermal);monkeypatch.setattr(meta,'POWER',power)
    r=meta.inspect_sysfs()
    assert r['thermal'][0]['nodes']['temp']['raw']=='50000'
    assert not r['power'][0]['nodes']['temp']['readable'] and not r['temperature_conversion_performed']
    assert 'REVIEW' in r['power'][0]['unit_contract']


def test_preloaded_torch_is_not_a_valid_metadata_only_invocation(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',object())
    with pytest.raises(RuntimeError,match='fresh process'):meta.collect()
