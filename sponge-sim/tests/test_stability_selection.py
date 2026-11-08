"""Selection cannot favor a large effect: only controls/quality are inputs."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('stability',Path(__file__).parents[1]/'studies/stability_calibration/analyze.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_quality_selection_and_no_eligible():
    profiles=dict(all=dict(eligible=True,flags=1,mde_frac=.001),
                  four=dict(eligible=True,flags=0,mde_frac=.002),
                  one=dict(eligible=False,flags=0,mde_frac=.0001))
    assert mod.choose(profiles)=='four'
    assert mod.choose({k:dict(v,eligible=False) for k,v in profiles.items()}) is None
