"""Degenerate bootstrap outcomes retain informative Wilson proportion intervals."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('val500_statistics',Path(__file__).parents[1]/'studies/validation_500/analyze.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_all_success_and_failure_are_not_zero_width_proportions():
    positive=mod.summary([0]*4,[300]*4)
    negative=mod.summary([1]*4,[0]*4)
    assert positive['attacked_median_ci95']==[300,300]
    assert positive['reach']['300']['wilson_ci95'][0]<1
    assert positive['reach']['300']['wilson_ci95'][1]==1
    assert negative['reach']['50']['wilson_ci95'][0]<1e-12
    assert negative['reach']['50']['wilson_ci95'][1]>0
