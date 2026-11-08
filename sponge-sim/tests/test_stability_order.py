"""Balanced stage positions and paired arm order over the declared 16 repeats."""
import importlib.util
from pathlib import Path
from collections import Counter
import numpy as np

spec=importlib.util.spec_from_file_location('stability_pipeline',Path(__file__).parents[1]/'studies/stability_pipeline/run.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_balanced_stage_positions_and_arms():
    jobs=[(arm,str(stage)) for arm in ['clean','attacked'] for stage in range(8)]
    orders=mod.balanced_orders(jobs,16,20261106)
    positions=Counter();first=Counter()
    for order in orders:
        assert sorted(order)==list(range(16))
        for pos in range(0,16,2):
            a,b=jobs[order[pos]],jobs[order[pos+1]]
            assert a[1]==b[1] and a[0]!=b[0]
            positions[a[1],pos//2]+=1;first[a[1],a[0]]+=1
    assert set(positions.values())=={2}
    assert set(first.values())=={8}


def test_aggregate_uses_all_three_sessions_with_registered_median():
    spec=importlib.util.spec_from_file_location('stability_pool',Path(__file__).parents[1]/'studies/stability_pipeline/analyze.py')
    analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
    result=analysis.pool_logs(np.log([[2,2,2,2],[4,4,4,4],[8,8,8,8]]))
    np.testing.assert_allclose(result['geometric_ratio'],4,rtol=1e-14)
    assert result['n_sessions']==3
