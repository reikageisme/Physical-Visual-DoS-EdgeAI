"""Prospective nested panel and audit subset remain fixed and non-overlapping."""
import json
from pathlib import Path


def test_nested_panel_and_real_start_variation():
    repo=Path(__file__).parents[2]
    folder=repo/'sponge-sim/studies/large_seed'
    cohort=json.loads((folder/'cohort1500.json').read_text())
    old=json.loads((repo/'sponge-sim/studies/validation_500/cohort.json').read_text())
    rows=cohort['images'];names=[r['file_name'] for r in rows]
    assert len(rows)==len(set(names))==1500
    assert names[:500]==[r['file_name'] for r in old['images']]
    assert not set(names[:1000])&set(names[1000:])
    assert sum(r['audit'] for r in rows)==64
    assert len(set(cohort['seeds']))==5 and cohort['seeds'][0]==0
    assert all(r['index']==i for i,r in enumerate(rows))
    lock=json.loads((folder/'execution_lock.json').read_text())
    assert all(r['batch']==4 and not r['fused'] for r in lock['selected'].values())
