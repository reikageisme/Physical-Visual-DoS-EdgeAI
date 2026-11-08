"""Two-P-core ownership, topology and profile-gate contracts without load."""
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'studies/laptop_seg'))
import pcores
from audit_pcores import verify_profile


class Proc:
    def __init__(self, pid, mask):self.pid=pid;self.mask=mask[:];self.children_list=[]
    def create_time(self):return float(self.pid)
    def cpu_affinity(self, mask=None):
        if mask is not None:self.mask=mask[:]
        return self.mask[:]
    def children(self, recursive=False):return self.children_list[:]
    def name(self):return 'owned-fixture'
    def is_running(self):return True
    def parents(self):return []


def separation(monkeypatch, child_mask):
    root=Proc(10,list(range(12)));child=Proc(11,child_mask);root.children_list=[child]
    monkeypatch.setattr(pcores.separate_app.psutil,'Process',lambda pid:{10:root,11:child}[pid])
    sep=pcores.PcoreSeparation.__new__(pcores.PcoreSeparation)
    sep.root=root;sep.records={};sep.file=io.StringIO()
    return sep,root,child


def test_both_p_cores_and_siblings_excluded_then_original_masks_restored(monkeypatch):
    sep,root,child=separation(monkeypatch,list(range(12)))
    sep.apply()
    events=[json.loads(x) for x in sep.file.getvalue().splitlines()]
    assert root.mask==child.mask==list(range(4,12))
    assert events[-1]['excluded']==[0,1,2,3]
    assert pcores.separate_app.restricted(list(range(12)))==[0,1,*range(4,12)]
    sep.restore()
    assert root.mask==child.mask==list(range(12))


def test_partial_preparation_failure_restores_changed_owned_root(monkeypatch):
    sep,root,child=separation(monkeypatch,[0])
    with pytest.raises(RuntimeError):sep.apply()
    assert root.mask==list(range(4,12)) and child.mask==[0]
    sep.restore()
    assert root.mask==list(range(12)) and child.mask==[0]


def test_physical_p_core_mapping_and_profile_cpu_checks():
    topology=[dict(kind='P',logical_cpus=[0,1]),dict(kind='P',logical_cpus=[2,3])]
    assert pcores.verify_topology(topology)==([0,2],[0,1,2,3])
    with pytest.raises(RuntimeError):pcores.verify_topology([dict(kind='P',logical_cpus=[0,2])])
    qos=dict(control_mask=1,state_mask=0,priority_class=32768)
    lock=dict(profiles={'P2':[0,2]},pcores_topology=topology,power_mode={'fixed':'fixture'})
    m=dict(cpu_set=[0,2],torch_threads=2,interop_threads=1,hardware=topology,
           power_start={'power_plugged':True},power_mode_start=lock['power_mode'],priority_start=32768,qos_start=qos)
    done=dict(affinity=[0,2],power_end={'power_plugged':True},power_mode_end=lock['power_mode'],qos_end=qos)
    verify_profile(m,done,lock)
    m['cpu_set']=[0,1]
    with pytest.raises(AssertionError):verify_profile(m,done,lock)
