"""E11: hai nhánh (sạch và có patch) phải có CÙNG tập thao tác trong vùng bấm giờ, và bước chèn patch nằm ngoài vùng đó.

Lỗi được sửa: E6 (studies/e6_remeasure/run.py:198-205) gọi insert trong vùng đo chỉ ở nhánh có patch.
"""
import importlib.util
from pathlib import Path

import torch

from sponge_sim import attacks_provenance as ap_
from sponge_sim import provenance as pv
from sponge_sim.backends import BACKENDS

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('e11_run', ROOT / 'studies/e11_patch_remeasure/run.py')
e11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e11)


class Trace:
    def __init__(self):
        self.ops = []


def _spies(trace):
    raw = torch.zeros(1, 84, 2100)
    raw[0, :4, :50] = torch.rand(4, 50) * 300
    raw[0, 4, :50] = torch.linspace(.02, .9, 50)

    class Model:
        def __call__(self, img):
            trace.ops.append(('model', tuple(img.shape)))
            return (raw,)

    class PV:
        @staticmethod
        def decode(r):
            trace.ops.append(('decode', tuple(r.shape)))
            return pv.decode(r)

        @staticmethod
        def tau_filter(d, tau, cap):
            trace.ops.append(('tau_filter', tau, cap))
            return pv.tau_filter(d, tau, cap)

    class Backend:
        def __init__(self, b):
            self.b = b

        def prepare(self, *args):
            trace.ops.append(('prepare',))
            return self.b.prepare(*args)

        def run(self, prepared, iou):
            trace.ops.append(('run', iou))
            return self.b.run(prepared, iou)

    def insert(*args):
        trace.ops.append(('insert',))
        return ap_.insert(*args)
    return Model(), PV, Backend, insert


def _timed_ops(arm, backend_name):
    trace = Trace()
    model, PV, Backend, insert = _spies(trace)
    base = torch.rand(1, 3, 320, 320)
    patch = torch.rand(3, 64, 64)
    img = e11.build_input(arm, base, patch, (100, 80), insert)
    built = list(trace.ops)
    trace.ops.clear()
    e11.pipeline_fn(model, img, Backend(BACKENDS[backend_name]), .45, PV, torch)()
    return built, trace.ops, img


def test_same_timed_ops_clean_vs_patched():
    for be in ('torch_pure', 'tv_nms'):
        built_c, ops_c, img_c = _timed_ops('clean_a', be)
        built_p, ops_p, img_p = _timed_ops('C1', be)
        assert built_c == [] and built_p == [('insert',)]  # chèn chỉ xảy ra khi dựng đầu vào
        assert ops_c == ops_p  # cùng chuỗi thao tác trong vùng đo
        assert ('insert',) not in ops_p
        assert img_c.shape == img_p.shape


def test_timed_function_has_no_insert_reference():
    full = e11.pipeline_fn(None, None, None, .45, pv, torch)
    names = set(full.__code__.co_names) | set(full.__code__.co_freevars)
    assert 'insert' not in names and 'patches' not in names and 'arm' not in names


def test_patched_input_differs_only_in_patch_region():
    base = torch.rand(1, 3, 320, 320)
    patch = torch.rand(3, 64, 64)
    img = e11.build_input('C1', base, patch, (100, 80), ap_.insert)
    diff = (img != base).any(1)[0]
    assert diff[100:164, 80:144].any() and not diff[:100].any() and not diff[164:].any()
    assert e11.build_input('clean_a', base, None, None, ap_.insert) is base
