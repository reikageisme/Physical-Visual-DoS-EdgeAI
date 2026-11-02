"""stats.py + cohorts.py tái tính số đã xác minh (CLAUDE.md; REVIEW/03 §2 K02 và §7) từ raw, chỉ đọc."""
import numpy as np
import pytest

from sponge_sim import cohorts, stats


def _primary(cohort):
    x = []
    for s in range(10):
        clean = cohorts.read_case(cohort, 'clean', s, 0)
        sg = cohorts.read_case(cohort, 'sg_ga', s, 2500)
        center = cohorts.read_case(cohort, 'ga_center', s, 2500)
        x.append(cohorts.mean_image_ratio(sg, clean) - cohorts.mean_image_ratio(center, clean))
    return np.array(x)


@pytest.mark.parametrize('cohort,mean,ci,p,dz', [
    ('OLD128', -.005132356025296, [-.011178378580684, .000706024479203], .150390625, -.498396136142396),
    ('X1-500', -.029335098118660065, [-.03832823790339332, -.021598326633977788], .001953125, -2.0545644255415554),
])
def test_primary_sg_minus_center(cohort, mean, ci, p, dz):
    s = stats.paired_summary(_primary(cohort), seed=20261002)
    assert s['mean'] == pytest.approx(mean, abs=1e-12)
    assert s['ci95'] == pytest.approx(ci, abs=1e-12)
    assert s['p_sign_flip'] == p
    assert s['dz'] == pytest.approx(dz, abs=1e-9)


def test_holm_simple():
    assert stats.holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])
