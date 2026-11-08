"""Synthetic analysis checks independent of the GPU attack implementation."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('large_analysis', Path(__file__).parents[1] / 'studies/large_seed/analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_constant_endpoint_and_zero_wilson():
    clean = np.full((8, 5, 1), 3)
    attacked = np.full((8, 5, 1), 103)
    ci = analysis.hierarchical(clean, attacked, 200)[0]
    assert ci['mean_paired_gain_ci95'] == [100., 100.]
    assert ci['attacked_median_image_mean_ci95'] == [103., 103.]
    assert ci['mean_start_reach50_ci95'] == [1., 1.]
    zero = analysis.wilson(np.zeros(500))
    assert zero['hits'] == 0 and 0 < zero['wilson_ci95'][1] < .008


def test_starts_do_not_inflate_wilson_sample_size():
    clean = np.zeros((10, 5), dtype=int)
    attacked = clean.copy(); attacked[:, 2] = 100
    result = analysis.descriptive(clean, attacked)
    assert result['mean_start_reach50'] == .2
    assert result['any_start']['50']['n_images'] == 10
    assert result['any_start']['50']['hits'] == 10
    assert result['per_start']['0']['reach']['50']['hits'] == 0
    assert result['per_start']['23456']['reach']['50']['hits'] == 10


if __name__ == '__main__':
    test_constant_endpoint_and_zero_wilson()
    test_starts_do_not_inflate_wilson_sample_size()
    print('large seed analysis: 2 checks passed')
