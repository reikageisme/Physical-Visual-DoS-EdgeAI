"""Saved-stage diagnostic coverage and timer/counter scope regressions."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('stage_drift', Path(__file__).parents[1]/'studies/laptop_seg/analyze_stage_drift.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def timing(cpu=0, value=248, status=0):
    return dict(kind='timing', arm='clean', stage='aa_A', ns=6400000, process_cpu_ns=cpu,
                t_start_ns=2000000, pdh_after=dict(interval_start_ns=0, collected_ns=10000000,
                read_duration_ns=100000, counters={'% Processor Performance':
                    dict(return_code=0, status=status, value=value)}))


def test_component_counter_is_visible_and_uses_whole_block_units():
    result = module.summarize_rows([timing()], 'nms_kernel', 64)['arms']['clean']
    assert result['pdh_counters']['% Processor Performance']['values']['median'] == 248
    assert result['raw_block_wall_ms']['median'] == 6.4
    assert result['pooled_block_average_per_call_ms']['median'] == .1
    assert result['pdh_read_over_raw_block_wall']['median'] == .015625
    assert result['pdh_intervals'][0]['timer_fraction'] == .64


def test_quantization_and_invalid_pdh_are_retained_without_causal_gate():
    result = module.summarize_rows([timing(), timing(15625000, 999, 5)], 'mask_native', 1)
    arm = result['arms']['clean']
    assert arm['zero_process_cpu_count'] == 1
    assert arm['cpu_over_wall_above_one_count'] == 1
    assert arm['pdh_counters']['% Processor Performance']['values']['median'] == 248
    assert arm['pdh_counters']['% Processor Performance']['missing_or_invalid'] == 1
    assert 'quality_pass' not in result


def test_unknown_normalization_and_empty_records_refused():
    with pytest.raises(ValueError):module.summarize_rows([timing()], 'nms_kernel', 128)
    with pytest.raises(ValueError):module.summarize_rows([], 'nms_kernel', 64)
