"""Regressions for lost negative evidence and stale numeric provenance.

These integration checks read saved small summaries and render only to a
pytest temporary directory. No inference or original artifact is rewritten.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

PAPER = Path(__file__).parents[2]/'paper_en'
SECTIONS = ('nested1500', 'laptop_cpu', 'laptop_defense', 'laptop_components', 'laptop_reference_trace', 'laptop_pcores')


def extension():
    spec = importlib.util.spec_from_file_location('extension_builder', PAPER/'segmentation_extension/make_extension.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_extension_regeneration_keeps_all_negative_controls(tmp_path, monkeypatch):
    module = extension()
    original = (module.OUT/'extension.tex').read_bytes()
    for name in SECTIONS:
        shutil.copyfile(module.OUT/f'{name}.tex', tmp_path/f'{name}.tex')
    monkeypatch.setattr(module, 'OUT', tmp_path)
    module.main()
    assert (tmp_path/'extension.tex').read_bytes() == original
    sources = json.loads((tmp_path/'sources.json').read_text(encoding='utf-8'))
    for name in SECTIONS:
        assert f'paper_en/segmentation_extension/{name}.tex' in sources
    assert (PAPER/'segmentation_extension/extension.tex').read_bytes() == original


def test_missing_failure_section_stops_before_writing_extension(tmp_path, monkeypatch):
    module = extension()
    for name in SECTIONS:
        if name != 'laptop_components':
            shutil.copyfile(module.OUT/f'{name}.tex', tmp_path/f'{name}.tex')
    monkeypatch.setattr(module, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):module.main()
    assert not (tmp_path/'extension.tex').exists()
    assert not (tmp_path/'sources.json').exists()


def test_numeric_registry_corruption_fails_even_when_table_text_matches(tmp_path):
    sys.path.insert(0, str(PAPER/'tables_q3'))
    import check_numbers_en as checker
    for path in (PAPER/'tables_q3').glob('tab_*.tex'):
        shutil.copyfile(path, tmp_path/path.name)
    registry = json.loads((PAPER/'tables_q3/numbers_en.json').read_text(encoding='utf-8'))
    (tmp_path/'numbers_en.json').write_text(json.dumps(registry), encoding='utf-8')
    assert checker.check(tmp_path)['provenance_registry_match']
    registry['tab_aa'][0]['source'] = 'wrong-source.json:unrelated'
    (tmp_path/'numbers_en.json').write_text(json.dumps(registry), encoding='utf-8')
    result = checker.check(tmp_path)
    assert result['tables_match']
    assert not result['provenance_registry_match']
    assert not result['narrative_claims_verified']
