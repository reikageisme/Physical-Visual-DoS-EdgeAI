"""Review bundle tests, no device or Torch load."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies/arm_preflight_v2'))
from build_draft_bundle import build
from pin_worker import validate
from qa_session import validate_cells


def test_bundle_is_hash_complete_and_every_unresolved_cell_refuses_load(tmp_path):
    root=tmp_path/'draft';manifest=build(root)
    assert manifest['status']=='DRAFT_UNRUN_NOT_AUTHORIZED_FOR_LOAD' and manifest['prereg_tag'] is None
    for path,expected in manifest['files_sha256'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected
    cells=[json.loads((root/path).read_text()) for path in manifest['cells']]
    assert len(cells)==4 and len({(c['pool_order'],c['pin_scope']) for c in cells})==4
    for cell in cells:
        with pytest.raises(ValueError):validate(cell)
    with pytest.raises(ValueError):validate_cells(cells)
    assert 'source/qa_session.py' in manifest['files_sha256']
    before=(root/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):build(root)
    assert before==(root/'manifest.json').read_bytes()
