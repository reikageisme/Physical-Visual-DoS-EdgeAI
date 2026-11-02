"""Render the final time-budget report to local PNGs for visual inspection."""

from __future__ import annotations

import os
from pathlib import Path

import fitz
import win32com.client

from pipeline_utils import PROJECT_ROOT, configure_project_environment, require_local_venv


configure_project_environment()
require_local_venv()
source = PROJECT_ROOT / "results" / "BaoCao_ThucNghiệm_MocThoiGian.docx"
qa_dir = PROJECT_ROOT / "results" / "time_budget" / "docx_qa"
tmp_dir = PROJECT_ROOT / ".cache" / "docx_tmp"
qa_dir.mkdir(parents=True, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(tmp_dir)
os.environ["TMP"] = str(tmp_dir)
os.environ["TMPDIR"] = str(tmp_dir)
pdf = qa_dir / "BaoCao_ThucNghiệm_MocThoiGian.pdf"

word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
word.AutomationSecurity = 3
doc = None
try:
    doc = word.Documents.Open(str(source), ReadOnly=True, AddToRecentFiles=False)
    doc.ExportAsFixedFormat(str(pdf), 17, OpenAfterExport=False)
finally:
    if doc is not None:
        doc.Close(False)
    word.Quit()

with fitz.open(pdf) as rendered:
    for index, page in enumerate(rendered, start=1):
        image = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        image.save(qa_dir / f"page-{index:02d}.png")
    print(f"Rendered {len(rendered)} pages to {qa_dir.relative_to(PROJECT_ROOT)}")
