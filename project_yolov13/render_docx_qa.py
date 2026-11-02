import os
from pathlib import Path

import fitz
import win32com.client

ROOT = Path(__file__).resolve().parent
TMP = ROOT / ".cache" / "docx_tmp"
TMP.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TMP)
os.environ["TMP"] = str(TMP)
os.environ["TMPDIR"] = str(TMP)

source = ROOT / "results" / "BaoCaoKhoaHoc_YOLOv13_AdvPatch.docx"
qa_dir = ROOT / "results" / "docx_qa"
qa_dir.mkdir(parents=True, exist_ok=True)
for stale in qa_dir.glob("page-*.png"):
    stale.unlink()
pdf = qa_dir / "BaoCaoKhoaHoc_YOLOv13_AdvPatch.pdf"

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
        pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        pix.save(qa_dir / f"page-{index:02d}.png")
    print(f"Rendered {len(rendered)} pages to {qa_dir}")
