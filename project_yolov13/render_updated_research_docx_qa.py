"""Native Word fallback after diagnosing the packaged renderer."""
import subprocess
from pathlib import Path
from pipeline_utils import configure_project_environment,project_path,require_local_venv,write_json,sha256_file


def main():
    require_local_venv();configure_project_environment()
    import fitz
    import win32com.client
    source=project_path("results/BaoCao_NCKH_YOLOv13_Updated.docx",must_exist=True);output=project_path("results/updated_research/docx_qa");output.mkdir(parents=True,exist_ok=True)
    for stale in output.glob("page-*.png"):stale.unlink()
    pdf=output/"BaoCao_NCKH_YOLOv13_Updated.pdf"
    word=win32com.client.DispatchEx("Word.Application");word.Visible=False;word.DisplayAlerts=0;word.AutomationSecurity=3;document=None
    try:
        document=word.Documents.Open(str(source),ReadOnly=True,AddToRecentFiles=False);document.ExportAsFixedFormat(str(pdf),17,OpenAfterExport=False)
    finally:
        if document is not None:document.Close(False)
        word.Quit()
    poppler=Path(r"C:\Users\Quoc Bao\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe")
    if not poppler.is_file():raise FileNotFoundError("Bundled Poppler is required")
    subprocess.run([str(poppler),"-r","144","-png",str(pdf),str(output/"page")],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    with fitz.open(pdf) as pages:
        write_json(output/"render_manifest.json",{"renderer":"Microsoft Word COM PDF + bundled Poppler pdftoppm 144 DPI","renderer_limitation":"Packaged render_docx.py cannot run in project venv without pdf2image; no bundled LibreOffice is available on Windows. Used installed Microsoft Word for native export; did not use a user LibreOffice installation.","source":str(source),"source_sha256":sha256_file(source),"pages":len(pages),"visual_qa":"pending"})
        print(f"Rendered {len(pages)} pages to {output}")


if __name__=="__main__":main()
