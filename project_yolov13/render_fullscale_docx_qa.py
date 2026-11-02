"""Render the full-scale report with Word and retain every page for visual QA."""

from pathlib import Path

from pipeline_utils import configure_project_environment, project_path, require_local_venv


def main() -> None:
    require_local_venv()
    configure_project_environment()
    import fitz
    import win32com.client

    source = project_path("results/BaoCao_TrienKhaiRong_YOLOv13.docx", must_exist=True)
    output = project_path("results/fullscale_experiments/docx_qa")
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("page-*.png"):
        stale.unlink()
    pdf = output / "BaoCao_TrienKhaiRong_YOLOv13.pdf"
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    word.AutomationSecurity = 3
    document = None
    try:
        document = word.Documents.Open(str(source), ReadOnly=True, AddToRecentFiles=False)
        document.ExportAsFixedFormat(str(pdf), 17, OpenAfterExport=False)
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()
    with fitz.open(pdf) as rendered:
        for index, page in enumerate(rendered, 1):
            page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False).save(output / f"page-{index:02d}.png")
        print(f"Rendered {len(rendered)} pages to {output}")


if __name__ == "__main__":
    main()
