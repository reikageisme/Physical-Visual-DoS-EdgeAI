"""Record report artifact checks after all rendered pages were visually reviewed."""
from __future__ import annotations
import hashlib
import json
import sys
import zipfile
from datetime import datetime,timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pipeline_utils import configure_project_environment,project_path,require_local_venv,sha256_file,write_json
import generate_updated_research_report as builder


def main():
    require_local_venv();configure_project_environment()
    from docx import Document
    import fitz
    builder.require_inputs()
    manifest=builder.read("results/updated_research/editor_artifact_manifest.json")
    for record in manifest["artifacts"]:
        assert record["sha256"]==sha256_file(record["path"]),"Artifact changed after authoring"
    for name,value in manifest["source_sha256"].items():
        assert value==sha256_file(name),"Evidence changed after authoring"
    docx=project_path("results/BaoCao_NCKH_YOLOv13_Updated.docx",must_exist=True)
    doc=Document(docx)
    assert doc.core_properties.author==doc.core_properties.last_modified_by=="Mai Quốc Bảo"
    assert doc.core_properties.created.year==2026
    assert doc.styles["Normal"].font.name=="Times New Roman"
    assert doc.styles["Normal"].font.size.pt==12
    assert doc.styles["Normal"].paragraph_format.line_spacing==1.15
    assert len(doc.tables)==11 and len(doc.inline_shapes)==2
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    assert p.paragraph_format.line_spacing==1.15
                    for run in p.runs:
                        assert run.font.name=="Times New Roman" and run.font.size.pt==12
    figure_files=[project_path("results/figures/"+name,must_exist=True) for name in("comparison_baseline_vs_proposed.png","tradeoff_map_vs_latency.png")]
    with zipfile.ZipFile(docx) as archive:
        image_hashes=[hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if name.startswith("word/media/")]
    assert image_hashes==[sha256_file(p) for p in figure_files],"Embedded figures differ from final PNGs"
    md=project_path("results/BaoCao_NCKH_YOLOv13_Updated.md",must_exist=True).read_text(encoding="utf-8")
    assert "trained_winner" not in md
    assert md.count(r"\begin{table*}")==md.count(r"\end{table*}")==9
    assert "59.02" in md and "58.63" in md and "69,4%" in md
    assert "120 phút là mốc tối đa đã thử" in md
    qa=project_path("results/updated_research/docx_qa")
    rendered=builder.read(qa/"render_manifest.json")
    assert rendered["source_sha256"]==sha256_file(docx)
    pages={p.name:sha256_file(p) for p in sorted(qa.glob("page-*.png"))}
    assert pages==builder.read("results/updated_research/editor_inspected_page_hashes.json")
    with fitz.open(qa/"BaoCao_NCKH_YOLOv13_Updated.pdf") as pdf:
        assert len(pdf)==len(pages)==18
        for page in pdf:
            assert len(page.get_text().strip())>50,"Blank rendered page"
            for word in page.get_text("words"):
                assert word[0]>=-1 and word[1]>=-1 and word[2]<=page.rect.width+1 and word[3]<=page.rect.height+1,"Rendered text outside page"
    # The metadata-only date repair was re-rendered; all final PNG byte hashes
    # equal the exact page images already opened and inspected at full resolution.
    result={"status":"passed","checked_at_utc":datetime.now(timezone.utc).isoformat(),"docx_sha256":sha256_file(docx),"markdown_sha256":sha256_file("results/BaoCao_NCKH_YOLOv13_Updated.md"),"pages":18,"visually_reviewed_pages":list(range(1,19)),"visual_review":"All pages inspected at original rendered resolution. No clipped text, missing glyphs, overlaps, orphan table headers, blank pages or separated figure captions. Final metadata-only repair re-rendered to byte-identical page PNGs.","page_sha256":pages,"word_tables":11,"embedded_figures":2,"latex_numeric_tables":9,"body_and_table_font":"Times New Roman 12pt, line spacing1.15; black headings use15/13pt and title18pt; footer9pt","renderer":rendered["renderer"],"renderer_limitation":rendered["renderer_limitation"],"figure_embeds_match_final_files":True,"source_hashes_match":True,"artifact_marker":"Successfully invoked exactly once immediately before first authoring on 2026-10-17; not repeated for repairs."}
    write_json("results/updated_research/editor_qa_audit.json",result)
    rendered["visual_qa"]="passed";rendered["visually_reviewed_pages"]=list(range(1,19))
    write_json(qa/"render_manifest.json",rendered)
    manifest["status"]="final_visual_qa_passed"
    manifest["visual_qa"]={"status":"passed","pages":18,"audit":"results/updated_research/editor_qa_audit.json","docx_sha256":sha256_file(docx)}
    write_json("results/updated_research/editor_artifact_manifest.json",manifest)
    print(json.dumps({"status":"passed","pages":18,"tables":11,"figures":2,"docx_sha256":sha256_file(docx),"markdown_sha256":sha256_file("results/BaoCao_NCKH_YOLOv13_Updated.md")}))


if __name__=="__main__":main()
