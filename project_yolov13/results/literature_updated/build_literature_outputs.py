"""Curated primary-source benchmark ledger; run with the project .venv."""
from pathlib import Path
import json
import math
import hashlib

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATE = "2026-10-13"

refs = [
    dict(id="lei2025yolov13", title="YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception", authors=["Mengqi Lei", "Siqi Li", "Yihong Wu", "Han Hu", "You Zhou", "Xinhu Zheng", "Guiguang Ding", "Shaoyi Du", "Zongze Wu", "Yue Gao"], year=2025, arxiv="2506.17733v1", doi="10.48550/arXiv.2506.17733", url="https://arxiv.org/html/2506.17733v1", publication_type="arXiv preprint", anchors=["Sec. III-B; Eqs. (3)-(5)", "Sec. III-C; Eq. (13)", "Sec. III-D; Fig. 4", "Sec. IV-A; Table I", "Table V"], verification="Full text and official iMoonLab/yolov13 repository agree on own N/S benchmark rows."),
    dict(id="tian2025yolov12", title="YOLOv12: Attention-Centric Real-Time Object Detectors", authors=["Yunjie Tian", "Qixiang Ye", "David Doermann"], year=2025, arxiv="2502.12524v1", doi="10.48550/arXiv.2502.12524", url="https://arxiv.org/html/2502.12524v1", publication_type="arXiv v1; subsequent NeurIPS 2025 conference version differs", anchors=["Sec. 4.1", "Table 1", "Table 6", "Table 7"], verification="Pin v1 metrics; do not mix with current Turbo or conference tables."),
    dict(id="jocher2024yolo11", title="Ultralytics YOLO11", authors=["Glenn Jocher", "Jing Qiu"], year=2024, arxiv=None, doi=None, url="https://docs.ultralytics.com/models/yolo11/", publication_type="official software documentation; no formal official research paper", anchors=["Performance Metrics: Detection (COCO)", "Citations and Acknowledgments"], verification="Official page explicitly states no formal research paper for YOLO11; citation is software, not a peer-reviewed architecture article."),
    dict(id="wang2024yolov10", title="YOLOv10: Real-Time End-to-End Object Detection", authors=["Ao Wang", "Hui Chen", "Lihao Liu", "Kai Chen", "Zijia Lin", "Jungong Han", "Guiguang Ding"], year=2024, arxiv="2405.14458v1", doi="10.48550/arXiv.2405.14458", url="https://arxiv.org/html/2405.14458v1", publication_type="NeurIPS 2024; metrics pinned to arXiv v1 and official repository", anchors=["Sec. 4.1", "Table 1", "Appendix A.4 Table 15"], verification="Main latency column is end-to-end; separate Latency-f column is forward-only. Use main column."),
    dict(id="zhao2024rtdetr", title="DETRs Beat YOLOs on Real-time Object Detection", authors=["Yian Zhao", "Wenyu Lv", "Shangliang Xu", "Jinman Wei", "Guanzhong Wang", "Qingqing Dang", "Yi Liu", "Jie Chen"], year=2024, preprint_year=2023, arxiv="2304.08069", doi="10.48550/arXiv.2304.08069", url="https://arxiv.org/abs/2304.08069", publication_type="CVPR 2024; official repository metrics", anchors=["Official repository model table: COCO rows only", "arXiv abstract"], verification="Do not substitute COCO+Objects365 rows for COCO-only variants."),
    dict(id="lv2024rtdetrv2", title="RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer", authors=["Wenyu Lv", "Yian Zhao", "Qinyao Chang", "Kui Huang", "Guanzhong Wang", "Yi Liu"], year=2024, arxiv="2407.17140v1", doi="10.48550/arXiv.2407.17140", url="https://arxiv.org/html/2407.17140v1", publication_type="arXiv technical report", anchors=["Sec. 3.2", "Table 2: FPS-bs=1"], verification="v1 S has 47.9 AP; current official repository improved S has 48.1, stored separately in version notes."),
    dict(id="wang2025badpatch", title="BadPatch: Diffusion-Based Generation of Physical Adversarial Patches", authors=["Zhixiang Wang", "Xingjun Ma", "Yu-Gang Jiang"], year=2025, preprint_year=2024, arxiv="2412.01440v5", doi="10.48550/arXiv.2412.01440", url="https://arxiv.org/html/2412.01440v5", publication_type="arXiv v5; ICCV 2025 Findings workshop listing", anchors=["Sec. 4.1", "Sec. 4.5; Fig. 8"], verification="Latest arXiv title/three authors supersede old DiffPatch/seven-author citation in dataset README. CVF workshop listing was found, but direct full-text opening failed; use verified arXiv full text."),
    dict(id="xu2020adversarialtshirt", title="Adversarial T-shirt! Evading Person Detectors in A Physical World", authors=["Kaidi Xu", "Gaoyuan Zhang", "Sijia Liu", "Quanfu Fan", "Mengshu Sun", "Hongge Chen", "Pin-Yu Chen", "Yanzhi Wang", "Xue Lin"], year=2020, preprint_year=2019, arxiv="1910.11099", doi="10.48550/arXiv.1910.11099", url="https://arxiv.org/abs/1910.11099", publication_type="ECCV 2020", anchors=["Abstract", "Official ECCV paper: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500647.pdf"], verification="Physical/digital attack success is a distinct endpoint from detector COCO AP."),
]
ref_by_id = {r["id"]: r for r in refs}

def row(model, ref, ap, ap50, params, flops, latency, source, anchor, *, fps=None, batch=None, pipeline=None, precision="FP16", backend="TensorRT", version=None):
    r = ref_by_id[ref]
    return dict(model=model, label=model, reference_id=ref, title=r["title"], authors=r["authors"], year=r["year"], source_url=source, doi=r["doi"], arxiv=r["arxiv"], source_anchor=anchor, retrieved_date=DATE, model_version=version,
        dataset="MS COCO 2017", dataset_version="2017", train_split="train2017", evaluation_split="val2017", num_classes=80,
        evaluation_protocol="COCO bounding-box AP averaged across 80 classes; IoU 0.50:0.05:0.95", map50_95_pct=ap, map50_pct=ap50,
        parameters_m=params, flops_g=flops, flops_input_resolution=[640,640], published_latency_ms=latency,
        published_fps=fps, fps_derived_from_latency=(1000/latency if latency else None),
        latency_protocol=dict(device="NVIDIA Tesla T4", input_size=[640,640], batch=batch, precision=precision, backend=backend, pipeline=pipeline, warmup=None, repetitions=None, statistic=None),
        directly_comparable_to_local=False, comparable=False,
        noncomparability_reasons=["80-class COCO val2017 vs one-class re-split COCO subset + AdvT-shirt local test", f"640px/T4 {backend}, precision {precision if precision else 'unreported in inspected table'} vs 416px/RTX5060 PyTorch FP32 local pipeline", "Published point estimates vs local five-seed aggregates"],
        missing_protocol_fields_policy="null means not reported in inspected primary evidence; never inferred")

rows = [
    row("YOLO11n", "jocher2024yolo11",39.5,None,2.6,6.5,1.5,"https://docs.ultralytics.com/models/yolo11/","Performance Metrics / Detection (COCO)",backend="TensorRT10",precision=None,version="official documentation as retrieved 2026-10-13; table does not label precision"),
    row("YOLO11s", "jocher2024yolo11",47.0,None,9.4,21.6,2.5,"https://docs.ultralytics.com/models/yolo11/","Performance Metrics / Detection (COCO)",backend="TensorRT10",precision=None,version="official documentation as retrieved 2026-10-13; table does not label precision"),
    row("YOLOv12-N", "tian2025yolov12",40.6,56.7,2.6,6.5,1.64,"https://arxiv.org/html/2502.12524v1","Tables 1 and 6; Sec.4.1",version="v1.0 / arXiv v1, not Turbo"),
    row("YOLOv12-S", "tian2025yolov12",48.0,65.0,9.3,21.4,2.61,"https://arxiv.org/html/2502.12524v1","Tables 1 and 6; Sec.4.1",version="v1.0 / arXiv v1, not Turbo"),
    row("YOLOv13-N", "lei2025yolov13",41.6,57.8,2.5,6.4,1.97,"https://arxiv.org/html/2506.17733v1","Table I; Sec.IV-A",version="arXiv v1 official N"),
    row("YOLOv13-S", "lei2025yolov13",48.0,65.2,9.0,20.8,2.98,"https://arxiv.org/html/2506.17733v1","Table I; Sec.IV-A",version="arXiv v1 official S"),
    row("YOLOv10-N", "wang2024yolov10",38.5,53.8,2.3,6.7,1.84,"https://arxiv.org/html/2405.14458v1","Tables 1 and 15; Sec.4.1",pipeline="Main end-to-end latency column; distinct from forward-only Latency-f",version="NMS-free one-to-one inference"),
    row("YOLOv10-S", "wang2024yolov10",46.3,63.0,7.2,21.6,2.49,"https://arxiv.org/html/2405.14458v1","Tables 1 and 15; Sec.4.1",pipeline="Main end-to-end latency column; distinct from forward-only Latency-f",version="NMS-free one-to-one inference"),
    row("RT-DETR-R18", "zhao2024rtdetr",46.5,63.8,20,60,None,"https://github.com/lyuwenyu/RT-DETR","Model table COCO-only ResNet18 row; RT-DETRv2 v1 Table2 confirms batch1",fps=217,batch=1,pipeline="NMS-free; full preprocessing/I-O inclusion not established",version="COCO-only, not Objects365-pretrained row"),
    row("RT-DETR-R50", "zhao2024rtdetr",53.1,71.3,42,136,None,"https://github.com/lyuwenyu/RT-DETR","Model table COCO-only ResNet50 row; RT-DETRv2 v1 Table2 confirms batch1",fps=108,batch=1,pipeline="NMS-free; full preprocessing/I-O inclusion not established",version="COCO-only, not Objects365-pretrained row"),
    row("RT-DETRv2-S", "lv2024rtdetrv2",47.9,64.9,20,None,None,"https://arxiv.org/html/2407.17140v1","Table2; Sec.3.2",fps=217,batch=1,pipeline="NMS-free; full preprocessing/I-O inclusion not established",version="v1, grid_sample; not later repo 48.1 AP update"),
]
for r in rows:
    r["latency_protocol"]["missing_fields"] = [k for k,v in r["latency_protocol"].items() if v is None]
    r["latency_protocol"]["latency_unit"] = "ms/image" if r["published_latency_ms"] is not None else None
    r["latency_protocol"]["fps_unit"] = "frames/second" if r["published_fps"] is not None else None
    r["latency_protocol"]["full_local_pipeline_equivalence_verified"] = False

version_notes = [
    {"model":"YOLOv12", "source_url":"https://github.com/sunsmarterjie/yolov12", "anchor":"Main Results", "retrieved_date":DATE, "note":"Current Turbo N: AP40.4, params2.5M, FLOPs6.0G, latency1.60ms. Original v1 N40.6/2.6M/6.5G/1.64ms. Do not silently substitute updated release for locally used weights."},
    {"model":"YOLO11/YOLOv12", "source_url":"https://arxiv.org/html/2506.17733v1", "anchor":"Sec.IV-A; Table I", "retrieved_date":DATE, "note":"Authors reran official YOLO11 and YOLOv12 v1 on same training hardware; N AP38.6/40.1, S45.8/47.1. Hence reported YOLO13 gains refer to these reruns, not official docs or original paper figures."},
    {"model":"RT-DETRv2-S", "source_url":"https://github.com/lyuwenyu/RT-DETR", "anchor":"News 2024.08.22; model table", "retrieved_date":DATE, "note":"Current repo S AP48.1/AP50 65.1 vs arXivv1 47.9/64.9."},
]

data = json.loads((ROOT / "results/fullscale_setup/dataset_build_summary.json").read_text(encoding="utf-8"))
split = json.loads((ROOT / "results/fullscale_setup/split_integrity_audit.json").read_text(encoding="utf-8"))
local = dict(dataset="COCO 2017 deterministic 10,000-image subset + available AdvT-shirt mirror 1,123", total_images=data["total_images"], num_classes=1, class_names=["person"], split_ratio="70/15/15", split_seed=2026, split_counts=data["split_counts"], test_coco_train2017_images=split["source_split_counts"]["test"]["COCO 2017:train2017"], latency_protocol=dict(device="RTX 5060", input_size=416, batch=1, precision="FP32",backend="PyTorch with SDPA fallback",pipeline="preprocess + inference + postprocess; excludes image decode/I-O",warmup=32,repetitions=500,statistic="median per seed then five-seed mean/SD/95% Student-t CI"), caveats=["No exclusion of pretrained exposure to 1,429 local-test COCOtrain2017 images", "Image-level splits do not establish subject/sequence disjointness", "Pooled AP is not adversarial robustness or attack success rate", "A training-only optimized configuration is a project contribution; HyperACE, FullPAD and DS-C3k2 remain original authors' architecture"])

ablations = dict(source_url="https://arxiv.org/html/2506.17733v1", retrieved_date=DATE, dataset="COCO2017 val2017", num_classes=80, input_size=640, ownership="Original authors' published ablations; not local experiments", fullpad_hyperace=dict(anchor="Sec.IV-C1; Table III",model="YOLOv13-S",no_tunnel_ap95=47.1,no_tunnel_ap50=64.1,full_tunnels_ap95=48.0,full_tunnels_ap50=65.2,delta_ap95_pp=0.9,note="No tunnels equals removing effective HyperACE according to authors; not independent isolated HyperACE/FullPAD factorial ablation"),ds_blocks=dict(anchor="Table V",rows=[dict(model="YOLOv13-N",ds=True,ap95=41.6,params_m=2.5,flops_g=6.4),dict(model="YOLOv13-N",ds=False,ap95=41.6,params_m=3.1,flops_g=7.9),dict(model="YOLOv13-S",ds=True,ap95=48.0,params_m=9.0,flops_g=20.8),dict(model="YOLOv13-S",ds=False,ap95=48.0,params_m=11.7,flops_g=27.1)],note="Parameter/FLOPs benefit at equal rounded AP in this published ablation; does not guarantee faster local PyTorch latency"))
matrix = dict(schema_version="1.0",retrieved_date=DATE,study_scope="Targeted narrative primary-source benchmark review, not exhaustive systematic SOTA ranking",metric_units="AP in percent; Params millions; FLOPs billions; latency milliseconds; FPS frames/sec",rows=rows,references=refs,version_notes=version_notes,local_protocol=local,published_architecture_ablations=ablations)
(OUT / "benchmark_matrix.json").write_text(json.dumps(matrix,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
evidence = dict(retrieved_date=DATE,method="Primary papers, official software documentation and original authors' repositories; tables/sections cross-checked manually",direct_quotes=[],references=refs,version_discrepancies=version_notes,published_architecture_ablations=ablations,local_evidence_files=["results/fullscale_setup/dataset_build_summary.json","results/fullscale_setup/split_integrity_audit.json","results/fullscale_experiments/latency_records.json"],unreported_values_policy="Use null; do not guess missing batch, timing boundary, dtype or repetitions.",retrieval_failures=["Direct opening of CVF BadPatch workshop PDF/HTML returned Internal Error; verified arXiv v5 full text used instead."])
(OUT / "source_evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

table = ["| Nguồn / biến thể | Năm | Dữ liệu / lớp / split | AP50 (%) | AP50:95 (%) | Params (M) | FLOPs (G), 640px | Latency/FPS công bố |", "|---|---:|---|---:|---:|---:|---:|---|"]
for r in rows:
    val = lambda x: "—" if x is None else f"{x:g}"
    timing = f"{r['published_latency_ms']:g} ms" if r['published_latency_ms'] else f"{r['published_fps']:g} FPS"
    table.append(f"| [{r['label']}]({r['source_url']}) | {r['year']} | COCO2017 / 80 / val2017 | {val(r['map50_pct'])} | {r['map50_95_pct']:g} | {r['parameters_m']:g} | {val(r['flops_g'])} | {timing} |")

synthesis = """# Tổng hợp nguồn và đối sánh giao thức

Ngày truy xuất: 13/10/2026. Đây là rà soát tường thuật có chủ đích các kiến trúc liên quan; không phải tìm kiếm hệ thống đầy đủ hay bảng xếp hạng toàn bộ SOTA. Số liệu trong bảng là kết quả công bố, không phải kết quả thực nghiệm của dự án.

## Ma trận bài báo và phần mềm

""" + "\n".join(table) + """

AP là COCO bounding-box AP; dấu — nghĩa là chưa xác định trong nguồn đã kiểm tra. YOLO11 được dẫn dưới dạng phần mềm chính thức vì nhà phát triển nói rõ chưa có bài báo nghiên cứu chính thức. Các hàng YOLOv12 cố định arXiv v1; YOLOv13 cố định arXiv v1. Hàng RT-DETR chỉ dùng COCO, không dùng kết quả có pretraining Objects365. GPU các hàng tốc độ là T4; YOLOv10/12/13 và RT-DETR dùng TensorRT FP16, còn bảng tài liệu YOLO11 ghi TensorRT10 và chưa xác nhận dtype trong trang đó. Batch, phạm vi preprocessing và số lần đo còn thiếu được giữ `null` trong JSON.

## Khác biệt phiên bản phải giải thích

- [YOLOv12 repo hiện tại](https://github.com/sunsmarterjie/yolov12) có Turbo; không đồng nhất với bản v1 đã dùng trong bài gốc và dự án.
- [YOLOv13, Table I và Sec. IV-A](https://arxiv.org/html/2506.17733v1) tái lập YOLO11/12 trên phần cứng huấn luyện của tác giả. Các mức chênh lệch trong paper đó phải tính theo baseline được tái lập, không ghép baseline từ tài liệu khác.
- [RT-DETRv2 repo](https://github.com/lyuwenyu/RT-DETR) có cập nhật S lên 48.1 AP; bảng này cố định technical report v1 (47.9).
- [BadPatch arXiv v5](https://arxiv.org/abs/2412.01440v5) có tiêu đề và ba tác giả hiện tại; citation DiffPatch cũ trong README bộ dữ liệu không nên chép nguyên làm nguồn mới.

## Bối cảnh vật lý và giới hạn về nhiệm vụ

[BadPatch, Sec. 4.1 và 4.5](https://arxiv.org/html/2412.01440v5) tạo patch để làm giảm khả năng phát hiện người, đánh giá ASR/AP trên các detector đã pretrain và xây dựng AdvT-shirt-1K 1,131 ảnh. AP/ASR tấn công này không cùng mục tiêu với việc tăng AP của mô hình fine-tune. [Adversarial T-shirt, ECCV2020](https://arxiv.org/abs/1910.11099) đặt nền tảng cho thử nghiệm patch trên áo có biến dạng khi người di chuyển. Hai nguồn này giúp mô tả threat model; không xác nhận rằng kiến trúc YOLOv13 đã chống được mọi patch hoặc gây/khắc phục DoS tính toán.

## Đối chiếu với thực nghiệm của dự án

Dự án dùng 11,123 ảnh (COCO subset 10,000 + mirror AdvT-shirt 1,123), chỉ một lớp person, train/val/test 7,786/1,669/1,668. Chỉ dùng val chọn cấu hình; test để đánh giá cấu hình đã khóa. Các paper detector đánh giá 80 lớp trên val2017 và thường huấn luyện đầy đủ hơn; không thể suy ra 47–48% của dự án bằng hay kém 47–48% trên COCO80 lớp. Thước đo dùng cùng dải IoU vẫn có phân phối dữ liệu và phép gộp lớp khác nhau.

Latency dự án ở RTX5060, PyTorch FP32, 416px, batch1, gồm preprocess/inference/postprocess. Latency công bố ở T4 TensorRT và 640px không dùng để tính hệ số tăng tốc của dự án. FPS suy từ 1,000/latency chỉ là phép quy đổi, không phải phép đo throughput độc lập. Đồ thị cần tách panel công bố và panel thực nghiệm; không chuẩn hóa radar rồi xếp hạng giữa các panel.

Có 1,429 ảnh test dự án xuất phát từ COCOtrain2017 và có thể đã xuất hiện khi pretrain; đây là holdout của fine-tuning, chưa là holdout của toàn bộ vòng đời mô hình. Split theo ảnh chưa loại trừ tương quan người/chuỗi. Pooled AP chủ yếu chịu ảnh COCO vì test Adv chỉ 168 ảnh; cần phân tầng clean/Adv và thử nghiệm patch chưa thấy nếu muốn kết luận về độ bền vật lý.

## Giải trình về ưu thế và tính độc đáo

[YOLOv13, Sec. III-B–D](https://arxiv.org/html/2506.17733v1) dùng hyperedge thích nghi để tổng hợp nhiều vị trí/thang đo; FullPAD phân phối đặc trưng; DS-C3k2 là biến thể C3k2 dùng depthwise-separable convolution. Đây là kiến trúc của Lei et al. Table III: đủ tunnel so với bỏ tunnel tăng AP95 47.1 lên 48.0. Table V: S dùng DS giảm 11.7 xuống 9.0M parameters và 27.1 xuống 20.8G FLOPs, cùng AP95 làm tròn 48.0. Đây là ablation của tác giả gốc; dự án chưa tự cô lập các thành phần. Tiềm năng ngữ cảnh không đồng nghĩa bảo đảm chống patch. Params/FLOPs thấp hơn không bảo đảm latency thấp hơn trên backend đang dùng.

Đóng góp có thể bảo vệ của dự án là quy trình áp dụng và tối ưu cho phát hiện người: dữ liệu có truy vết, lựa chọn trên validation, kiểm tra scheduler/loss/augmentation/sampling, đánh giá phân tầng và chi phí thực tế. Chỉ gọi là cải tiến thực nghiệm khi Agent tối ưu có số liệu kiểm chứng; chỉ quy nguyên nhân cho một thành phần khi có ablation giữ các yếu tố khác cố định. So sánh YOLOv13s với YOLO11n đồng thời thay dung lượng mô hình, nên không cô lập lợi ích của HyperACE. Kết quả một seed tối ưu sau tìm kiếm chưa thay thế bằng chứng năm seed ban đầu.

Một kết quả âm hoặc tốc độ thấp vẫn có giá trị nghiên cứu: nó xác định giới hạn triển khai trong môi trường Windows/SDPA và giúp chọn cấu hình theo yêu cầu ứng dụng. Lập luận thuyết phục cần nói rõ phạm vi đạt được, thay vì tuyên bố vượt trội phổ quát.
"""
(OUT / "synthesis.md").write_text(synthesis,encoding="utf-8")

pros = """# Ma trận ưu/nhược điểm và mức chứng cứ

| Thành phần / hướng đi | Ưu điểm có cơ sở | Hạn chế / đánh đổi | Mức chứng cứ và kiểm chứng cần có |
|---|---|---|---|
| HyperACE | Học nhóm tương quan nhiều vị trí và thang đo [1] | Lợi ích ngữ cảnh chưa chứng minh miễn nhiễm patch; chi phí phụ thuộc số hyperedge | Cơ chế và ablation của tác giả gốc; dự án cần ablation riêng |
| FullPAD | Đưa đặc trưng đã tổng hợp tới nhiều tầng [1] | Thêm nhánh/tunnel và phụ thuộc backend | Kiến trúc gốc; không coi là phát minh của dự án |
| DS-C3k2 | Depthwise-separable convolution giảm số phép tính/tham số trong ablation gốc [1] | FLOPs không đo băng thông bộ nhớ hay hiệu quả kernel; ít FLOPs vẫn có thể chậm | Lý thuyết và Table V gốc; cần đo pipeline trên thiết bị đích |
| YOLOv12 | Area Attention/R-ELAN hỗ trợ biểu diễn ngữ cảnh [2] | Backend attention và phiên bản Turbo/v1 ảnh hưởng tái lập | Bài gốc và repo; so sánh phải khóa phiên bản |
| YOLOv10 | Dual assignment hỗ trợ suy luận không NMS [3] | Loại NMS không tự chứng minh AP cao nhất cho person/patch | Bài gốc; cần cùng dataset và giao thức nếu bổ sung baseline |
| RT-DETR | Encoder đa thang và lựa chọn query; điều chỉnh số lớp decoder [4] | Dung lượng lớn hơn các nano được chạy; không cùng thiết lập đã đo | Nguồn gốc; chưa có benchmark địa phương |
| Tối ưu scheduler/loss/augmentation | Có thể thích ứng bài toán person và giảm bất ổn fine-tuning | Tìm nhiều cấu hình làm tăng nguy cơ overfit validation; augmentation quá mạnh có thể phá dữ liệu vật lý | Phải lưu mọi trial, chỉ dùng val chọn, khóa trước test |
| Sampling theo nguồn train | Có thể tăng tần suất nhìn thấy nguồn Adv ít ảnh | Oversampling có thể lặp cảnh/người và tăng lệch phân phối | Giả thuyết của dự án; cần ablation và chỉ resample train |
| Thực nghiệm nhiều seed | Cho biết biến thiên do tối ưu ngẫu nhiên | CI theo seed không đo bất định do split/đối tượng/dataset | Kết quả địa phương; dùng Student-t CI và paired difference khi hợp lý |
| Dữ liệu gộp COCO + Adv | Có cả cảnh phổ thông và áo patch vật lý [5] | Pretrain exposure và split ảnh; pooled AP chưa đo causal defense | Cần báo cáo riêng nguồn, subject/sequence split và thử nghiệm ngoài phân phối |

Nguồn:

[1] [Lei et al., YOLOv13, Sec. III và Table V](https://arxiv.org/html/2506.17733v1).

[2] [Tian et al., YOLOv12 v1](https://arxiv.org/html/2502.12524v1).

[3] [Wang et al., YOLOv10](https://arxiv.org/html/2405.14458v1).

[4] [Zhao et al., RT-DETR](https://arxiv.org/abs/2304.08069).

[5] [Wang et al., BadPatch v5, Sec. 4.5](https://arxiv.org/html/2412.01440v5).

Phần suy luận về sampling, chọn cấu hình và giới hạn CI là phân tích phương pháp của dự án, không phải kết quả đo đã có. Không chuyển giá trị công bố thành giá trị thực nghiệm địa phương.
"""
(OUT / "pros_cons.md").write_text(pros,encoding="utf-8")

bib = []
for i,r in enumerate(refs,1):
    kind = "software" if r["id"] == "jocher2024yolo11" else "misc"
    fields = [f"  author = {{{' and '.join(r['authors'])}}}",f"  title = {{{r['title']}}}",f"  year = {{{r['year']}}}",f"  url = {{{r['url']}}}",f"  note = {{Accessed {DATE}; {r['publication_type']}}}"]
    if r['doi']: fields.append(f"  doi = {{{r['doi']}}}")
    if r['arxiv']: fields.append(f"  eprint = {{{r['arxiv']}}}")
    bib.append("@"+kind+"{"+r['id']+",\n"+",\n".join(fields)+"\n}")
(OUT / "references.bib").write_text("\n\n".join(bib)+"\n",encoding="utf-8")

failures = []
numeric_checks = []
for r in rows:
    for name in ["map50_95_pct", "map50_pct"]:
        x = r[name]
        passed = x is None or (isinstance(x, (int,float)) and math.isfinite(x) and 0 <= x <= 100)
        numeric_checks.append(dict(model=r["model"],field=name,passed=passed,null_allowed=name=="map50_pct"))
        if not passed: failures.append(f"{r['model']}: invalid {name}")
    for name in ["parameters_m","flops_g","published_latency_ms","published_fps"]:
        x = r[name]
        passed = (x is None and name != "parameters_m") or (isinstance(x,(int,float)) and math.isfinite(x) and x>0)
        numeric_checks.append(dict(model=r["model"],field=name,passed=passed,null_allowed=name!="parameters_m"))
        if not passed: failures.append(f"{r['model']}: invalid {name}")
    if not r["source_url"].startswith("https://") or not r["source_anchor"]: failures.append(f"{r['model']}: source locator absent")
    if r["latency_protocol"]["precision"] is None and "FP16" in r["noncomparability_reasons"][1]: failures.append(f"{r['model']}: inferred dtype despite null")
    if r["comparable"] or r["directly_comparable_to_local"]: failures.append(f"{r['model']}: improper local comparability")
audit = dict(date=DATE,scope="Self-audit of curated literature ledger structure, numeric units and claim scope; not a formal bibliographic certificate, independent replication or exhaustive systematic review",status="pass" if not failures else "fail",failures=failures,row_count=len(rows),reference_count=len(refs),numeric_checks=numeric_checks,
    missing_timing_protocol_by_model={r["model"]:r["latency_protocol"]["missing_fields"] for r in rows},
    claim_scope_checks=[
        dict(claim="YOLO11 published latency dtype",status="explicitly unreported",evidence="Official Detection COCO table labels T4 TensorRT10 but does not label dtype; two rows retain precision null and dynamic noncomparability text"),
        dict(claim="HyperACE/FullPAD S ablation 47.1 to 48.0 AP95",status="source matched",source_url=ablations["source_url"],anchor="Sec.IV-C1 / Table III",scope="Same published S architecture with tunnel placement changed; no-tunnel disables effective HyperACE according to authors. Supports combined module/tunnel benefit, not separate causal effect or local defense benefit."),
        dict(claim="DS blocks S 11.7 to 9.0M and27.1 to20.8G at48.0AP95",status="source matched",source_url=ablations["source_url"],anchor="Table V",scope="Published same S family, DS presence varied. Rounded AP is equal. Structural Params/FLOPs at640px; no latency listed in TableV, so no hardware speedup claimed."),
        dict(claim="Architecture ablation hardware and training setting",status="context recorded, raw replication unavailable",source_url=ablations["source_url"],anchor="Sec.IV-A2",scope="Paper general setting600epochs, batch256, SGD0.01,640px; S training8RTX4090, general latency T4TensorRTFP16. TableIII/V do not independently restate every control or provide run-level variance. Source-level match is not independent verification of their raw experiment."),
        dict(claim="Ownership and local transfer",status="bounded",scope="Architectural invention credited to Lei etal; applicability to adversarial person data remains an empirical hypothesis. Local contribution is applied optimization/protocol, not invented HyperACE/DS-C3k2."),
        dict(claim="Cross-paper and local ranking",status="disallowed",scope="All eleven published rows comparable=false to local one-class test. Version notes retain contrary figures; published FPS inverses marked derived separately."),
        dict(claim="Dataset and threat endpoints",status="bounded",scope="Official AdvT-shirt1131 vs available mirror1123 preserved; clean pooled AP does not establish ASR resistance, unseen-patch defense or computational DoS."),
    ],coverage=dict(detector_architectures=["YOLO11 software","YOLOv12v1","YOLOv13v1","YOLOv10","RT-DETR","RT-DETRv2v1"],threat_context=["BadPatch arXivv5","Adversarial T-shirt ECCV2020"],excluded_scope=["Exhaustive2026SOTA search","Formal citation authentication certificate","Independent replication of published figures"]),
    artifacts_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ["benchmark_matrix.json","source_evidence.json","synthesis.md","pros_cons.md","references.bib"]})
(OUT / "literature_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if failures: raise RuntimeError("; ".join(failures))
print(json.dumps({"rows":len(rows),"references":len(refs),"outputs":[p.name for p in OUT.iterdir() if p.is_file()]},ensure_ascii=False))
