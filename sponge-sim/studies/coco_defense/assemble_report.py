"""Build a Vietnamese session report directly from saved evidence, no measurements."""
import hashlib
import json
from pathlib import Path
import numpy as np


if __name__=='__main__':
    repo=Path(__file__).resolve().parents[3]
    def read(p):return json.loads((repo/p).read_text(encoding='utf-8'))
    prefix='sponge-sim/results/confirmation_20261106/'
    cpu=read(prefix+'m2_cpu4_full_20261106/summary_verified.json')
    gpu=read(prefix+'m2_gpu_full_20261106/summary_verified.json')
    cpu1=read(prefix+'m2_cpu1_full_20261106/summary_verified.json')
    m3=read(prefix+'m3_native_full_20261106/summary.json')
    m4=read('sponge-sim/results/m4_nms_dispatch/m4_full_20261106/summary.json')
    m4b=read('sponge-sim/results/m4_nms_dispatch/m4b_full_20261106/summary.json')
    coco=[json.loads(s) for s in (repo/'sponge-sim/results/coco_defense_20261106/full_remote/results.jsonl').read_text().splitlines()]
    replay=read('sponge-sim/results/coco_defense_20261106/full_remote/metric_replay_receipt.json')
    cap=[json.loads(s) for s in (repo/'_provenance/codex_gpu_20261106/cap_full_20261106/raw.jsonl').read_text().splitlines()]
    assert len(coco)==8 and all(r['seen']==5000 for r in coco)
    assert len(replay)==8 and max(r['max_abs_error'] for r in replay)<1e-12
    assert cpu['quality_pass'] and gpu['quality_pass'] and not cpu1['quality_pass']
    lines=['# Kết quả phiên Codex — hướng phân đoạn, 06/11/2026',
      '## Đã xong',
      '- Toàn bộ phép đo mới có PREREG và tag trước chạy; giữ run thất bại, smoke và lịch sử. Nhánh `harness/step1`.',
      '- Chốt cơ chế chọn NMS: Ultralytics 8.4.203 kiểm `torchvision` trong `sys.modules`; có import dùng torchvision.ops.nms, thiếu import dùng TorchNMS.nms. Các script cũ khác trạng thái import. M4 cùng tensor/tiền xử lý cho đầu ra khớp 40/40 records; không gọi chênh lệch giữa script là ngẫu nhiên trên cùng kernel.',
      '- Đã chạy M2 CPU1, CPU4 và GPU, M3 nguồn + chuyển giao native YOLO26, M4 full và M4B fixed-tv full, COCO 8 tổ hợp × 5.000 ảnh, kiểm workload cap20 ảnh.',
      '', '| Thiết bị/chế độ mask | Tỷ số total K300/K10 [CI95%] | MDE (%) | K* | Cờ drift |',
      '|---|---|---:|---:|---:|']
    for label,s in [('CPU4',cpu),('GPU',gpu)]:
        for retina in ['False','True']:
            r=s['ratios'][retina+'|300'];lines.append(f"| {label}, {'full-res' if retina=='True' else 'proto-res'} | {r['ratio']:.4f} [{r['ci95'][0]:.4f}; {r['ci95'][1]:.4f}] | {s['mde'][retina]['frac']*100:.3f} | {s['k_star'][retina]} | {len(s['flagged'])} |")
    lines += ['', 'Nguồn bảng: `sponge-sim/results/confirmation_20261106/m2_{cpu4,gpu}_full_20261106/summary_verified.json:ratios,mde,k_star`; raw cùng thư mục, selector `kind=cell,image,k,retina,rep,copy`. Bootstrap ảnh20.000, seed20261106; MDE xấp xỉ t. Cả hai kiểm tải CPU4 đạt ~3,97 CPU hiệu dụng. Topology thuê chỉ lộ18 logical processors đồng nhất; không nhận là P/E hay lõi vật lý.',
      '', '| Model | K sạch → K final (trung vị) | Số ảnh K≥50 | K≥100 | K=300 |', '|---|---|---:|---:|---:|']
    for name,key in [('YOLOv8n-seg','k_v8'),('YOLO26n-seg native','k26_native')]:
        s=m3['models'][key];lines.append(f"| {name} | {s['clean_median']:g} → {s['attacked_median']:g} | {s['reach50']}/20 | {s['reach100']}/20 | {s['reach300']}/20 |")
    lines += ['', 'Nguồn: `.../m3_native_full_20261106/{raw.jsonl,summary.json}`. Nhiễu tối ưu YOLOv8n-seg, CÙNG nhiễu chuyển YOLO26, `end2end=True` bật rõ; đếm score>0,25 trực tiếp trong `[1,300,38]`, không NMS trên YOLO26. Endpoint là final iterate, không chọn best iterate. Kết quả không hỗ trợ chuyển giao tới K cao trên native one-to-one; chưa kiểm tấn công riêng tối ưu trên YOLO26.',
      '', '| Model | Phòng thủ | Seg mAP50–95 (%) | Δ so baseline (điểm %) |', '|---|---|---:|---:|']
    for model in ['yolov8n-seg.pt','yolo11n-seg.pt']:
        rs=[r for r in coco if r['model']==model];base=rs[0]['results']['metrics/mAP50-95(M)']
        for r in rs:
            val=r['results']['metrics/mAP50-95(M)'];lines.append(f"| {model} | {r['variant']} | {val*100:.4f} | {(val-base)*100:+.4f} |")
    lines += ['', 'Nguồn: `sponge-sim/results/coco_defense_20261106/full_remote/results.jsonl`, selector model/variant và `results.metrics/mAP50-95(M)`. Đủ5.000 ảnh mỗi lượt. Đây là metric polygon-label Ultralytics; không giả là kết quả official COCO API độc lập. `metric_inputs.npz` từng model/variant đã replay8/8, sai số tối đa '+str(max(r['max_abs_error'] for r in replay))+'. Conf mAP=0,001 khác conf attack/count=0,25 theo PREREG.',
      '', '| Variant trên ảnh tấn công khóa | N vào kernel min–max | K median |', '|---|---|---:|']
    for variant in ['baseline','maxdet50','prenms300','prenms1000']:
        rs=[r for r in cap if r['kind']=='counts' and r['arm']=='attacked' and r['variant']==variant]
        assert len(rs)==20
        lines.append(f"| {variant} | {min(r['kernel_n'] for r in rs)}–{max(r['kernel_n'] for r in rs)} | {float(np.median([r['k'] for r in rs])):g} |")
    lines += ['', 'Nguồn: `_provenance/codex_gpu_20261106/cap_full_20261106/raw.jsonl`, kind=counts,arm=attacked,variant. Chỉ là đếm workload, không khẳng định latency phòng thủ.',
      '', '- Bản thảo đã tích hợp phần bổ sung trong `paper_en/segmentation_extension/extension.tex`, bảng/figure sinh từ JSON bởi make_extension.py, source map `sources.json`; figure PDF+PNG đã xem kiểm trực quan.',
      '- Đã kiểm cấu trúc LaTeX sau mở rộng các input:31/31 citations có nguồn, không thiếu label/ref, môi trường/ngoặc cân bằng, không placeholder. Hai test mới kiểm số học và gate thiếu phép lặp:2 pass. Máy local không có TeX engine, nên CHƯA build/render PDF toàn bài.',
      '- Backup đầy đủ `sponge-sim/results/confirmation_20261106/codex_evidence_20261106.zip` (183 files,261.796.399 bytes) có SHA/CRC xác minh, gồm tensor, native outputs, weights và run lỗi. `local_backup_verification.json` xác nhận hash replay input. Metadata/raw nhỏ được extract trong `_provenance/codex_gpu_20261106`.',
      '', '## Phát hiện cần biết',
      f"- CPU1 có {len(cpu1['flagged'])} cờ drift; M4 có {len(m4['flagged_references'])}; M4B có {len(m4b['flagged_references'])}. Đều vượt cổng2. Không dùng các phiên này làm bảng timing sạch; đã giữ toàn bộ. M4/M4B không được chạy lặp tới khi có phiên đẹp.",
      '- Kết luận chính mới được chấp nhận: mặt chi phí mask tổng hợp trên CPU4/GPU; M3 đạt K cao ở nguồn; chuyển giao native YOLO26 không đạt K≥50. CHƯA có phiên timing toàn trình tấn công thật qua cổng, nên không kết luận real-attack slowdown chính xác hoặc DoS.',
      '- Sửa claim cũ: 16–20× không được làm headline; “YOLO26 không NMS vẫn K≈300” trước đó chưa dùng đúng inference/counting path. max_det50 không “vô hiệu toàn bộ”: K*=50 vẫn là ngưỡng phát hiện trong điều kiện full-res; cap output chỉ chặn mask, không chặn NMS.',
      '- Hướng phân đoạn còn là nghiên cứu trên nhiễu số toàn ảnh và workload tổng hợp; chưa print-camera, chưa ARM ổn định, chưa chứng minh hiệu lực tấn công riêng lên YOLO26 native.',
      '- main.tex đã dirty từ trước phiên. Các sửa tích hợp appendix/scope/AI declaration được giữ trong working tree; không commit toàn file để tránh gom chỉnh sửa cũ của người dùng. Nguồn extension, script, evidence và báo cáo được commit riêng.',
      '', '## Việc tiếp theo — chọn 1 trong các cách sau',
      'A. Rà nội dung nghiên cứu để quyết định khung bài hợp nhất → gõ: “Rà bài sau bộ kết quả phân đoạn mới.”',
      'B. Build PDF và sửa layout khi có TeX engine → gõ: “Biên dịch và kiểm PDF paper_en.”',
      '', '## Đang chờ gì ở tôi',
      'Không cần duyệt thêm phép đo. Đã dừng sshd trên máy thuê sau khi backup đủ; không còn tiến trình Python đo đạc theo final readback. Người dùng cần TRẢ MÁY trên dịch vụ để dừng phí8k/giờ; dừng SSH không dừng tính tiền. Không xoá dữ liệu remote hay sửa power plan.']
    (repo/'docs/reports/FULL_SEGMENTATION_CODEX_20261106.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote full report with raw selectors')
