"""Generate expanded-cohort report and distribution figure from complete raw records."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__=='__main__':
    repo=Path(__file__).resolve().parents[3];run=repo/'sponge-sim/results/validation_500_20261106/full'
    s=json.loads((run/'summary.json').read_text());assert s['complete'] and s['n_images']==500 and not s['smoke']
    mf=json.loads((run/'manifest.json').read_text());raw=[json.loads(x) for x in (run/'raw.jsonl').read_text().splitlines()]
    counts={(r['model'],r['image'],r['arm'],r['variant']):r for r in raw if r['kind']=='counts'}
    native={(r['image'],r['arm']):r['k'] for r in raw if r['kind']=='native26'}
    images=[r['file_name'] for r in mf['cohort']['images']]
    out=repo/'docs/reports'
    lines=['# Mở rộng COCO-val500 — 06/11/2026','',
      '## Đã xong',
      '- Đã mở rộng từ20 ảnh COCO128 lên500 ảnh COCO val khác, chọn ngẫu nhiên không hoàn lại seed20261106 trước chạy. Hai model nguồn tối ưu riêng từng ảnh:1000 trường hợp tấn công;8.000 records count qua clean/attacked và4 cấu hình cap. Chuyển giao cùng nhiễu v8→YOLO26 native trên cả500 ảnh.',
      '- PREREG/tag `prereg-val500-20261106`, cohort/hash trong `sponge-sim/studies/validation_500/cohort.json`. PGD100 bước,eps16/255,alpha2/255,temp0,02,final iterate,FP32,batch4,không chọn ảnh/seed tốt nhất. Đây là nhiễu toàn ảnh, chưa là patch vật lý.',
      '', '| Model/variant | K clean median | K attacked median [CI95%] | K≥50 (n/500, Wilson95%) | K≥100 | K=300 |',
      '|---|---:|---|---|---:|---:|']
    for model,variants in s['models'].items():
        for variant,v in variants.items():
            ci=v['attacked_median_ci95'];r=v['reach']['50'];p=r['wilson_ci95']
            lines.append(f"| {model}/{variant} | {v['clean_median']:g} | {v['attacked_median']:g} [{ci[0]:g};{ci[1]:g}] | {r['count']}/500 [{100*p[0]:.2f}%;{100*p[1]:.2f}%] | {v['reach']['100']['count']} | {v['reach']['300']['count']} |")
    v=s['native26'];p=v['reach']['50']['wilson_ci95'];ci=v['attacked_median_ci95']
    lines.append(f"| YOLO26 native/transfer v8 | {v['clean_median']:g} | {v['attacked_median']:g} [{ci[0]:g};{ci[1]:g}] | {v['reach']['50']['count']}/500 [{100*p[0]:.2f}%;{100*p[1]:.2f}%] | {v['reach']['100']['count']} | {v['reach']['300']['count']} |")
    lines += ['', 'Nguồn: `sponge-sim/results/validation_500_20261106/full/summary.json:models,native26`; raw selector kind=counts/model/image/arm/variant hoặc kind=native26/image/arm. CI median bootstrap20.000 ảnh seed20261106; tỷ lệ dùng Wilson. CI có điều kiện trên cohort/cấu hình đã cố định, không là khoảng tin cậy qua thiết bị hay seed tối ưu khác.',
      '', '## Phát hiện cần biết',
      '- K≥50 là cutoff đếm chung để đối chiếu; không được coi K*=50 là ngưỡng latency đã hiệu chuẩn cho mọi model. Native YOLO26 không qua NMS, chỉ đếm score column4>0,25 trong tensor[B,300,38], one-to-one bật rõ.',
      '- Tỷ lệ đạt cutoff không tự chứng minh mức tăng do tấn công nếu ảnh sạch đã đạt cutoff. Kiểm tra phụ sau chạy dưới đây báo số ảnh VƯỢT MỚI từ clean<50 sang attacked≥50, cùng chiều ngược; đây là mô tả exploratory, không thay endpoint đã đăng ký.',
      '', '| Model baseline | Clean≥50 | Attacked≥50 | Vượt mới | Rơi dưới50 |', '|---|---:|---:|---:|---:|']
    for model in s['models']:
        clean=np.array([counts[model,i,'clean','baseline']['k'] for i in images]);adv=np.array([counts[model,i,'attacked','baseline']['k'] for i in images])
        lines.append(f"| {model} | {int((clean>=50).sum())} | {int((adv>=50).sum())} | {int(((clean<50)&(adv>=50)).sum())} | {int(((clean>=50)&(adv<50)).sum())} |")
    clean=np.array([native[i,'clean'] for i in images]);adv=np.array([native[i,'attacked'] for i in images])
    lines.append(f"| YOLO26 native transfer | {int((clean>=50).sum())} | {int((adv>=50).sum())} | {int(((clean<50)&(adv>=50)).sum())} | {int(((clean>=50)&(adv<50)).sum())} |")
    lines += ['', '- Mẫu500 mới không được gộp với20/30 ảnh cũ. Kết quả giữa hai model nguồn là mô tả, không quy nguyên nhân cho kiến trúc. Negative transfer không đồng nghĩa miễn nhiễm với tấn công tối ưu riêng trên YOLO26.',
      '- Nghiên cứu này không đo latency: elapsed_s là thời gian xử lý PGD để lập kế hoạch tài nguyên. Các timing campaign thất bại vẫn giữ quality=false trong STABILITY_CODEX_20261106.md; không dùng counts500 để làm chúng thành pass.',
      '- Chi phí accuracy của cap vẫn lấy riêng từ full COCO val5000 ở FULL_SEGMENTATION_CODEX_20261106.md; threshold mAP0,001 khác threshold count0,25.',
      '- Count/attack dùng resize640 vuông trực tiếp; full mAP sạch dùng letterbox640. Hai phép đo không tạo thành đường trade-off accuracy–latency cùng preprocessing. Chưa có AP trên ảnh tấn công trong nghiên cứu500 này.',
      '', '## Việc tiếp theo — chọn 1 trong các cách sau',
      'A. Rà khung bài với dữ liệu lớn hơn → gõ: “Rà bản thảo theo cohort500 mới.”',
      '', '## Đang chờ gì ở tôi',
      'Không cần duyệt thêm cho nghiên cứu mở rộng này. Xem receipt replay/backup cùng run để biết trạng thái tái kiểm số đếm và lưu dữ liệu.']
    replay_path=run/'replay_receipt.json'
    verify_path=run.parent/'local_archive_verification.json'
    if replay_path.exists() and verify_path.exists():
        replay=json.loads(replay_path.read_text());verify=json.loads(verify_path.read_text())
        lines += ['', '### Receipt kiểm dữ liệu',
          f"- Artifact hashes: {replay['artifact_hashes_checked']}; native clean/attacked count checks: {replay['native_counts_verified']}; source-case budget checks: {replay['images_budget_verified']}.",
          f"- Source count replay: {replay['source_count_replays']} records, {replay['mismatch_count']} mismatch. Không ghi đè raw; không tái tối ưu PGD trong replay.",
          f"- ZIP local: {verify['n_files']} tệp, SHA256 `{verify['archive_sha256']}`, CRC pass={verify['zip_crc_pass']}. Deltas/native arrays lưu trong `codex_val500_evidence_20261106.zip` cùng thư mục cha của run.",
          '- Đã sinh phần bổ sung500 và hình trong paper_en/segmentation_extension. Kiểm cấu trúc LaTeX mở rộng inputs không thiếu references/labels hoặc trích dẫn; chưa build PDF toàn bài vì không có TeX engine.',
          '- Máy thuê không còn tiến trình đo/PGD sau hoàn tất. SSH giữ mở theo yêu cầu tiếp tục của người dùng; nếu không làm tiếp cần trả máy trên dịch vụ để dừng8k/giờ.']
    (out/'VAL500_CODEX_20261106.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,(ax,kernel)=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for model,color in [('yolov8n-seg.pt','tab:blue'),('yolo11n-seg.pt','tab:orange')]:
        values=np.sort([counts[model,i,'attacked','baseline']['k'] for i in images])
        capped=s['models'][model]['baseline']['reach']['300']['fraction']*100
        ax.step(values,np.arange(1,501)/500,where='post',label=model.replace('.pt','')+f' attacked (cap: {capped:.1f}%)',color=color)
        values=np.sort([counts[model,i,'clean','baseline']['k'] for i in images])
        ax.step(values,np.arange(1,501)/500,where='post',linestyle='--',color=color,alpha=.6)
    ax.step(np.sort(adv),np.arange(1,501)/500,where='post',label='YOLO26 native transfer',color='tab:green')
    ax.plot([],[],linestyle='--',color='gray',label='Clean source inputs (dashed)')
    ax.set(xlabel='Retained count K',ylabel='Empirical cumulative probability',xlim=(0,305),ylim=(0,1.02),title='New COCO-val500 cohort')
    ax.legend(fontsize=7,frameon=False);ax.grid(alpha=.2)
    variants=['baseline','maxdet50','prenms300','prenms1000'];x=np.arange(4)
    for offset,(model,color) in zip([-.17,.17],[('yolov8n-seg.pt','tab:blue'),('yolo11n-seg.pt','tab:orange')]):
        medians=[float(np.median([counts[model,i,'attacked',v]['kernel_n'] for i in images])) for v in variants]
        kernel.bar(x+offset,medians,width=.32,label=model.replace('.pt',''),color=color)
    kernel.set(xticks=x,xticklabels=['Baseline','Output cap50','Pre-NMS300','Pre-NMS1000'],ylabel='Median kernel input count',title='Candidate workload, attacked images')
    kernel.tick_params(axis='x',labelsize=8);kernel.legend(fontsize=8,frameon=False);kernel.grid(axis='y',alpha=.2)
    fig.savefig(out/'val500_counts_20261106.pdf');fig.savefig(out/'val500_counts_20261106.png',dpi=180);plt.close(fig)
    paper=repo/'paper_en/segmentation_extension'
    for ext in ['pdf','png']:
        (paper/f'val500_counts.{ext}').write_bytes((out/f'val500_counts_20261106.{ext}').read_bytes())
    print('Generated val500 report and figures')
