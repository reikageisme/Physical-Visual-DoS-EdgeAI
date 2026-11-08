"""Document all failed stability experiments; never promote rejected timings."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__=='__main__':
    repo=Path(__file__).resolve().parents[3]
    base=repo/'sponge-sim/results/stability_20261106'
    full=json.loads((base/'stability_full_20261106/confirmation_summary.json').read_text())
    cal=json.loads((base/'stability_cal_20261106/selection.json').read_text())
    conditioning=json.loads((base/'conditioning_cal_20261106/selection.json').read_text())
    out=repo/'docs/reports';out.mkdir(exist_ok=True)
    lines=['# Kiểm tra độ ổn định bổ sung — 06/11/2026','',
      '## Đã xong',
      '- Một campaign hiệu chuẩn ảnh sạch cho ba host affinity, khóa lựa chọn trước timing đầy đủ; ba phiên M6 đầy đủ20 ảnh/16 lặp; một campaign matched-load sạch cho warmup5/50/200. Không đổi xung, nguồn điện hay ngân sách tấn công.',
      '- Source/tag: `prereg-stability-calibration-20261106`, `prereg-stability-full-20261106`, `prereg-conditioning-diagnostic-20261106`. Dataset raw20 vẫn là COCO128; không gộp với cohort val500 mới.',
      '', '| Host profile hiệu chuẩn sạch | Drift flags | MDE (%) | Eligible |', '|---|---:|---:|---|']
    for n,s in cal['profiles'].items():lines.append(f"| {n} | {s['flags']} | {s['mde_frac']*100:.3f} | {s['eligible']} |")
    lines += ['',f"Lựa chọn controls-only: `{cal['selected']}`; SHA selection ghi trong từng manifest M6. Source: `sponge-sim/results/stability_20261106/stability_cal_20261106/selection.json`.",
      '', '| Phiên M6 | Drift flags | MDE clean/attacked (%) | Ratio full-res [CI95%] | Quality |','|---|---:|---|---|---|']
    fig,(ax,forest)=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for s in full['all_sessions']:
        r=s['contrasts']['total_tv_True'];ci=r['ci95'];n=s['session']
        lines.append(f"| {n} | {len(s['flagged_references'])} | {s['mde']['clean']['mde_frac']*100:.3f}/{s['mde']['attacked']['mde_frac']*100:.3f} | {r['geometric_ratio']:.4f} [{ci[0]:.4f};{ci[1]:.4f}] | {s['quality_pass']} |")
        rows=[json.loads(x) for x in (base/f'stability_full_20261106/session{n}/raw.jsonl').read_text().splitlines()]
        ratios=[row['ratio'] for row in rows if row['kind']=='reference']
        ax.plot(range(len(ratios)),ratios,marker='.',label=f'Session {n}: rejected')
        forest.errorbar(r['geometric_ratio'],n,xerr=[[r['geometric_ratio']-ci[0]],[ci[1]-r['geometric_ratio']]],fmt='o',capsize=3)
    ax.axhspan(.97,1.03,color='green',alpha=.12,label='Registered +/-3% band');ax.axhline(1,color='gray',lw=.7)
    ax.set(xlabel='Reference checkpoint',ylabel='Reference / initial median',title='Reference gate fails');ax.legend(fontsize=7,frameon=False)
    forest.axvline(1,color='gray',lw=.7);forest.set(xlabel='Paired total-time ratio [95% CI]',ylabel='Session',yticks=[1,2,3],title='Diagnostic only: quality-rejected sessions');forest.grid(alpha=.2)
    fig.savefig(out/'stability_diagnostic_20261106.pdf');fig.savefig(out/'stability_diagnostic_20261106.png',dpi=170);plt.close(fig)
    lines += ['', 'Các ratio trên là DIAGNOSTIC, không được dùng làm kết luận timing sạch. Ba phiên báo toàn bộ; không chọn phiên đẹp. Aggregate đúng định trước (median log-ratio theo ảnh qua cả3 phiên) chỉ là descriptive vì stable=false.',
      '', '| Warmup sau matched mask load sạch | Flags | MDE (%) | Eligible |', '|---|---:|---:|---|']
    for n,s in conditioning['profiles'].items():lines.append(f"| {n} | {s['flags']} | {s['mde_frac']*100:.3f} | {s['eligible']} |")
    lines += ['', 'Source: `.../conditioning_cal_20261106/selection.json` và `*/raw.jsonl`. selected=null, vì vậy KHÔNG mở campaign M7 hay lặp tiếp để tìm phiên pass.',
      '', '## Phát hiện cần biết',
      '- Paired A/A thấp trong khi reference tuyệt đối đổi mức; không được dùng A/A thấp để bỏ qua cổng drift đã đăng ký. Tăng warmup từ5 lên200 không giải quyết được ổn định trên môi trường này.',
      '- NVML read-only trong matched-load cho thấy P-state1, memory13801MHz và SM khoảng2760–2775MHz tại các snapshot. Biến thiên xung nhỏ này chưa giải thích được đỉnh/đổi mức reference lớn; không kết luận “do GPU clock” chỉ từ telemetry snapshot, cũng chưa xác định được host scheduling/WDDM/cache là nguyên nhân.',
      '- Không ghi đè các run trước; không nới cổng3%/2flags, không chọn số theo độ lớn hiệu ứng. Giữ nguyên giới hạn timing trong paper.',
      '- Người dùng yêu cầu tăng số ảnh: cohort COCO-val500 đã khóa riêng bằng seed20261106, không giao với20 ảnh COCO128 cũ. Nghiên cứu mở rộng là counts/transfer/caps, không suy latency từ thời gian xử lý PGD.',
      '', '## Việc tiếp theo — chọn 1 trong các cách sau',
      'A. Xem cohort500 mở rộng trên hai model trong VAL500_CODEX_20261106.md; không trộn nó với các timing failures trong báo cáo này.',
      '', '## Đang chờ gì ở tôi',
      'Không cần duyệt thêm cho diagnostic này. Không chạy thêm absolute-timing campaign khi chưa có thay đổi môi trường cụ thể; trạng thái máy hiện hành nằm ở cuối PROGRESS.md.']
    (out/'STABILITY_CODEX_20261106.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote honest stability diagnostic and plots')
