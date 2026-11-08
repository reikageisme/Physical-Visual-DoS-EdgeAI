"""Generate nested-cohort report, paper subsection and figures from final JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import load, TASKS, SEEDS

LABELS = {'v8': 'YOLOv8n-seg', 'v11': 'YOLO11n-seg', 'y26_many': 'YOLO26, many objective', 'y26_native': 'YOLO26, native objective'}


def pct(value): return f'{100*value:.2f}%'
def interval(values): return f'[{values[0]:.2f}, {values[1]:.2f}]'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder', type=Path); ap.add_argument('--repo', type=Path, required=True)
    a = ap.parse_args(); a.folder = a.folder.resolve(); a.repo = a.repo.resolve()
    cohort_path = a.repo/'sponge-sim/studies/large_seed/cohort1500.json'
    cohort = json.loads(cohort_path.read_text()); sources = {}
    summaries = {}
    for n in [1000, 1500]:
        p = a.folder/f'summary{n}.json'; summaries[n] = json.loads(p.read_text()); assert summaries[n]['complete']
        sources[str(p.relative_to(a.repo)).replace('\\', '/')] = hashlib.sha256(p.read_bytes()).hexdigest()
        verified = a.folder/f'stage{n}_local_verification.json'
        verification = json.loads(verified.read_text()); assert verification['zip_crc_pass']
        sources[str(verified.relative_to(a.repo)).replace('\\', '/')] = hashlib.sha256(verified.read_bytes()).hexdigest()
    audit_path = a.folder.parent/'audit_replay/receipt.json'
    audit = json.loads(audit_path.read_text()); assert audit['audit_cases'] == 1280
    sources[str(audit_path.relative_to(a.repo)).replace('\\', '/')] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    sensitivity_path = a.folder/'bootstrap_sensitivity.json'
    sensitivity = json.loads(sensitivity_path.read_text())['cohorts']['1500']['y26_many:native26']
    sources[str(sensitivity_path.relative_to(a.repo)).replace('\\', '/')] = hashlib.sha256(sensitivity_path.read_bytes()).hexdigest()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
    for ax, n in zip(axes, [1000, 1500]):
        clean, attacked, _ = load(a.folder, cohort, [1000] if n == 1000 else [1000, 1500])
        for t, task in enumerate(TASKS):
            values = np.sort(attacked[:, :, t, 4].mean(1))
            ax.step(values, np.arange(1, n+1)/n, where='post', label=LABELS[task])
        ax.set(xlabel='Native YOLO26 K (mean across five starts)', ylabel='Fraction of images', title=f'Nested {n:,} images', xlim=(0, 305), ylim=(0, 1.01))
        ax.axvline(50, color='gray', linestyle=':', linewidth=.8)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc='lower right')
    reports = a.repo/'docs/reports'; stem = reports/'NESTED1500_CODEX_20261106'
    fig.savefig(str(stem)+'_native.pdf'); fig.savefig(str(stem)+'_native.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
    for ax, n in zip(axes, [1000, 1500]):
        for t, task in enumerate(TASKS):
            stat = summaries[n]['tasks'][task]['native26']
            x = np.arange(5)+(t-1.5)*.18
            y = [stat['per_start'][str(s)]['reach']['50']['fraction'] for s in SEEDS]
            ax.bar(x, np.asarray(y)*100, width=.18, label=LABELS[task])
        ax.set(xticks=np.arange(5), xticklabels=['zero', '12345', '23456', '34567', '45678'], ylabel='Native K >= 50 (%)', title=f'Nested {n:,} images', ylim=(0, 105))
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(str(stem)+'_starts.pdf'); fig.savefig(str(stem)+'_starts.png', dpi=180); plt.close(fig)
    lines = ['# Nested 1.000/1.500 ảnh, năm khởi tạo — 06/11/2026', '', '## Đã xong', '',
        '- Hoàn tất 30.000 trường hợp: 1.500 ảnh × 5 khởi tạo × 4 hướng tấn công. Mốc 1.000 là tập con của 1.500; đo lại cả 500 ảnh cũ với cấu hình đồng nhất.',
        '- PGD toàn ảnh 100 bước, eps=16/255, alpha=2/255, FP32, không fuse, CUDA Graph batch4. Một khởi tạo bằng 0 và bốn khởi tạo ngẫu nhiên thực sự. Đây chưa phải patch cục bộ hay phép thử camera.',
        f'- Đã kiểm tra hash/CRC và số đếm từ tensor lưu. Chạy lại mô hình trên 1.280 trường hợp audit: {audit["count_comparisons"]:,} so sánh K, {audit["count_mismatches"]} sai khác. Mọi kết quả gốc được giữ nguyên.',
        '- Nguồn số: `sponge-sim/results/large_seed_20261106/export/stage*_records.jsonl`, selectors task/seed/images/clean_variants/attacked_variants/*native_k; tổng hợp `summary1000.json`, `summary1500.json`.', '',
        '| Ảnh | Hướng tối ưu | Đầu ra đánh giá | Trung vị K sạch → nhiễu (trung bình 5 khởi tạo mỗi ảnh) | Tỷ lệ K≥50 trung bình qua khởi tạo [CI95%] |',
        '|---:|---|---|---:|---:|']
    tex = [r'\subsection{Nested cohorts and target-specific attacks on an NMS-free model}',
        r'\label{sec:nested1500}',
        r'We recomputed a nested 1,000/1,500-image COCO-val cohort with five starts per image and four attack paths (30,000 image--start attacks). The prefixes are not independent replications. All paths use 100 PGD steps, $\epsilon=16/255$, $\alpha=2/255$, FP32 unfused evaluation, and the final iterate. A zero start and four image-keyed uniform random starts were fixed prospectively. The two target-specific objectives share the same initial perturbation for each image and start.',
        r'\begin{table*}[t]\centering\small',
        r'\caption{Nested cohort count endpoints. K is averaged over five starts per image before taking the median. Reach is the average image--start fraction; hierarchical 95\% intervals resample images then starts within each sampled image (20,000 draws). Native outputs use score column 4 above 0.25 and no NMS.}',
        r'\label{tab:nested1500}\begin{tabular}{rlrr}\hline',
        r'Images & Objective/evaluation & Median K [95\% CI] & Reach $K\geq50$ (\%)\\\hline']
    for n in [1000, 1500]:
        for task, variant in [('v8','baseline'), ('v11','baseline'), ('v8','native26'), ('v11','native26'), ('y26_many','native26'), ('y26_native','native26')]:
            s = summaries[n]['tasks'][task][variant]
            ci = s['mean_start_reach50_ci95']; reach = s['mean_start_reach50']
            label = LABELS[task]; evaluation = 'native26' if variant == 'native26' else 'many + torchvision NMS'
            lines.append(f'| {n} | {label} | {evaluation} | {s["clean_median_image_mean"]:.2f} → {s["attacked_median_image_mean"]:.2f} | {pct(reach)} [{pct(ci[0])}, {pct(ci[1])}] |')
            short = {'v8':'v8', 'v11':'11', 'y26_many':'26-many', 'y26_native':'26-native'}[task]+'/'+('native26' if variant == 'native26' else 'own')
            tex.append(f'{n} & {short} & {s["attacked_median_image_mean"]:.2f} {interval(s["attacked_median_image_mean_ci95"])} & {100*reach:.2f} [{100*ci[0]:.2f}, {100*ci[1]:.2f}]'+r'\\')
    tex += [r'\hline\end{tabular}\end{table*}',
        'The earlier zero-of-500 transfer result concerns perturbations optimized on YOLOv8 and evaluated on the native one-to-one YOLO26 head. It does not establish immunity to a target-specific attack. Installed-source inspection and a separately registered four-image diagnostic found finite, nonzero input gradients for both target heads; backbone features are detached only in training mode. The larger study distinguishes source transfer, target one-to-many optimization, and native-score optimization while always evaluating native outputs directly.',
        r'The primary count endpoint is the image-level mean across starts, with hierarchical image--start uncertainty. Per-start Wilson intervals use the image count as the denominator. Any-start success is secondary and spends up to 500 gradient steps per image; it is not an equal-cost comparison with one 100-step start. The shared native $K\geq50$ cutoff is a count threshold, not an empirically established latency threshold for YOLO26.',
        'Empirical bootstrap intervals can collapse at all-zero or all-one endpoints. Such a boundary interval is not population certainty; per-start Wilson bounds are retained in the evidence report. Inference is conditional on the prospectively fixed mixture of one zero and four random starts.',
        f'The registered hierarchical percentile interval for the 1,500-image target-many median is {interval(sensitivity["registered_hierarchical_ci95"])}, whereas its point estimate is {sensitivity["point"]:.1f}. We retain this shift rather than correcting the registered interval after seeing outcomes. An exploratory audit reproduces every registered median interval exactly; an image-only sensitivity interval, conditional on the five observed starts, is {interval(sensitivity["conditional_image_only_ci95"])}. It does not replace uncertainty over both images and starts. The nonlinear median interval is therefore treated cautiously; the paired mean-gain and reach endpoints remain separately reported.',
        r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{segmentation_extension/nested1500_native.pdf}\caption{Native YOLO26 count distributions under both transfer sources and both target-specific objectives. Each image contributes its mean count over five starts.}\label{fig:nested1500}\end{figure*}',
        'Every case retains candidate-score vectors, detections, and native outputs. Exact adversarial inputs are retained only for the prospectively selected 64-image audit subset across all objectives and starts; the remaining cases have input hashes and saved-output verification, not full input replay. These count results do not repair the failed timing quality gates and do not establish an end-to-end denial of service.']
    lines += ['', '| Ảnh | Hướng / đầu ra | Tăng K trung bình ghép cặp [CI95%] | Tỷ lệ sạch K≥50 | Tỷ lệ mới vượt50 (thăm dò) |',
        '|---:|---|---:|---:|---:|']
    for n in [1000, 1500]:
        for task, variant in [('v8','baseline'), ('v11','baseline'), ('v8','native26'), ('v11','native26'), ('y26_many','native26'), ('y26_native','native26')]:
            s = summaries[n]['tasks'][task][variant]
            lines.append(f'| {n} | {task}/{variant} | {s["mean_paired_gain"]:.2f} {interval(s["mean_paired_gain_ci95"])} | {pct(s["exploratory_clean_mean_start_reach50"])} | {pct(s["exploratory_new_crossing50_fraction"])} |')
    lines += ['', '## Phát hiện cần biết', '',
        '- Kết quả 0/500 trước đó là chuyển nhiễu v8 → YOLO26 native ở ngân sách cụ thể. Thí nghiệm cũ có K lớn trên YOLO26 lại tự tối ưu chính model đích và dùng nhánh one-to-many. Hai câu hỏi khác nhau.',
        '- Mã installed chỉ detach đặc trưng backbone khi training; eval có gradient đầu vào ở cả hai nhánh. Diagnostic 4 ảnh giữ vai trò thăm dò, không thay thế 1.500 ảnh.',
        '- Native K≥50 chỉ là mốc đếm chung, chưa phải ngưỡng chậm được đo trên YOLO26. Kết quả timing thất bại vẫn giữ nhãn thất bại; không suy ra DoS từ K.',
        '- Mỗi seed dùng Wilson với n=ảnh. “Có ít nhất một trong năm khởi tạo thành công” tốn tối đa 500 bước, không so ngang ngân sách 100 bước. CI có điều kiện trên hỗn hợp một zero và bốn random starts đã khóa.',
        '- Bootstrap thực nghiệm có thể cho khoảng rộng bằng 0 ở tỷ lệ toàn 0/toàn 1; không diễn giải là chắc chắn cho quần thể. Wilson từng seed bên dưới vẫn có độ rộng ở biên.', '',
        f'- CI trung vị 26-many ở 1.500 ảnh = {interval(sensitivity["registered_hierarchical_ci95"])} không chứa ước lượng điểm {sensitivity["point"]:.1f}; giữ nguyên kết quả đã đăng ký. Kiểm tra thăm dò sau kết quả tái tạo chính xác 12 CI; trung tâm bootstrap lệch {sensitivity["hierarchical_median_shift"]:.1f} K. Image-only CI = {interval(sensitivity["conditional_image_only_ci95"])} chỉ có điều kiện trên năm khởi tạo, không thay thế CI hai tầng. Nguồn: `export/bootstrap_sensitivity.json`.', '',
        '| Hướng | Confidence native hạng50 sạch → nhiễu, trung vị của trung bình 5 starts (thăm dò) |', '|---|---:|']
    for task in TASKS:
        r = summaries[1500]['tasks'][task]['exploratory_native_confidence']
        lines.append(f'| {LABELS[task]} | {r["clean_median_image_mean_ranks_1_50_100_300"][1]:.6f} → {r["attacked_median_image_mean_ranks_1_50_100_300"][1]:.6f} |')
    lines += ['',
        '| Hướng | Seed | Native K≥50 / 1.500 | Wilson95% |', '|---|---:|---:|---:|']
    for task in TASKS:
        s = summaries[1500]['tasks'][task]['native26']
        for seed in SEEDS:
            r = s['per_start'][str(seed)]['reach']['50']; lo, hi = r['wilson_ci95']
            lines.append(f'| {LABELS[task]} | {seed} | {r["hits"]}/{r["n_images"]} | [{pct(lo)}, {pct(hi)}] |')
        r = s['any_start']['50']; lo, hi = r['wilson_ci95']
        lines.append(f'| {LABELS[task]} | bất kỳ 1/5 | {r["hits"]}/{r["n_images"]} | [{pct(lo)}, {pct(hi)}] |')
    lines += ['', '## Việc tiếp theo — chọn 1 trong các cách sau', '',
        'A. Rà phản biện và ghép bản thảo → gõ: "Rà phản biện kết quả 1500 ảnh và hoàn thiện paper_en".', '',
        '## Đang chờ gì ở tôi', '', 'Không cần xác nhận thí nghiệm đã hoàn tất. Nếu ngừng dùng máy thuê, trả máy trên dịch vụ để dừng phí 8k/giờ; SSH vẫn để mở theo yêu cầu.', '',
        '### Truy vết và kiểm tra', '',
        '- Mọi số bảng: `summary{1000,1500}.json:tasks.<task>.<variant>`. CI chính: `*_ci95`; bảng seed: `per_start.<seed>.reach.50`; any-start: `any_start.50`.',
        '- Ngẫu nhiên/bootstrap: 20.000 lần, RNG20261106; lấy lại ảnh rồi năm khởi tạo trong từng ảnh; cùng chỉ số lấy mẫu cho các endpoint ghép cặp.',
        '- Hash, CRC và coverage: `export/stage{1000,1500}_{receipt,local_verification}.json`. Forward replay riêng: `audit_replay/receipt.json`. Exact input chỉ 64 ảnh đã khóa, không tuyên bố replay toàn bộ 30.000 case.',
        '- Archive giữ mọi attempt, chọn attempt hoàn tất đầu tiên theo quy tắc trước chạy; không chọn theo hiệu ứng.',
        '- Hình `NESTED1500_CODEX_20261106_examples.png` dùng vị trí 0/21/42/63 trong 64 audit IDs theo thứ tự cohort, seed 0 và đủ bốn hướng; không chọn ảnh theo kết quả. Selector/hash ở file `_examples_sources.json`.',
        '- Đã bổ sung `paper_en/segmentation_extension/nested1500.tex`; kiểm cấu trúc LaTeX, trích dẫn và nhãn qua. Chưa biên dịch PDF toàn bài vì máy local thiếu TeX. Main.tex có thay đổi của người dùng từ trước, không được commit toàn bộ.',
        '- Dữ liệu cũ 500 ảnh, kết quả âm và mọi phiên timing trượt cổng được giữ nguyên.']
    with (reports/'NESTED1500_CODEX_20261106.md').open('x', encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    extension = a.repo/'paper_en/segmentation_extension'
    with (extension/'nested1500.tex').open('x', encoding='utf-8') as f: f.write('\n'.join(tex)+'\n')
    (extension/'nested1500_native.pdf').write_bytes((reports/'NESTED1500_CODEX_20261106_native.pdf').read_bytes())
    (extension/'nested1500_sources.json').write_text(json.dumps(sources, indent=2))
    print('Generated final nested1500 report, figures and standalone paper subsection')
