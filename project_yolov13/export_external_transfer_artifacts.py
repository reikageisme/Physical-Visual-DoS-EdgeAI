from __future__ import annotations
import csv
import json
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, project_path, require_local_venv, utc_timestamp, write_json

configure_project_environment()


def main() -> int:
    require_local_venv()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = project_path('results/benchmark_adv_tshirt_transfer.csv', must_exist=True)
    with source.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if row.get('status') == 'ok']
    if not rows:
        raise RuntimeError('No successful transfer benchmark rows to export.')
    family_order = ['YOLOv8', 'YOLOv10', 'YOLO11', 'YOLOv12', 'YOLOv13']
    scale_order = ['n', 's', 'l']
    rows.sort(key=lambda r: (scale_order.index(r['scale']), family_order.index(r['family'])))

    def n(row: dict, key: str, digits: int = 2) -> str:
        return f'{float(row[key]):.{digits}f}'

    header = '| Model | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | FLOPs (G) | GPU latency (ms) | GPU FPS | CPU latency (ms) | CPU FPS |'
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    md_lines = [
        '# AdvT-shirt-1K zero-shot transfer benchmark',
        '',
        'COCO-pretrained person detectors evaluated without fine-tuning on the full available AdvT-shirt-1K mirror (1,123 images; 4,403 person boxes). This external transfer set is distinct from the missing NCKH task dataset.',
        '',
        'Latency is median model-forward time from 5 warm-ups and 30 timed FP32 calls at 640x640, batch 1; it excludes preprocessing, postprocessing, image transfer, and model loading. The exact same protocol was used for CPU and GPU. Metrics were computed on the complete corpus using the COCO person class.',
        '',
        header,
        sep,
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model_name']} | {100*float(r['precision']):.2f} | {100*float(r['recall']):.2f} | {100*float(r['map50']):.2f} | {100*float(r['map50_95']):.2f} | {float(r['params_m']):.3f} | {float(r['flops_g']):.2f} | {float(r['gpu_forward_median_ms']):.2f} | {float(r['gpu_fps']):.1f} | {float(r['cpu_forward_median_ms']):.2f} | {float(r['cpu_fps']):.1f} |"
        )
    md_lines += ['', f"Models evaluated: {len(rows)}. Full per-model hashes and run metadata are in `{source.relative_to(PROJECT_ROOT).as_posix()}`."]
    md_path = project_path('results/external_transfer_comparison.md')
    md_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

    tex_lines = [
        '% Requires \\usepackage{booktabs,graphicx}',
        '\\begin{table*}[t]',
        '\\centering\\scriptsize',
        '\\caption{Zero-shot transfer evaluation of COCO-pretrained person detectors on the complete available AdvT-shirt-1K mirror (1,123 images, 4,403 person boxes). No fine-tuning was performed. Latencies are median model-forward times at 640\\times640, batch 1, FP32; preprocessing and postprocessing are excluded.}',
        '\\label{tab:adv-tshirt-transfer}',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{lrrrrrrrrrr}',
        '\\toprule',
        'Model & Precision (\\%) & Recall (\\%) & mAP@0.5 (\\%) & mAP@0.5:0.95 (\\%) & Params (M) & FLOPs (G) & GPU ms & GPU FPS & CPU ms & CPU FPS \\\\',
        '\\midrule',
    ]
    for idx, r in enumerate(rows):
        if idx and r['scale'] != rows[idx-1]['scale']:
            tex_lines.append('\\midrule')
        tex_lines.append(
            f"{r['model_name']} & {100*float(r['precision']):.2f} & {100*float(r['recall']):.2f} & {100*float(r['map50']):.2f} & {100*float(r['map50_95']):.2f} & {float(r['params_m']):.3f} & {float(r['flops_g']):.2f} & {float(r['gpu_forward_median_ms']):.2f} & {float(r['gpu_fps']):.1f} & {float(r['cpu_forward_median_ms']):.2f} & {float(r['cpu_fps']):.1f} \\\\"
        )
    tex_lines += ['\\bottomrule', '\\end{tabular}%', '}', '\\end{table*}']
    tex_path = project_path('results/external_transfer_comparison.tex')
    tex_path.write_text('\n'.join(tex_lines) + '\n', encoding='utf-8')

    family_colors = {'YOLOv8': '#4c78a8', 'YOLOv10': '#f58518', 'YOLO11': '#54a24b', 'YOLOv12': '#e45756', 'YOLOv13': '#b279a2'}
    scale_markers = {'n': 'o', 's': 's', 'l': '^'}
    for x_key, x_label, file_name in [
        ('gpu_forward_median_ms', 'GPU model-forward latency (ms)', 'external_transfer_tradeoff_latency.png'),
        ('flops_g', 'Model FLOPs at 640x640 (G)', 'external_transfer_tradeoff_flops.png'),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        for family in family_order:
            for scale in scale_order:
                subset = [r for r in rows if r['family'] == family and r['scale'] == scale]
                if not subset:
                    continue
                ax.scatter([float(r[x_key]) for r in subset], [100*float(r['map50_95']) for r in subset],
                           color=family_colors[family], marker=scale_markers[scale], s=72,
                           label=f'{family} {scale.upper()}', alpha=0.9)
                for r in subset:
                    ax.annotate(r['model_name'].replace('YOLO', ''),
                                (float(r[x_key]), 100*float(r['map50_95'])),
                                xytext=(4, 3), textcoords='offset points', fontsize=7)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Person mAP@0.5:0.95 (%)')
        ax.set_title('AdvT-shirt-1K zero-shot transfer: accuracy and model cost')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=3)
        fig.savefig(project_path(f'results/{file_name}'), dpi=220)
        plt.close(fig)

    by_name = {r['model_name']: r for r in rows}
    deltas = {}
    for scale in scale_order:
        ours = by_name.get(f'YOLOv13-{scale.upper()}')
        if not ours:
            continue
        deltas[scale] = {}
        for family in family_order[:-1]:
            baseline = by_name.get(f'{family}-{scale.upper()}')
            if baseline:
                deltas[scale][family] = {
                    'map50_95_delta_points': (float(ours['map50_95']) - float(baseline['map50_95'])) * 100,
                    'gpu_latency_delta_ms': float(ours['gpu_forward_median_ms']) - float(baseline['gpu_forward_median_ms']),
                    'flops_delta_g': float(ours['flops_g']) - float(baseline['flops_g']),
                }
    manifest = {
        'created_at_utc': utc_timestamp(),
        'source_csv': 'results/benchmark_adv_tshirt_transfer.csv',
        'successful_rows': len(rows),
        'comparison_markdown': 'results/external_transfer_comparison.md',
        'comparison_latex': 'results/external_transfer_comparison.tex',
        'tradeoff_plots': ['results/external_transfer_tradeoff_latency.png', 'results/external_transfer_tradeoff_flops.png'],
        'yolov13_deltas_by_scale': deltas,
        'scope': 'External zero-shot transfer benchmark only; not fine-tuned project-dataset performance.',
    }
    write_json(project_path('results/external_transfer_export_manifest.json'), manifest)
    print(f'Exported {len(rows)} benchmark rows, LaTeX/Markdown tables, and two trade-off plots.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
