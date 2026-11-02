"""Generate academic result tables, efficiency plots, and an evidence-based ablation summary."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from pipeline_utils import (
    PROJECT_ROOT,
    configure_project_environment,
    project_path,
    require_local_venv,
    write_csv,
    write_json,
)

configure_project_environment()


RESULT_FIELDS = [
    "model_name", "family", "scale", "split", "device", "precision", "recall",
    "map50", "map50_95", "params_m", "flops_g", "latency_ms", "fps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/benchmark_eval.csv")
    parser.add_argument("--paper-results", default="configs/paper_results.csv")
    parser.add_argument("--paper-ablation", default="configs/paper_ablation.csv")
    parser.add_argument("--paper-device-latency", default="configs/paper_device_latency.csv")
    parser.add_argument("--ablation", default="results/ablation.csv")
    parser.add_argument("--benchmark-config", default="configs/benchmark.yaml")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fmt(value, digits=3):
    parsed = number(value)
    return "--" if parsed is None else f"{parsed:.{digits}f}"


def fmt_fraction_as_percent(value, digits=2):
    parsed = number(value)
    return "--" if parsed is None else f"{parsed * 100.0:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def latex_escape(value) -> str:
    text = str(value)
    for source, replacement in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
        ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]:
        text = text.replace(source, replacement)
    return text


def latex_table(headers: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    alignment = "l" + "r" * (len(headers) - 1)
    body = [
        "% Requires \\usepackage{booktabs,graphicx}",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{" + latex_escape(caption) + "}",
        r"\label{" + latex_escape(label) + "}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(latex_escape(value) for value in headers) + r" \\",
        r"\midrule",
    ]
    body.extend(" & ".join(latex_escape(value) for value in row) + r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""])
    return "\n".join(body)


def make_local_rows(results: list[dict], config: dict) -> list[list[str]]:
    if results:
        output = []
        for row in results:
            output.append([
                row.get("model_name", "--"), row.get("split", "--"), row.get("device", "--"),
                fmt_fraction_as_percent(row.get("precision")), fmt_fraction_as_percent(row.get("recall")),
                fmt_fraction_as_percent(row.get("map50")), fmt_fraction_as_percent(row.get("map50_95")),
                fmt(row.get("params_m"), 2), fmt(row.get("flops_g"), 2),
                fmt(row.get("latency_ms"), 2), fmt(row.get("fps"), 1),
            ])
        return output

    # Provide a manuscript-ready structure before local evaluations have been run.
    models = config.get("models", [])
    return [[
        model.get("name", "Model"), "val/test", "CPU/GPU", "--", "--", "--", "--", "--", "--", "--", "--"
    ] for model in models]


def make_paper_rows(paper: list[dict]) -> list[list[str]]:
    output = []
    for row in paper:
        latency = number(row.get("latency_ms"))
        fps = 1000.0 / latency if latency and latency > 0 else None
        output.append([
            row.get("model", "--"), row.get("dataset", "--"), fmt(row.get("precision")),
            fmt(row.get("recall")), fmt(row.get("map50")), fmt(row.get("map50_95")),
            fmt(row.get("params_m"), 2), fmt(row.get("flops_g"), 2), fmt(latency, 2),
            row.get("latency_device", "--"), fmt(fps, 1),
        ])
    return output


def write_tradeoff_plots(results: list[dict], paper: list[dict], out_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Run setup_env.bat inside this project.") from exc

    plots = []
    if results:
        data = [
            row for row in results
            if number(row.get("map50_95")) is not None and number(row.get("latency_ms")) is not None
        ]
        title = "Project dataset: measured mAP50:95 vs latency"
        suffix = "local"
    else:
        data = [
            row for row in paper
            if number(row.get("map50_95")) is not None and number(row.get("latency_ms")) is not None
        ]
        title = "MS COCO literature context (not project measurements)"
        suffix = "literature_context"
    if data:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for row in data:
            x, y = number(row.get("latency_ms")), number(row.get("map50_95"))
            ax.scatter(x, y, s=42)
            label = row.get("model_name") or row.get("model", "")
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
            ax.set_xlabel("Inference latency (ms/image; Tesla T4, TensorRT FP16)" if suffix != "local" else "Inference latency (ms/image; local validator)")
        ax.set_ylabel("mAP@0.5:0.95 (%)" if number(data[0].get("map50_95")) and number(data[0].get("map50_95")) > 1 else "mAP@0.5:0.95")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = out_dir / f"tradeoff_map_latency_{suffix}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(str(path.relative_to(PROJECT_ROOT)))

    data = [
        row for row in (results or paper)
        if number(row.get("map50_95")) is not None and number(row.get("flops_g")) is not None
    ]
    if data:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for row in data:
            x, y = number(row.get("flops_g")), number(row.get("map50_95"))
            ax.scatter(x, y, s=42)
            label = row.get("model_name") or row.get("model", "")
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("FLOPs (G)")
        metric = number(data[0].get("map50_95"))
        ax.set_ylabel("mAP@0.5:0.95 (%)" if metric and metric > 1 else "mAP@0.5:0.95")
        ax.set_title("mAP@0.5:0.95 vs FLOPs" if results else "MS COCO literature context: mAP@0.5:0.95 vs FLOPs")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        suffix = "local" if results else "literature_context"
        path = out_dir / f"tradeoff_map_flops_{suffix}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(str(path.relative_to(PROJECT_ROOT)))
    return plots


def ablation_report(rows: list[dict]) -> str:
    if not rows:
        return (
            "# Ablation analysis\n\n"
            "No measured ablation data was found. Fill results/ablation.csv with controlled runs listed in "
            "configs/ablation_template.csv. No component-level performance attribution can be made yet.\n"
        )
    measured_fields = ["map50_95", "params_m", "flops_g", "latency_ms"]
    def scale_key(row: dict) -> str | None:
        parts = str(row.get("variant", "")).lower().split("-")
        return parts[1] if len(parts) > 2 and parts[0] == "yolov13" and parts[1] in {"n", "s", "l", "x"} else None

    def is_full(row: dict) -> bool:
        label = str(row.get("variant", "")).lower()
        return label in {"yolov13-full", "full"} or label.endswith("-full")

    full_rows = [row for row in rows if is_full(row)]
    if not full_rows or any(number(row.get(key)) is None for row in full_rows for key in measured_fields):
        return (
            "# Ablation analysis\n\n"
            "A full-model reference row for each compared scale, with numeric mAP50:95, parameters, FLOPs, and latency, is required. "
            "The available rows are retained below but no deltas or causal interpretation are produced.\n"
        )
    lines = [
        "# Ablation analysis",
        "",
        "Deltas below are variant minus the full model at the same scale. Interpret them as evidence for a component only when the split, "
        "initialization, training schedule, augmentation, seed policy, and evaluation device were held constant.",
        "",
    ]
    for full in full_rows:
        scale = scale_key(full)
        group = [row for row in rows if row is not full and scale_key(row) == scale]
        if scale is not None:
            group = [row for row in group if scale_key(row) == scale]
            lines.extend([f"## Scale {scale.upper()}", ""])
        lines.extend([
            "| Variant | Δ mAP@0.5:0.95 (pp) | Δ parameters (M) | Δ FLOPs (G) | Δ latency (ms) |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in group:
            values = [number(row.get(key)) for key in measured_fields]
            if any(value is None for value in values):
                lines.append(f"| {row.get('variant', '--')} | -- | -- | -- | -- |")
                continue
            deltas = [values[i] - number(full.get(key)) for i, key in enumerate(measured_fields)]
            lines.append(
                f"| {row.get('variant', '--')} | {deltas[0] * 100.0:+.2f} | {deltas[1]:+.3f} | "
                f"{deltas[2]:+.3f} | {deltas[3]:+.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def paper_ablation_rows(rows: list[dict]) -> list[list[str]]:
    return [[
        row.get("experiment", "--"), row.get("variant", "--"), row.get("dataset", "--"),
        fmt(row.get("map50_95"), 1), fmt(row.get("map50"), 1),
        fmt(row.get("params_m"), 1), fmt(row.get("flops_g"), 1),
    ] for row in rows]


def paper_device_latency_rows(rows: list[dict]) -> list[list[str]]:
    output = []
    for row in rows:
        latency = number(row.get("latency_ms"))
        fps = 1000.0 / latency if latency and latency > 0 else None
        output.append([
            row.get("model", "--"), row.get("device", "--"), row.get("runtime", "--"),
            fmt(latency, 2), fmt(fps, 1), row.get("dataset", "--"),
        ])
    return output


def paper_ablation_discussion(rows: list[dict]) -> str:
    lookup = {(row.get("experiment"), row.get("variant")): row for row in rows}
    full = lookup.get(("FullPAD-HyperACE", "All three FullPAD paths"))
    no_routes = lookup.get(("FullPAD-HyperACE", "No distribution (paper treats as HyperACE removed)"))
    ds_n = lookup.get(("DS-blocks", "YOLOv13-N with DS blocks"))
    vanilla_n = lookup.get(("DS-blocks", "YOLOv13-N with vanilla blocks"))
    ds_s = lookup.get(("DS-blocks", "YOLOv13-S with DS blocks"))
    vanilla_s = lookup.get(("DS-blocks", "YOLOv13-S with vanilla blocks"))
    m2 = lookup.get(("Hyperedge-count", "M=2"))
    m8 = lookup.get(("Hyperedge-count", "M=8"))
    m16 = lookup.get(("Hyperedge-count", "M=16"))

    lines = [
        "# Published YOLOv13 ablation reference",
        "",
        "These are the original paper's MS COCO validation ablations, not measurements on this project's dataset.",
        "",
        "| Experiment | Variant | Dataset | mAP@0.5:0.95 (%) | mAP@0.5 (%) | Params (M) | FLOPs (G) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('experiment', '--')} | {row.get('variant', '--')} | {row.get('dataset', '--')} | "
            f"{fmt(row.get('map50_95'), 1)} | {fmt(row.get('map50'), 1)} | "
            f"{fmt(row.get('params_m'), 1)} | {fmt(row.get('flops_g'), 1)} |"
        )
    lines.extend(["", "## Quantitative reading", ""])
    if full and no_routes:
        delta95 = number(full.get("map50_95")) - number(no_routes.get("map50_95"))
        delta50 = number(full.get("map50")) - number(no_routes.get("map50"))
        lines.append(
            f"- On YOLOv13-S, all three FullPAD paths score {delta95:+.1f} percentage points in AP50:95 "
            f"and {delta50:+.1f} points in AP50 over the paper's no-distribution setting, which the authors "
            "treat as removing HyperACE."
        )
    if ds_n and vanilla_n and ds_s and vanilla_s:
        n_flops = number(vanilla_n.get("flops_g")) - number(ds_n.get("flops_g"))
        n_params = number(vanilla_n.get("params_m")) - number(ds_n.get("params_m"))
        s_flops = number(vanilla_s.get("flops_g")) - number(ds_s.get("flops_g"))
        s_params = number(vanilla_s.get("params_m")) - number(ds_s.get("params_m"))
        lines.append(
            f"- The paper's Table V gives DS-block savings of {n_flops:.1f} GFLOPs and {n_params:.1f}M parameters "
            f"for Nano, and {s_flops:.1f} GFLOPs and {s_params:.1f}M for Small, with unchanged AP50:95 "
            "and a 0.1-point lower AP50 in both tabulated comparisons."
        )
        lines.append(
            "- The paper's prose reports different Small-model DS savings (4.2 GFLOPs and 2.2M parameters) "
            "than the Table V arithmetic (6.3 GFLOPs and 2.7M); this report uses the table values and flags the discrepancy."
        )
    if m2 and m8 and m16:
        lines.append(
            f"- For YOLOv13-S, increasing hyperedges from M=2 to M=8 changes AP50:95 by "
            f"{number(m8.get('map50_95')) - number(m2.get('map50_95')):+.1f} points; moving from M=8 to M=16 "
            f"adds {number(m16.get('map50_95')) - number(m8.get('map50_95')):+.1f} points while adding "
            f"{number(m16.get('flops_g')) - number(m8.get('flops_g')):.1f} GFLOPs and "
            f"{number(m16.get('params_m')) - number(m8.get('params_m')):.1f}M parameters."
        )
    lines.extend([
        "",
        "Use these results to motivate the project-specific ablations; do not present them as outcomes on the project's own data.",
        "",
        "Source: [Lei et al., YOLOv13, arXiv:2506.17733](https://arxiv.org/abs/2506.17733), Tables III-V and accompanying discussion.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    require_local_venv()
    import yaml

    results_path = project_path(args.results)
    paper_path = project_path(args.paper_results, must_exist=True)
    paper_ablation_path = project_path(args.paper_ablation, must_exist=True)
    paper_device_path = project_path(args.paper_device_latency, must_exist=True)
    ablation_path = project_path(args.ablation)
    config_path = project_path(args.benchmark_config, must_exist=True)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = read_csv(results_path)
    paper = read_csv(paper_path)
    paper_ablation = read_csv(paper_ablation_path)
    paper_device_latency = read_csv(paper_device_path)
    ablations = read_csv(ablation_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}

    local_headers = [
        "Model", "Split", "Device", "Precision (%)", "Recall (%)", "mAP@0.5 (%)", "mAP@0.5:0.95 (%)",
        "Params (M)", "FLOPs (G)", "Latency (ms/image)", "FPS",
    ]
    local_rows = make_local_rows(results, config)
    local_note = (
        "Metrics below are local checkpoint measurements on the configured project dataset."
        if results else
        "Template only: no local evaluation has been run. Replace -- only with measured project-dataset results."
    )
    local_md = "# Project dataset results\n\n" + local_note + "\n\n" + markdown_table(local_headers, local_rows)
    local_tex = latex_table(
        local_headers, local_rows,
        "Comparison on the project dataset; report validation and test results from the same fixed splits.",
        "tab:project-yolov13-results",
    ) + "\n% " + local_note

    paper_headers = [
        "Model", "Dataset", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95",
        "Params (M)", "FLOPs (G)", "Reported latency (ms)", "Latency device/protocol", "Derived FPS",
    ]
    paper_rows = make_paper_rows(paper)
    paper_md = (
        "# Literature reference: official MS COCO results\n\n"
        "This is a separate literature-context table, not a direct substitute for the project-dataset benchmark. "
        "The paper's implementation details specify a Tesla T4 GPU with TensorRT FP16 for Table I latency; FPS is derived as 1000/latency.\n\n"
        + markdown_table(paper_headers, paper_rows)
    )
    paper_tex = latex_table(
        paper_headers, paper_rows,
        "Official YOLOv13 paper MS COCO comparison. Table I latency uses a Tesla T4 GPU with TensorRT FP16 per Section IV-A; FPS is derived as 1000/latency.",
        "tab:yolov13-coco-literature",
    )
    device_latency_headers = ["Model", "Device", "Runtime", "Latency (ms/image)", "Derived FPS", "Dataset"]
    device_latency_table_rows = paper_device_latency_rows(paper_device_latency)
    device_latency_md = (
        "# Published YOLOv13 device latency\n\n"
        "Paper Table VII reports GPU latency with TensorRT FP16 and CPU latency with ONNX; the CPU used an Intel Xeon Platinum 8352V. "
        "FPS is derived as 1000/latency. These are literature values, not measurements on this project hardware.\n\n"
        + markdown_table(device_latency_headers, device_latency_table_rows)
    )
    device_latency_tex = latex_table(
        device_latency_headers,
        device_latency_table_rows,
        "Published YOLOv13 latency on Tesla T4 and RTX 4090 GPUs (TensorRT FP16), and Intel Xeon Platinum 8352V CPU (ONNX); FPS is derived.",
        "tab:yolov13-paper-device-latency",
    )
    paper_ablation_headers = [
        "Experiment", "Variant", "Dataset", "mAP@0.5:0.95 (%)", "mAP@0.5 (%)", "Params (M)", "FLOPs (G)",
    ]
    paper_ablation_table_rows = paper_ablation_rows(paper_ablation)
    paper_ablation_tex = latex_table(
        paper_ablation_headers,
        paper_ablation_table_rows,
        "Published YOLOv13 ablations on MS COCO validation; not measurements on the project dataset.",
        "tab:yolov13-paper-ablation",
    )

    (out_dir / "comparison_results.md").write_text(local_md, encoding="utf-8")
    (out_dir / "comparison_results.tex").write_text(local_tex, encoding="utf-8")
    (out_dir / "literature_reference.md").write_text(paper_md, encoding="utf-8")
    (out_dir / "literature_reference.tex").write_text(paper_tex, encoding="utf-8")
    (out_dir / "literature_device_latency.md").write_text(device_latency_md, encoding="utf-8")
    (out_dir / "literature_device_latency.tex").write_text(device_latency_tex, encoding="utf-8")
    (out_dir / "paper_ablation_reference.md").write_text(paper_ablation_discussion(paper_ablation), encoding="utf-8")
    (out_dir / "paper_ablation_reference.tex").write_text(paper_ablation_tex, encoding="utf-8")
    local_ablation = ablation_report(ablations)
    paper_ablation_text = paper_ablation_discussion(paper_ablation)
    (out_dir / "ablation_analysis.md").write_text(local_ablation + "\n" + paper_ablation_text, encoding="utf-8")

    plot_files = write_tradeoff_plots(results, paper, out_dir)
    write_json(
        out_dir / "export_manifest.json",
        {
            "local_result_rows": len(results),
            "literature_rows": len(paper),
            "published_ablation_rows": len(paper_ablation),
            "published_device_latency_rows": len(paper_device_latency),
            "ablation_rows": len(ablations),
            "plots": plot_files,
            "local_table_status": "measured" if results else "template_only",
            "source": "Official iMoonLab YOLOv13 repository table for MS COCO literature context.",
        },
    )
    print(f"Exported academic tables and ablation report to {out_dir}")
    if plot_files:
        print("Trade-off plots: " + ", ".join(plot_files))
    if not results:
        print("The project-dataset table is a placeholder until benchmark_eval.py is run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
