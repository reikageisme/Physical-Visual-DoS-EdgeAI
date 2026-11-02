"""Reproducible, protocol-separated graphics for the updated YOLO study.

Run exclusively with .venv/Scripts/python.exe. Published COCO numbers are
context panels, never mixed with the local one-class holdout as a ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from pipeline_utils import PROJECT_ROOT, configure_project_environment, project_path, require_local_venv

configure_project_environment()

MODELS = ("yolo11n", "yolov12n", "yolov13n", "yolov13s")
SEEDS = (42, 100, 2026, 777, 999)
NAMES = {"yolo11n": "YOLO11n", "yolov12n": "YOLOv12n", "yolov13n": "YOLOv13n", "yolov13s": "YOLOv13s"}
COLORS = {"yolo11n": "#456A8C", "yolov12n": "#6D8395", "yolov13n": "#36958E", "yolov13s": "#CB7752", "optimized": "#9A425F", "control": "#50505A", "zero": "#156F47", "native": "#82712E"}
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
LATENCY_PROTOCOL = "actual-batch1-txt-loader-v2"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize(values: list[float]) -> dict:
    clean = [v for value in values if (v := finite(value)) is not None]
    if not clean:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    n = len(clean)
    mean = statistics.mean(clean)
    sd = statistics.stdev(clean) if n > 1 else None
    half = T95.get(n, 1.96) * sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half] if half is not None else None}


def verified_latency(record: dict) -> bool:
    """Configured batch=1 alone does not establish actual image/forward batches."""
    loader = record.get("loader", {})
    batches = record.get("network_forward_batch_size_counts", {})
    return (record.get("protocol_id") == LATENCY_PROTOCOL and
            record.get("imgsz") == 416 and record.get("batch") == 1 and
            record.get("actual_batch_size") == 1 and record.get("timing_precision") == "fp32" and
            record.get("warmup_frames") == 32 and record.get("measured_frames") == 500 and
            record.get("actual_prediction_batches") == 532 and
            loader.get("class_name") == "LoadImagesAndVideos" and loader.get("actual_batch_size") == 1 and
            bool(batches) and set(map(str, batches)) == {"1"})


def load_local(files: list[Path], exclusions: list[str]) -> list[dict]:
    lat_path = project_path("results/fullscale_experiments/latency_records.json", must_exist=True)
    files.append(lat_path)
    latency = read_json(lat_path)
    corrected_path = project_path("results/optimization/historical_latency_retimes.json")
    corrected = {}
    if corrected_path.exists():
        files.append(corrected_path)
        candidate = read_json(corrected_path)
        if candidate.get("status") == "complete" and candidate.get("protocol_id") == LATENCY_PROTOCOL:
            corrected = candidate
        else:
            exclusions.append("Historical corrected retiming incomplete/unverified; prior latency/FPS remains excluded.")
    output = []
    for model in MODELS:
        budget = latency["sweet_spots"][model]["sweet_spot_minutes"]
        selected = []
        for seed in SEEDS:
            record_path = project_path(f"results/fullscale_experiments/records/{model}/seed-{seed}.json", must_exist=True)
            files.append(record_path)
            record = read_json(record_path)
            row = next(r for r in record["milestones"] if r["budget_minutes"] == budget)
            if record["status"] != "complete" or row["status"] != "complete":
                raise ValueError(f"Incomplete baseline {model} seed {seed}")
            selected.append(row)
        speed = [row for row in corrected.get("records", []) if row.get("model_key", row.get("model")) == model]
        checkpoint_by_seed = {seed: row["checkpoint"].replace("\\", "/").lower() for seed, row in zip(SEEDS, selected)}
        retiming_verified = (len(speed) == 5 and {row.get("seed") for row in speed} == set(SEEDS) and
                            all(verified_latency(row.get("latency", {})) and
                                str(row.get("checkpoint", "")).replace("\\", "/").lower() == checkpoint_by_seed.get(row.get("seed"))
                                for row in speed))
        # Technically corrected batches did not stabilize cross-cohort timing:
        # the identical seed100 control checkpoint measured18.973vs32.142ms.
        # Keep every legacy speed out of the direct scientific comparison.
        compatible = False
        exclusions.append(f"{model}: historical speed excluded despite any valid batch1 retiming; runtime-context instability prevents common-cohort comparison.")
        output.append({
            "key": model, "label": NAMES[model], "kind": "original_multiseed",
            "budget_minutes": budget, "seeds": list(SEEDS),
            "map50_pct": summarize([r["test"]["map50"] * 100 for r in selected]),
            "map95_pct": summarize([r["test"]["map50_95"] * 100 for r in selected]),
            "parameters_m": summarize([r["parameters_m"] for r in selected]),
            "latency_ms": summarize([row["latency"]["latency_median_ms_per_frame"] for row in speed]) if compatible else summarize([]),
            "fps": summarize([row["latency"]["fps_from_median_latency"] for row in speed]) if compatible else summarize([]),
            "latency_exclusion": "Original list-loader speed invalid; independent corrected historical retimings excluded for runtime-context instability (identical control checkpoint18.973vs32.142ms).",
            "corrected_historical_retiming_verified_but_excluded": retiming_verified,
            "accuracy_evidence": "Prior fixed-holdout test measurements from validation-selected historical sweet-spot checkpoints, n=5; not the new n=3 confirmation.",
            "latency_evidence": None,
            "latency_records": [row["latency"] for row in speed] if compatible else [],
            "checkpoints": [r["checkpoint"] for r in selected],
            "protocol": {"dataset": "COCO sample + AdvT-shirt local person holdout", "classes": 1, "imgsz": 416,
                         "latency_status": "corrected_retiming" if compatible else "invalid_excluded",
                         "latency_protocol": LATENCY_PROTOCOL if compatible else "historical list-loader; no valid batch1 speed estimate"},
        })
    return output


def group_stat(value: dict, scale: float = 1.0) -> dict:
    """Read algorithm agent's precomputed seed-level summary without fabrication."""
    ci = value.get("ci95")
    if ci is None and value.get("ci95_low") is not None and value.get("ci95_high") is not None:
        ci = [value["ci95_low"], value["ci95_high"]]
    return {"n": value.get("n", 0), "mean": finite(value.get("mean")) * scale if finite(value.get("mean")) is not None else None,
            "sd": finite(value.get("sd")) * scale if finite(value.get("sd")) is not None else None,
            "ci95": [float(v) * scale for v in ci] if ci else None}


def load_test_diagnostics(summary: dict, exclusions: list[str]) -> list[dict]:
    """Use actual post-lock test measurements; never substitute validation mAP.

    These deterministic checkpoint observations do not form independent seed
    groups, even when their inference benchmark measures many image frames.
    """
    rows = []
    for field, key, label in (
        ("zero_locked_test_diagnostic", "zero", "13s restored / 0ep"),
        ("pretrained_locked_test_diagnostic", "native", "13s native 80c / 0ep"),
    ):
        record = summary.get(field)
        if not record:
            continue
        test = record.get("test", {})
        if record.get("status") != "complete" or finite(test.get("map50_95")) is None:
            exclusions.append(f"{field} has no completed test observation; omitted.")
            continue
        speed = record.get("latency", {})
        compatible = verified_latency(speed)
        if not compatible:
            exclusions.append(f"{field} latency protocol unverified; speed/scatter omitted.")
        rows.append({
            "key": key, "label": label, "plot_label": label,
            "kind": "zero_finetuning_test_diagnostic", "n": 1, "seeds": [],
            "map50_pct": summarize([v * 100] if (v := finite(test.get("map50"))) is not None else []),
            "map95_pct": summarize([float(test["map50_95"]) * 100]),
            "parameters_m": summarize([test.get("parameters_m")]),
            "latency_ms": summarize([speed.get("latency_median_ms_per_frame")]) if compatible else summarize([]),
            "fps": summarize([speed.get("fps_from_median_latency")]) if compatible else summarize([]),
            "checkpoints": [record.get("checkpoint")], "protocol": speed,
            "head_classes": record.get("head_classes"), "prediction_classes": record.get("prediction_classes"),
            "interpretation": record.get("interpretation"),
            "source_field": field, "uncertainty": "single checkpoint observation; no seed SD or CI",
        })
    return rows


def load_optimized(path: Path, files: list[Path], exclusions: list[str]) -> list[dict]:
    if not path.exists():
        exclusions.append("Optimization summary not available; figures provisional.")
        return []
    files.append(path)
    summary = read_json(path)
    if summary.get("status") != "complete":
        exclusions.append("Optimization not complete; no partial trial represented as a winner.")
        return []
    if summary.get("groups"):
        rows = []
        for index, group in enumerate(summary["groups"]):
            records = group.get("seed_records", [])
            metrics = group.get("metrics", {})
            protocol = summary.get("protocol", {})
            speed_records = [r.get("latency", {}) for r in records]
            compatible = bool(speed_records) and all(verified_latency(r) for r in speed_records)
            if not compatible:
                exclusions.append(f"Confirmation group {group['name']} speed protocol unverified; FPS/scatter omitted.")
            key = "control" if ("control" in group.get("name", "").lower() or "untuned" in group.get("name", "").lower()) else "optimized"
            rows.append({
                "key": key, "label": "YOLOv13s control (6ep)" if key == "control" else "YOLOv13s optimized (6ep)",
                "plot_label": "13s control / 6ep" if key == "control" else "13s fix+tune / 6ep",
                "source_group_name": group["name"], "kind": "matched_confirmation",
                "seeds": [r.get("seed") for r in records], "configuration": group.get("configuration"),
                "map50_pct": group_stat(metrics.get("map50", {}), 100), "map95_pct": group_stat(metrics.get("map50_95", {}), 100),
                "parameters_m": group_stat(metrics.get("parameters_m", {})),
                "latency_ms": group_stat(metrics.get("latency_ms", {})) if compatible else summarize([]),
                "fps": group_stat(metrics.get("fps", {})) if compatible else summarize([]),
                "checkpoints": [r.get("checkpoint") for r in records], "protocol": protocol,
                "latency_records": speed_records,
            })
        return rows + load_test_diagnostics(summary, exclusions)
    winner = summary.get("winner", summary.get("selected", summary))
    test = winner.get("test", {})
    if finite(test.get("map50_95")) is None:
        exclusions.append("Optimized winner has no completed test mAP95; omitted.")
        return []
    speed = winner.get("latency", {})
    compatible = verified_latency(speed)
    latency_ms = finite(speed.get("latency_median_ms_per_frame")) if compatible else None
    fps = finite(speed.get("fps_from_median_latency")) if compatible else None
    if not compatible:
        exclusions.append("Optimized latency lacks verified actual-batch1-v2 loader/network evidence; omitted from shared scatter/FPS.")
    return [{
        "key": "optimized", "label": winner.get("name", winner.get("model_name", "YOLOv13 optimized")),
        "kind": "optimized_single_run", "seeds": [winner.get("seed")],
        "map50_pct": summarize([float(test["map50"]) * 100]) if finite(test.get("map50")) is not None else summarize([]),
        "map95_pct": summarize([float(test["map50_95"]) * 100]),
        "parameters_m": summarize([winner.get("parameters_m")]),
        "latency_ms": summarize([latency_ms]), "fps": summarize([fps]),
        "checkpoints": [winner.get("checkpoint")], "protocol": speed,
        "selection": summary.get("selection", winner.get("selection")),
    }] + load_test_diagnostics(summary, exclusions)


def load_published(path: Path, files: list[Path], exclusions: list[str]) -> list[dict]:
    if not path.exists():
        exclusions.append("Literature matrix not yet available; published context panel provisional.")
        return []
    files.append(path)
    matrix = read_json(path)
    rows = matrix.get("models", matrix.get("rows", matrix.get("benchmarks", []))) if isinstance(matrix, dict) else matrix
    result = []
    for row in rows:
        reported_fps = finite(row.get("published_fps"))
        reported_latency = finite(row.get("published_latency_ms"))
        speed_value = reported_fps if reported_fps is not None else (1000 / reported_latency if reported_latency is not None and reported_latency > 0 else None)
        result.append({
            "label": row.get("label", row.get("model", row.get("name", "Unnamed"))),
            "map95_pct": finite(row.get("map50_95_pct")),
            "parameters_m": finite(row.get("parameters_m")),
            "latency_ms": finite(row.get("published_latency_ms")),
            "protocol": row.get("latency_protocol"),
            "dataset": row.get("dataset"), "classes": row.get("num_classes"),
            "evaluation_protocol": row.get("evaluation_protocol"), "source": row.get("source_url"),
            "model_version": row.get("model_version"), "reference_id": row.get("reference_id"),
            "published_fps": reported_fps, "fps_context": speed_value,
            "fps_context_kind": "reported_FPS" if reported_fps is not None else ("derived_reciprocal_latency" if speed_value is not None else "not_reported"),
            "fps_derivation": "1000 / published_latency_ms" if reported_fps is None and speed_value is not None else None,
        })
        if "YOLOv12" in result[-1]["label"]:
            result[-1]["plot_label"] = result[-1]["label"] + " v1"
        else:
            result[-1]["plot_label"] = result[-1]["label"]
    return result


def ci_half(row: dict, metric: str) -> float:
    ci = row[metric].get("ci95")
    return (ci[1] - ci[0]) / 2 if ci else 0


def style_ax(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DFE4E8", linewidth=0.65)
    ax.tick_params(labelsize=8.5)


def local_bars(ax, rows: list[dict], metric: str, title: str, xlabel: str) -> None:
    y = list(range(len(rows)))
    vals = [r[metric]["mean"] for r in rows]
    for index, (row, value) in enumerate(zip(rows, vals)):
        if value is None:
            ax.text(0, index, "N/A", va="center", fontsize=8.5, color="#66747F")
            continue
        ax.barh(index, value, height=.58, color=COLORS[row["key"]], alpha=.9,
                xerr=ci_half(row, metric), error_kw={"capsize": 2, "elinewidth": .8, "ecolor": "#283642"})
        ax.text(value + ci_half(row, metric), index, f" {value:.2f}", fontsize=8.2, va="center")
    ax.set_yticks(y, [r.get("plot_label", r["label"]) + (" ‡" if r["kind"] == "zero_finetuning_test_diagnostic" else " *" if r["kind"] == "optimized_single_run" else "") for r in rows])
    ax.set_ylim(len(rows) - .5, -.5)
    maximum = max((v + ci_half(r, metric) for r, v in zip(rows, vals) if v is not None), default=1)
    ax.set_xlim(0, max(maximum, .01) * 1.25)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=8.8)
    style_ax(ax)


def published_bars(ax, rows: list[dict], metric: str, title: str, xlabel: str) -> None:
    if not rows:
        ax.axis("off")
        ax.text(.5, .5, "Published reference data pending\nNo substituted or inferred values", ha="center", va="center", transform=ax.transAxes, color="#66747F")
        return
    for index, row in enumerate(rows):
        value = row[metric]
        if value is None:
            ax.text(0, index, "N/A", va="center", fontsize=8, color="#66747F")
        else:
            ax.barh(index, value, height=.58, color="#A4ADB5")
            derivative_marker = "†" if metric == "fps_context" and row["fps_context_kind"] == "derived_reciprocal_latency" else ""
            value_label = f"{value:.0f}" if metric == "fps_context" else f"{value:.1f}"
            ax.text(value, index, f" {value_label}{derivative_marker}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r.get("plot_label", r["label"]) for r in rows])
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xlim(0, max((r[metric] for r in rows if r[metric] is not None), default=1) * 1.22)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=8.8)
    style_ax(ax)


def save_figure(fig, stem: str, out: Path) -> list[str]:
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", metadata={"Creator": "project_yolov13 plotting script"} if suffix == "pdf" else None)
        paths.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return paths


def comparison_figure(local, published, out, provisional):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.6), gridspec_kw={"hspace": .66, "wspace": .20})
    fig.suptitle("Local results and published benchmark context", x=.035, ha="left", y=.985, fontsize=13, fontweight="bold")
    fig.text(.035, .925, "A. Local person test holdout | speed shown only for corrected batch1 protocol", fontsize=9.5, fontweight="bold")
    for ax, metric, title, xlabel in zip(axes[0], ("map95_pct", "fps", "parameters_m"),
                                        ("Detection accuracy", "FPS from pipeline latency", "Model size"),
                                        ("Test mAP@0.5:0.95 (%)", "Frames/s", "Parameters (M)")):
        local_bars(ax, local, metric, title, xlabel)
    fig.text(.035, .535, "B. Published COCO references | 80 classes / 640px | non-comparable context", fontsize=9.5, fontweight="bold")
    for ax, metric, title, xlabel in zip(axes[1], ("map95_pct", "fps_context", "parameters_m"),
                                        ("Published accuracy", "Source-specific FPS context", "Published model size"),
                                        ("COCO mAP@0.5:0.95 (%)", "FPS (reported / derived†)", "Parameters (M)")):
        published_bars(ax, published, metric, title, xlabel)
    for ax in axes[:, 1:].flat:
        ax.tick_params(axis="y", labelleft=False, length=0)
    fig.subplots_adjust(left=.17, right=.985, bottom=.235, top=.855)
    fig.text(.035, .023, "Prior accuracy: n=5 at 120/120/5/10-min sweet spots; new matched arms: 6 epochs, n=3; 95% t CI.\nPrior FPS omitted: old loader invalid; corrected retiming unstable (same checkpoint: 18.97 vs 32.14 ms).\n‡ Zero-finetuning test diagnostics: n=1, no seed SD/CI; native 80c predictions filtered to person.\nFix+tune = restoration + recipe. RTX5060 / 416px / FP32 / batch1 / pipeline, no loading or I/O.\nPublished † = 1000 / reported latency; unmarked FPS reported directly, not a common throughput measurement.\nPublished hardware/backend/dtype differ; no common speed ranking. Full protocols and sources: manifest.", fontsize=7.3, color="#42515E", linespacing=1.25)
    if provisional:
        fig.text(.985, .975, "PROVISIONAL", ha="right", color="#9A425F", fontsize=8.5, fontweight="bold")
    paths = save_figure(fig, "comparison_baseline_vs_proposed", out)
    plt.close(fig)
    return paths


def scatter_figure(local, out, provisional):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    rows = [r for r in local if r["latency_ms"]["mean"] is not None]
    fig, ax = plt.subplots(figsize=(9.2, 6.1))
    if not rows:
        ax.axis("off")
        ax.text(.5, .55, "Corrected batch1 latency measurements pending", ha="center", va="center", transform=ax.transAxes, fontsize=13)
        ax.text(.5, .43, "Historical list-loader timing is invalid and excluded.\nNo latency points are inferred from accuracy or configured batch size.", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#42515E")
        fig.text(.97, .975, "PROVISIONAL", ha="right", fontsize=8.5, color="#9A425F", fontweight="bold")
        paths = save_figure(fig, "tradeoff_map_vs_latency", out)
        plt.close(fig)
        return paths
    for index, row in enumerate(rows):
        x, y = row["latency_ms"]["mean"], row["map95_pct"]["mean"]
        optimized = row["key"] == "optimized"
        marker = {"optimized": "*", "control": "s", "zero": "D", "native": "^"}.get(row["key"], "o")
        ax.errorbar(x, y, xerr=ci_half(row, "latency_ms"), yerr=ci_half(row, "map95_pct"),
                    fmt=marker, markersize=12 if optimized else 7,
                    color=COLORS[row["key"]], capsize=4, elinewidth=1.3, zorder=5)
        single = row["kind"] in ("optimized_single_run", "zero_finetuning_test_diagnostic")
        label = row.get("plot_label", row["label"]) + (" (n=1; no CI)" if single else " (prior n=5)" if row["kind"] == "original_multiseed" else f" (n={row['map95_pct']['n']})")
        offsets = {"yolo11n": (10, 12), "yolov12n": (-4, -30), "yolov13n": (10, 14),
                   "yolov13s": (-145, -28), "control": (-165, -42), "optimized": (-170, 25),
                   "zero": (-185, 25), "native": (-195, -30)}
        ax.annotate(label, (x, y), xytext=offsets[row["key"]], textcoords="offset points", fontsize=9,
                    color=COLORS[row["key"]], arrowprops={"arrowstyle": "-", "color": COLORS[row["key"]], "lw": .6})
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 20, 33.33, 50, 80], ["2", "3", "5", "10", "20", "33.3", "50", "80"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(1.9, max(80, max(r["latency_ms"]["mean"] for r in rows) * 1.35))
    ax.set_ylim(min(r["map95_pct"]["mean"] - ci_half(r, "map95_pct") for r in rows) - 2.4,
                max(r["map95_pct"]["mean"] + ci_half(r, "map95_pct") for r in rows) + 3)
    ax.axvline(1000 / 30, color="#858B91", linestyle="--", linewidth=1)
    ax.text(1000 / 30, .035, "30 FPS timing guide", rotation=90, va="bottom", ha="right",
            transform=ax.get_xaxis_transform(), fontsize=9, color="#687078")
    ax.set_xlabel("Mean per-seed median pipeline latency (ms/frame; logarithmic axis)", fontsize=10)
    ax.set_ylabel("Test mAP@0.5:0.95 (%)", fontsize=10)
    ax.set_title("Accuracy–latency trade-off: local common protocol", loc="left", fontsize=12, fontweight="bold", pad=15)
    style_ax(ax)
    ax.grid(axis="y", color="#DFE4E8", linewidth=.65)
    fig.subplots_adjust(left=.1, right=.97, top=.89, bottom=.29)
    fig.text(.1, .025, "RTX5060 / 416px / FP32 / actual batch1; preprocessing + inference + postprocessing, no loading or I/O.\nPrimary matched 6-epoch arms: n=3 with 95% t CI; corrected loader/network batches verified.\nSecondary zero-finetuning diagnostics: n=1 without seed SD/CI; native 80c filtered to person.\nHistorical timing excluded: old loader invalid; corrected cohorts unstable (same checkpoint: 18.97 vs 32.14 ms).\nNo speedup inferred; timing summaries describe this session. Fix+tune combines restoration + recipe.\nCombined interventions and n=1 diagnostics do not isolate an architecture effect.", fontsize=7.7, color="#42515E", linespacing=1.3)
    if provisional:
        fig.text(.97, .975, "PROVISIONAL", ha="right", fontsize=8.5, color="#9A425F", fontweight="bold")
    paths = save_figure(fig, "tradeoff_map_vs_latency", out)
    plt.close(fig)
    return paths


def main():
    require_local_venv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization-summary", default="results/optimization/final_summary.json")
    parser.add_argument("--literature-matrix", default="results/literature_updated/benchmark_matrix.json")
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
    files, exclusions = [], ["All prior fullscale latency/FPS excluded: original list loader invalid; independently corrected historical retimings also excluded for runtime-context instability (identical checkpoint18.973vs32.142ms). No speedup claim."]
    local = load_local(files, exclusions)
    summary_path = project_path(args.optimization_summary)
    winner = load_optimized(summary_path, files, exclusions)
    published = load_published(project_path(args.literature_matrix), files, exclusions)
    audit_path = project_path("results/updated_research/metric_integrity_audit.json")
    audit_verified = False
    if audit_path.exists() and summary_path.exists():
        files.append(audit_path)
        audit = read_json(audit_path)
        audit_verified = (audit.get("status") == "passed" and
                          audit.get("final_summary_sha256") == hashlib.sha256(summary_path.read_bytes()).hexdigest())
    if not audit_verified:
        exclusions.append("Root metric-integrity audit missing, failed, or stale; figures remain provisional.")
    matched = [row for row in winner if row["kind"] == "matched_confirmation"]
    timing_ready = (len(matched) == 2 and all(row["latency_ms"]["n"] == 3 and row["fps"]["n"] == 3 and
                                             row["latency_ms"]["mean"] is not None for row in matched))
    if not timing_ready:
        exclusions.append("Both n=3 matched groups require corrected actual-batch1 timing before final figures.")
    provisional = not winner or not published or not audit_verified or not timing_ready
    if args.require_final and provisional:
        raise RuntimeError("Final plots require completed summary, literature matrix and matching passed integrity audit: " + "; ".join(exclusions))
    if winner:
        local.extend(winner)
    out = project_path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    artifacts = comparison_figure(local, published, out, provisional) + scatter_figure(local, out, provisional)
    from PIL import Image
    artifact_records = []
    for artifact in artifacts:
        details = {"path": artifact, "sha256": hashlib.sha256(project_path(artifact).read_bytes()).hexdigest()}
        if artifact.endswith(".png"):
            with Image.open(project_path(artifact)) as raster:
                details.update({"size_pixels": list(raster.size), "dpi_metadata": list(raster.info.get("dpi", [])),
                                "native_width_inches": raster.width / 300, "native_height_inches": raster.height / 300,
                                "recommended_word_width_inches": 9.2})
        artifact_records.append(details)
    manifest = {
        "status": "provisional" if provisional else "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dpi": 300, "local_operating_points": local, "published_context": published, "exclusions": exclusions,
        "input_sha256": {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "artifacts": artifact_records,
        "captions": {
            "comparison_baseline_vs_proposed.png": "Local person-detection operating points and published COCO references in separate non-comparable panels. Accuracy uncertainty is a 95% Student t interval: prior fixed-holdout models n=5, new matched six-epoch arms n=3. All prior-model FPS is N/A: original list-loader timings were invalid; corrected historical retimings are technically batch1 but unstable across runtime cohorts (identical checkpoint18.973versus32.142ms). Current FPS requires corrected loader/network batch1 proof and is descriptive of this session, without a speedup claim. Zero-finetuning diagnostics marked double dagger are n=1 without seed SD/CI; native80 predictions are person-filtered. Fix+tune combines restoration and recipe. Published FPS is source-reported or dagger-marked reciprocal reported latency; differing dataset/classes/hardware/backend/dtype/timing scope prevent direct comparisons. Validation-only diagnostics excluded.",
            "tradeoff_map_vs_latency.png": "Corrected local accuracy–latency at416px, FP32, verified actual batch1 onRTX5060. Primary comparison comprises matched six-epoch control/proposed groups, n=3 with95% Student t intervals. Secondary zero-finetuning test diagnostics are n=1 without seed SD/CI; native80 predictions are person-filtered. All historical-model timing points omitted: original list-loader timings invalid, corrected historical cohorts unstable (samecheckpoint18.973versus32.142ms). No speedup claim; latency is descriptive of this runtime session and excludes loading/decoding/I/O. FPS is reciprocal median pipeline latency. Restoration+tuning does not isolate architecture effects. Logarithmic latency axis and illustrative33.3ms guide do not establish real-time readiness.",
        },
    }
    (out / "figures_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "outputs": artifacts, "exclusions": exclusions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
