"""Publication plots from saved measurements; does not change experiment data.

The primary raw key ``checkerboard`` actually denotes diagonal stripes. This
plotter fixes the displayed label and retains provenance of the original run.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = ROOT / "outputs/local_q1_20261002_v2"
    assert "finished_utc" in read(run / "manifest.json")
    data = read(run / "aggregate.json")
    out = run / "figures_corrected"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ns = sorted(map(int, data["nms"]))
    ax.errorbar(ns, [data["nms"][str(n)]["mean"] for n in ns],
                yerr=[data["nms"][str(n)]["std"] for n in ns], fmt="o-", capsize=3)
    ax.set(xlabel="Số hộp tổng hợp đưa vào NMS", ylabel="Độ trễ NMS (ms)")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "nms_local.png", dpi=240)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4))
    labels = {"sg_ga": "GA vị trí nổi bật", "ga_center": "GA vị trí trung tâm", "random_search": "Tìm kiếm ngẫu nhiên"}
    for method, label in labels.items():
        a = np.array([read(run / f"seed_{seed}_{method}.json")["running_best"] for seed in range(10)])
        queries = np.arange(1, 21) * 20
        mean, sd = a.mean(0), a.std(0, ddof=1)
        ax.plot(queries, mean, label=label)
        ax.fill_between(queries, mean-sd, mean+sd, alpha=.15)
    ax.set(xlabel="Số truy vấn ảnh (cùng ngân sách)", ylabel="Fitness EOT tốt nhất đã quan sát")
    ax.legend(fontsize=10)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "convergence_local.png", dpi=240)
    plt.close(fig)
    conditions = ["clean", "noise", "checkerboard", "solid", "sg_ga", "ga_center", "random_search"]
    names = ["Sạch", "Nhiễu", "Sọc chéo", "Màu đặc", "GA nổi bật", "GA trung tâm", "Tìm ngẫu nhiên"]
    for image in ["bus", "zidane"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
        for ax, metric, ylabel in zip(axes, ["active_candidates", "nms_ms"],
                                      ["Ứng viên sau ngưỡng", "Độ trễ NMS (ms)"]):
            for tau, offset, label in [(.01, -.18, "Ngưỡng 0,01"), (.25, .18, "Ngưỡng 0,25")]:
                rows = [next(r for r in data["evaluation"] if r["image"] == image and
                             r["condition"] == c and r["threshold"] == tau and r["cap"] is None) for c in conditions]
                ax.bar(np.arange(7)+offset, [r[metric]["mean"] for r in rows], .36,
                       yerr=[r[metric]["std"] for r in rows], label=label, capsize=2)
            ax.set_xticks(range(7), names, rotation=35, ha="right")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=.2)
        axes[0].legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"candidates_latency_{image}.png", dpi=240)
        plt.close(fig)
    dataset = ROOT / "outputs/local_coco128_20261002"
    assert "finished_utc" in read(dataset / "manifest.json")
    records = read(dataset / "aggregate.json")
    dataset_figures = dataset / "figures_corrected"
    dataset_figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    names = ["clean", "noise", "diagonal_stripes", "checkerboard", "solid", "sg_ga", "ga_center", "random_search"]
    captions = ["Sạch", "Nhiễu", "Sọc chéo", "Ô bàn cờ", "Màu đặc", "GA nổi bật", "GA trung tâm", "Tìm ngẫu nhiên"]
    for ax, metric, ylabel in [(axes[0], "recall_iou50", "Độ bao phủ tại IoU 0,5"),
                               (axes[1], "AP50_101point_diagnostic", "AP50 chẩn đoán")]:
        for tau, cap, offset, label in [(.01, None, -.25, "Ngưỡng 0,01"), (.25, None, 0, "Ngưỡng 0,25"), (.01, 100, .25, "0,01 + top-100")]:
            values = [next(r["metrics"][metric] for r in records if r["condition"] == name
                           and r["threshold"] == tau and r["cap"] == cap) for name in names]
            ax.bar(np.arange(8)+offset, values, .25, label=label)
        ax.set_xticks(range(8), captions, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(dataset_figures / "accuracy_defense.png", dpi=240)
    plt.close(fig)
    provenance = {"plotter_sha256": digest(Path(__file__)), "source_aggregate_sha256": digest(run / "aggregate.json"),
                  "source_manifest_sha256": digest(run / "manifest.json"),
                  "label_correction": "primary raw checkerboard key is (row+column)//4 modulo 2, i.e. diagonal stripes",
                  "data_changed": False,
                  "figure_sha256": {p.name: digest(p) for p in sorted(out.glob("*.png"))}}
    (out / "manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_provenance = {"plotter_sha256": digest(Path(__file__)), "source_aggregate_sha256": digest(dataset / "aggregate.json"),
                          "source_manifest_sha256": digest(dataset / "manifest.json"), "data_changed": False,
                          "figure_sha256": digest(dataset_figures / "accuracy_defense.png")}
    (dataset_figures / "manifest.json").write_text(json.dumps(dataset_provenance, indent=2), encoding="utf-8")
    print("Publication plots:", out)


if __name__ == "__main__":
    main()
