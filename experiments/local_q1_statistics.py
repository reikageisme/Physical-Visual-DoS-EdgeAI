"""Exploratory paired bootstrap of saved local experiments; no new inference.

Intervals describe seed variation on two photos or image variation in COCO128.
They do not establish population generalization or confirm a preregistered test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_interval(a, b, rng, draws=10000):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    assert len(delta) >= 2 and np.isfinite(delta).all()
    means = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    return {"n_pairs": len(delta), "mean_difference": float(delta.mean()),
            "sample_std_difference": float(delta.std(ddof=1)),
            "percentile_ci95": np.quantile(means, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-run", default="outputs/local_q1_20261002_v2")
    parser.add_argument("--dataset-run", default="outputs/local_coco128_20261002")
    args = parser.parse_args()
    primary, dataset = ROOT / args.patch_run, ROOT / args.dataset_run
    manifests = [read(path / "manifest.json") for path in [primary, dataset]]
    assert all("finished_utc" in manifest for manifest in manifests), "Both experiments must finish first."
    assert manifests[0]["protocol"]["seeds"] == 10
    rng = np.random.default_rng(20261002)
    results = []
    fitness = {method: [read(primary / f"seed_{seed}_{method}.json")["best_eot_fitness"]
                        for seed in range(10)]
               for method in ["sg_ga", "ga_center", "random_search"]}
    for reference in ["ga_center", "random_search"]:
        results.append({"group": "optimizer", "unit": "seed", "contrast": "sg_ga minus " + reference,
                        "metric": "best_eot_fitness", **paired_interval(fitness["sg_ga"], fitness[reference], rng)})
    rows = [row for seed in range(10) for row in read(primary / f"evaluation_seed_{seed}.json")]
    for image in ["bus", "zidane"]:
        for threshold, cap in [(.01, None), (.25, None), (.01, 100)]:
            block = [r for r in rows if r["image"] == image and r["threshold"] == threshold
                     and r["pre_nms_cap"] == cap]
            clean = sorted([r for r in block if r["condition"] == "clean"], key=lambda r: r["seed"])
            patch = sorted([r for r in block if r["condition"] == "sg_ga"], key=lambda r: r["seed"])
            assert [r["seed"] for r in clean] == [r["seed"] for r in patch] == list(range(10))
            for metric in ["active_candidates", "nms_ms", "total_ms", "throughput_fps"]:
                results.append({"group": "two_images", "unit": "seed", "image": image,
                                "threshold": threshold, "cap": cap, "contrast": "sg_ga minus clean",
                                "metric": metric, **paired_interval([r[metric] for r in patch],
                                                                   [r[metric] for r in clean], rng)})
    rows = [r for path in sorted(dataset.glob("image_*.json")) for r in read(path)]
    for threshold, cap in [(.01, None), (.25, None), (.01, 100)]:
        block = [r for r in rows if r["threshold"] == threshold and r["cap"] == cap]
        clean = sorted([r for r in block if r["condition"] == "clean"], key=lambda r: r["image"])
        patch = sorted([r for r in block if r["condition"] == "sg_ga"], key=lambda r: r["image"])
        assert len(clean) == len(patch) == 128
        assert [r["image"] for r in clean] == [r["image"] for r in patch]
        for metric in ["active_candidates", "nms_ms", "postprocessing_ms"]:
            results.append({"group": "coco128", "unit": "image", "threshold": threshold, "cap": cap,
                            "contrast": "sg_ga minus clean", "metric": metric,
                            **paired_interval([r[metric] for r in patch], [r[metric] for r in clean], rng)})
    output = {"script_sha256": digest(__file__), "bootstrap_seed": 20261002, "draws": 10000,
              "method": "paired percentile bootstrap of mean differences, 2.5 and 97.5 percentiles",
              "manifest_sha256": {str(p.relative_to(ROOT)): digest(p / "manifest.json")
                                  for p in [primary, dataset]},
              "limitations": ["exploratory post-hoc intervals; no confirmatory p-values",
                              "no multiplicity correction across contrasts",
                              "seed intervals conditional on a single optimization photo",
                              "COCO128 image intervals conditional on a fixed seed-0 patch and train subset",
                              "not confidence intervals for AP; AP is computed across all records"],
              "contrasts": results}
    (primary / "paired_bootstrap.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Paired bootstrap contrasts:", len(results))


if __name__ == "__main__":
    main()
