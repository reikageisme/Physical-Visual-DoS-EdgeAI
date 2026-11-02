# YOLOv13 research pipeline

This project keeps its code, local dataset, model checkpoints, caches, logs, and plots under this directory. The upstream iMoonLab implementation is cloned to yolov13_src/ by the setup script.

## 1. Prepare the dataset

Put the existing, fixed split only in dataset/images/train, dataset/images/val, and dataset/images/test, with YOLO labels under the corresponding dataset/labels/... folders. Edit dataset/data.yaml so names lists every class in the correct class-ID order. Do not move examples between splits for this comparison. Keep the same YAML and split for every baseline run. The pipeline enforces the project-local .venv, sets YOLO_CONFIG_DIR to .config, and writes downloaded dataset snapshots and Hugging Face caches under dataset/.

The original NCKH task split is currently audited as empty. A separately reported, supplementary patch-localization benchmark can be prepared from the licensed AdvT-shirt-1K source with `audit_adv_tshirt_patch_labels.py` and `prepare_adv_tshirt_patch_dataset.py`; it writes a deterministic 70/15/15 derived split and `dataset/adv_tshirt_patch_detection.yaml`. The split is a proxy for physical adversarial patch localization and must not be described as the original task dataset.

## 2. Install the local environment

Run setup_env.bat from this folder. It creates .venv, points pip/PyTorch/Ultralytics/Matplotlib/temp caches into .cache/, installs the PyTorch wheel from the CUDA index in YOLOV13_TORCH_INDEX_URL (default: cu128), installs the project dependencies, then installs the checked-out iMoonLab source in editable mode.

If the selected CUDA wheel is not suitable for the GPU driver, set YOLOV13_TORCH_INDEX_URL before running setup. Use https://download.pytorch.org/whl/cpu for CPU-only installation. The official iMoonLab README gives a FlashAttention wheel for Linux, Python 3.11, CUDA 11, and PyTorch 2.2; that wheel is not compatible with this Windows environment. Upstream code falls back to PyTorch scaled dot-product attention when flash_attn is absent, so FlashAttention is an optional acceleration, not a missing required module.

The environment should be Python 3.10 or newer. The setup script records the exact YOLOv13 source revision in results/source_revision.txt.

## 3. Train YOLOv13

Activate the project venv before invoking any script:

    call .venv\Scripts\activate.bat
    python train_yolov13.py --scale n

Use --scale s, --scale l, or --scale x to select another released model size. The script downloads the official pretrained checkpoint into weights/ if needed. To train from a randomly initialized model, pass --scratch. Edit configs/experiment.yaml once and keep those training/augmentation settings fixed across all models in a fair comparison.

Training artifacts, including best.pt, last.pt, results.csv, loss plots, and confusion matrices, are written under runs/ and results/. The script evaluates both validation and test splits after training.

## 4. Fine-tune and evaluate baselines

Activate the project venv and fine-tune YOLOv8/10/11/12 at the same scale and with the shared experiment settings. Repeat for each desired scale (the benchmark config is prepared for n/s/l/x):

    python train_baselines.py --scale n
    python train_baselines.py --scale s
    python train_baselines.py --scale l
    python train_baselines.py --scale x

This writes each best checkpoint into runs/train/<model>/weights/best.pt. Use the same dataset split, image size, epochs, augmentation, and seed for every family and scale.

## 5. Evaluate baseline and YOLOv13 checkpoints

The benchmark config lists YOLOv8/10/11/12/13 checkpoints at all four scales n/s/l/x. Train the corresponding YOLOv13 and baseline models first; then run:

    python benchmark_eval.py --splits val test

The evaluator records precision, recall, mAP50, mAP50:95, parameter count, FLOPs, per-image inference latency, and FPS for CPU and, when available, CUDA. Validation plots are stored in results/benchmark/; the CSV is results/benchmark_eval.csv.

## 6. Export tables, plots, and ablation summary

After evaluation, run:

    python export_results.py

This writes Markdown and LaTeX tables, trade-off plots, and an ablation report under results/. The paper-reference tables include the official iMoonLab MS COCO comparison and ablations, labelled separately from measurements on this project's dataset. Local result cells remain -- until an evaluation has been run; do not replace them with literature values.

Train the variants listed in configs/ablation_template.csv with train_ablation.py. The runner records measured validation results in results/ablation.csv and test results in per-run metadata; rerun the exporter to calculate deltas. No module-level causal attribution is generated from missing measurements.

The project includes a controlled ablation runner. First smoke-check the ablation forward paths, then train the S-scale reference and each one-component variant with the same experiment config:

    python verify_ablation_models.py --scale n
    python train_ablation.py --scale s --variant full
    python train_ablation.py --scale s --variant no-hyperace
    python train_ablation.py --scale s --variant no-fullpad
    python train_ablation.py --scale s --variant no-dsc3k2
    python export_results.py

The ablation runner appends validation P/R/mAP, parameters, FLOPs, inference latency, device, and seed to results/ablation.csv; it also evaluates the test split and records those metrics in each run's metadata. The DS-C3k2 replacement converts each pretrained depthwise-plus-pointwise kernel to its algebraically equivalent dense convolution before fine-tuning, avoiding random initialization of those layers.

## Folder map

- train_yolov13.py: pretrained/fine-tune or scratch training, plus post-train validation and test evaluation.
- train_baselines.py: same-config fine-tuning for YOLOv8/10/11/12.
- train_ablation.py and ablation_utils.py: one-component YOLOv13 ablation training and architecture switches.
- verify_ablation_models.py: finite-forward smoke check for all ablation variants.
- benchmark_eval.py: same-checkpoint validation/test and CPU/GPU timing.
- export_results.py: result tables, plots, and ablation deltas.
- pipeline_utils.py: project-path guards, local cache configuration, and shared utilities.
- configs/: reproducible train/evaluation settings and literature/ablation templates.
- data/: project dataset and split YAML.
- weights/: local model checkpoints.
- runs/, results/: model artifacts and reports.
- yolov13_src/: upstream iMoonLab implementation.
- .venv/, .cache/: isolated runtime and project-local caches.
