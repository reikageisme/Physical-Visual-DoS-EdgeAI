# Sponge Patch — Physical Visual DoS on Edge-AI Systems

> **Threat Model:** Gray-box (pre-NMS score access)  
> **Attack:** Genetic Algorithm + Saliency-guided Patch Optimization  
> **Target:** YOLOv8n on Edge Server (simulated / physical)

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full simulation (no camera needed)
```bash
# Quick demo (~2 minutes)
python simulate_edge_server.py --quick

# Full experiment: train GA patch + test both scenarios
python simulate_edge_server.py --frames 200 --train --pop 20 --gen 30

# Simulate Raspberry Pi constraints
python simulate_edge_server.py --frames 200 --train --edge-profile raspberry_pi

# Use a pre-trained patch
python simulate_edge_server.py --frames 200 --patch outputs/sponge_patch.png
```

### 3. Run individual experiments
```bash
# Multi-seed statistical evaluation (10 seeds, mean ± std)
python experiments/multi_seed_experiment.py --n-seeds 10 --pop 20 --gen 30

# Ablation: patch size 1% to 16%
python experiments/ablation_patch_size.py

# Baseline comparison (random / checkerboard / sponge)
python experiments/baseline_comparison.py --patch outputs/sponge_patch.png

# Defense evaluation (conf threshold / max_det cap)
python experiments/defense_evaluation.py --patch outputs/sponge_patch.png
```

### 4. Train only (save patch for later)
```bash
# Train GA patch (64×64px, ~4% of 320×320 frame)
python main_train.py --size 64 --pop 30 --gen 50 --seed 42

# Train with area percentage
python main_train.py --area-pct 4.0 --resolution 320 --seed 42

# Train with different crossover
python main_train.py --size 64 --pop 30 --gen 50 --crossover uniform
```

### 5. Test with real camera
```bash
# Clean baseline
python test_physical_dos.py --scenario clean

# Digital attack overlay
python test_physical_dos.py --patch outputs/sponge_patch.png --scenario digital_attack

# Headless simulation (no camera)
python test_physical_dos.py --simulate --frames 200 --scenario digital_attack
```

### 6. Plot results
```bash
# Plot latest log
python utils/plot_results.py

# Plot with latency breakdown
python utils/plot_results.py --file logs/resource_log_xxx.csv --breakdown

# Multi-seed convergence plot
python utils/plot_results.py --multi-seed outputs/multi_seed/
```

---

## Edge Hardware Profiles

| Profile | Threads | GPU | Simulates |
|---|---|---|---|
| `raspberry_pi` | 1 | ❌ | Raspberry Pi 4 |
| `intel_nuc` | 2 | ❌ | Intel NUC |
| `jetson_nano` | 2 | ✅ | Jetson Nano |
| `full_server` | all | ✅ | Full Edge Server |

```bash
python simulate_edge_server.py --edge-profile raspberry_pi --frames 200 --train
```

---

## Project Structure

```
Physical-Visual-DoS-EdgeAI/
├── simulate_edge_server.py       # ← MAIN: Full pipeline simulation
├── main_train.py                 # Patch training (GA only)
├── fast_train.py                 # Patch training (PGD gradient)
├── test_physical_dos.py          # Camera test / headless simulation
│
├── core/
│   ├── victim_model.py           # YOLOv8 wrapper (gray-box + NMS profiling)
│   ├── sponge_fitness.py         # Fitness: GrayBox + ObservableFitness
│   └── eot_transforms.py         # EOT augmentations (Kornia)
│
├── attack/
│   └── genetic_algo.py           # Saliency-Guided GA (seed + convergence)
│
├── experiments/
│   ├── multi_seed_experiment.py  # Statistical rigor (Major Issue 5)
│   ├── ablation_patch_size.py    # Patch size ablation (Major Issue 4)
│   ├── baseline_comparison.py    # Baseline controls (Major Issue 9)
│   └── defense_evaluation.py    # Defense eval (Major Issue 10)
│
├── utils/
│   ├── monitor.py                # Resource monitor + latency breakdown
│   └── plot_results.py           # Performance / latency / multi-seed plots
│
├── docs/
│   ├── REVIEWED.md               # Detailed peer review
│   └── BaoCaoKhoaHocAdversarialAttacks2.docx
│
└── outputs/                      # Generated patches + run summaries
```

---

## Threat Model Clarification

| Mode | Access | Used by |
|---|---|---|
| **Gray-box** | Pre-NMS raw confidence scores (local model copy) | `main_train.py`, GA |
| **Observable Black-box** | Post-NMS detections + latency only | `ObservableFitness` in `sponge_fitness.py` |
| **White-box** | Full gradient access | `fast_train.py` (PGD) |

---

## Outputs

After running the simulation:

```
outputs/
├── sponge_patch_*.png            # Optimized patch (small + A4 print)
├── sponge_patch_A4_*.png         # A4 300DPI print version
├── ga_summary_*.json             # Per-run GA statistics
├── plot_clean.png                # Clean baseline performance
├── plot_attack.png               # Attack scenario performance
├── latency_clean.png             # NMS/Forward/Preproc breakdown (clean)
├── latency_attack.png            # NMS/Forward/Preproc breakdown (attack)
└── scenario_comparison.png       # Side-by-side scenario comparison

logs/
├── resource_log_clean_*.csv      # Frame-by-frame: FPS/CPU/RAM/NMS_ms
└── resource_log_digital_attack_*.csv
```