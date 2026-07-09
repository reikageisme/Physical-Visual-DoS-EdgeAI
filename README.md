# Sponge Patch — Physical Visual Denial-of-Service on Edge-AI Systems

<p align="center">
  <a href="https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/README.md">🇺🇸 English</a> | 
  <a href="https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/README.vi.md">🇻🇳 Tiếng Việt</a>
</p>

---

## 📖 Project Overview

**Sponge Patch** is a research framework designed to evaluate and demonstrate **Physical Visual Denial-of-Service (DoS)** attacks against Edge-AI object detection systems, specifically targeting YOLOv8n. Unlike conventional adversarial attacks that aim to reduce model accuracy (e.g., hiding objects or misclassifying them), a Visual DoS attack aims to exhaust the computational resources of the victim model. 

By utilizing a **Saliency-Guided Genetic Algorithm (GA)**, we generate physical adversarial patches that induce a massive number of false positive bounding boxes. When the victim model processes these inputs, the post-processing step—specifically Non-Maximum Suppression (NMS)—is overwhelmed, causing significant latency spikes. On resource-constrained edge devices, this leads to frame drops, system unresponsiveness, and a complete Denial of Service.

### Key Capabilities and Features
- **Saliency-Guided Optimization**: Focuses the Genetic Algorithm on highly salient regions to maximize patch efficiency and visual stealth.
- **Physical-World Robustness**: Employs Expectation Over Transformation (EOT) to ensure the patch remains effective when printed and captured by real-world cameras under varying angles, lighting, and noise.
- **Comprehensive Edge Simulation**: Profiles the attack's impact on various hardware constraints, simulating environments like Raspberry Pi, Intel NUC, and Jetson Nano.
- **Robust Evaluation Suite**: Includes statistical multi-seed experiments, ablation studies, and baseline comparisons (Random, Checkerboard).

---

## 🎯 Threat Model & Attack Scenarios

Understanding the attacker's capabilities is crucial. This project operates primarily under a **Gray-box** threat model for patch generation, while simulating real-world deployment where the attacker only has physical access to the camera's field of view.

| Threat Model Level | Attacker Access | Component / Usage in this Framework |
| :--- | :--- | :--- |
| **Gray-box (Primary)** | Pre-NMS raw confidence scores and bounding box coordinates from a local proxy model. No access to internal gradients. | Used by `main_train.py` and the Genetic Algorithm to optimize the patch without backpropagation. |
| **Observable Black-box** | Post-NMS detections and overall inference latency measurements. | Used by `ObservableFitness` in `sponge_fitness.py` for profiling the actual impact on the edge device. |
| **White-box (Baseline)** | Full access to model architecture and gradients. | Used by `fast_train.py` (PGD) to establish a theoretical upper bound for the attack. |

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI.git
cd Physical-Visual-DoS-EdgeAI
pip install -r requirements.txt
```

### 2. Run the Full Pipeline Simulation
You can simulate the entire attack pipeline (from training the patch to evaluating its impact on NMS latency) without needing a physical camera.

```bash
# Run a quick demonstration (~2 minutes)
python simulate_edge_server.py --quick

# Run a full experiment: trains a GA patch and tests both clean and attack scenarios
python simulate_edge_server.py --frames 200 --train --pop 20 --gen 30

# Simulate specific hardware constraints (e.g., Raspberry Pi)
python simulate_edge_server.py --frames 200 --train --edge-profile raspberry_pi

# Evaluate using a pre-trained patch
python simulate_edge_server.py --frames 200 --patch outputs/sponge_patch.png
```

---

## 🔬 Detailed Experimentation Suite

The framework provides independent modules to rigorously evaluate the attack's effectiveness, robustness, and potential defenses.

### Statistical Rigor & Convergence
To ensure the attack is consistent, we provide a multi-seed evaluation script that calculates the mean and standard deviation across multiple optimization runs.
```bash
python experiments/multi_seed_experiment.py --n-seeds 10 --pop 20 --gen 30
```

### Ablation Studies
Analyze how the physical size of the adversarial patch correlates with the resulting NMS latency spike.
```bash
python experiments/ablation_patch_size.py
```

### Baseline Comparisons
Compare the optimized Sponge Patch against standard control patches (Random noise and Checkerboard patterns) to validate the effectiveness of the Genetic Algorithm.
```bash
python experiments/baseline_comparison.py --patch outputs/sponge_patch.png
```

### Defense Evaluations
Test the patch against standard mitigation techniques, such as raising the confidence threshold or capping the maximum allowed detections (`max_det`).
```bash
python experiments/defense_evaluation.py --patch outputs/sponge_patch.png
```

---

## 🎛️ Training & Real-world Testing

### Standalone Patch Training
If you only want to generate a patch for later use (e.g., to print it out), you can run the standalone training script.

```bash
# Train a 64x64px patch (~4% of a 320x320 frame)
python main_train.py --size 64 --pop 30 --gen 50 --seed 42

# Train using an area percentage relative to the frame
python main_train.py --area-pct 4.0 --resolution 320 --seed 42
```

### Testing with a Physical Camera
Test the generated patch in a real-world scenario using your webcam.

```bash
# Run the clean baseline (no attack)
python test_physical_dos.py --scenario clean

# Run the digital overlay attack (projects the patch onto the camera feed)
python test_physical_dos.py --patch outputs/sponge_patch.png --scenario digital_attack

# Run a headless simulation (bypasses the camera, uses synthetic frames)
python test_physical_dos.py --simulate --frames 200 --scenario digital_attack
```

---

## 📊 Hardware Profiling & Results Visualization

### Edge Hardware Profiles
The framework can artificially throttle resources to simulate deploying YOLOv8n on constrained hardware.

| Profile Name | CPU Threads | GPU Enabled | Target Simulation |
| :--- | :--- | :--- | :--- |
| `raspberry_pi` | 1 | ❌ | Raspberry Pi 4 |
| `intel_nuc` | 2 | ❌ | Intel NUC Edge Server |
| `jetson_nano` | 2 | ✅ | NVIDIA Jetson Nano |
| `full_server` | All Available | ✅ | Unrestricted Desktop / Server |

### Visualizing Results
After running simulations, telemetry data is saved in the `logs/` directory. Use the plotting utility to visualize performance degradation.

```bash
# Plot the most recent simulation log
python utils/plot_results.py

# Plot detailed latency breakdowns (Pre-processing vs. Forward Pass vs. NMS)
python utils/plot_results.py --file logs/resource_log_xxx.csv --breakdown
```

---

## 📁 Project Architecture

A brief overview of the codebase to help you navigate:

```text
Physical-Visual-DoS-EdgeAI/
├── simulate_edge_server.py       # Main entry point for end-to-end simulation
├── main_train.py                 # Standalone script for GA patch optimization
├── test_physical_dos.py          # Real-world camera and headless testing
│
├── core/
│   ├── victim_model.py           # YOLOv8 wrapper with NMS profiling hooks
│   ├── sponge_fitness.py         # Fitness functions evaluating GrayBox/BlackBox latency
│   └── eot_transforms.py         # Kornia-based EOT augmentations for physical robustness
│
├── attack/
│   └── genetic_algo.py           # The Saliency-Guided Genetic Algorithm implementation
│
├── experiments/                  # Scripts for statistical validation and ablation
│   ├── multi_seed_experiment.py  
│   ├── ablation_patch_size.py    
│   ├── baseline_comparison.py    
│   └── defense_evaluation.py     
│
├── utils/
│   ├── monitor.py                # Hardware resource monitoring and latency tracking
│   └── plot_results.py           # Data visualization for CSV logs
│
└── outputs/                      # Generated patches (including A4 printable versions) and plots
```
