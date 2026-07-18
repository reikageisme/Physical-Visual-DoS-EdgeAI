# Physical Visual DoS: Saliency-Guided Edge-AI Attack Framework

<p align="center">
  <a href="https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/README.md">🇺🇸 English</a> | 
  <a href="https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI/blob/main/README.vi.md">🇻🇳 Tiếng Việt</a>
</p>

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MobileNet%2FYOLO-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Ubuntu%20Server%20(i5--2400)-E95420?logo=ubuntu&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active_Research-success)
![Ethics](https://img.shields.io/badge/AI_Ethics-Strict_Compliance-critical)

> A comprehensive, physical Visual Denial-of-Service (DoS) attack framework targeting Edge-AI object detectors via NMS (Non-Maximum Suppression) overloading. This system utilizes Saliency-Guided Gray-box Genetic Algorithms to crash hardware availability.

![System Architecture](docs/Sponge_GA_Flowchart.png)

---

## 📑 Table of Contents
1. [Overview & Motivation](#1-overview--motivation)
2. [Theoretical Background](#2-theoretical-background)
3. [Attack Architecture & Threat Model](#3-attack-architecture--threat-model)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Hardware & Software Prerequisites](#5-hardware--software-prerequisites)
6. [Installation Guide](#6-installation-guide)
7. [Comprehensive Usage](#7-comprehensive-usage)
8. [Empirical Results](#8-empirical-results)
9. [Project Structure](#9-project-structure)
10. [Defense & Mitigation](#10-defense--mitigation)
11. [Cybersecurity & AI Ethics](#11-cybersecurity--ai-ethics)
12. [Authors & Citation](#12-authors--citation)
13. [License](#13-license)

---

## 1. Overview & Motivation

### The Paradigm Shift in Adversarial Attacks
Historically, adversarial attacks against Deep Neural Networks (DNNs) have focused exclusively on **Misclassification** (i.e., compromising the *Integrity* of the model by making it classify a Stop Sign as a Speed Limit sign). While dangerous, these attacks do not halt the operation of the system itself. 

Our research pivots towards a far more critical but often overlooked vulnerability: **Availability**. By forcing the hardware running the AI to exhaust its computational resources (CPU/RAM), we achieve a **Visual Denial-of-Service (Visual DoS)**. When an Edge Server processing security camera feeds is hit with this attack, the entire surveillance infrastructure freezes. This is particularly devastating for IoT and smart city deployments where edge nodes lack the thermal and computational headroom to recover quickly from sustained resource spikes.

### The Gap in Current Research
Prior works on "Sponge Examples" (Shumailov et al.) or "Phantom Sponges" (Shapira et al.) required *White-box* access (full knowledge of model weights) and were primarily *Digital* (modifying every pixel of the input image). This made them impractical for real-world deployment against physical cameras. Our proposed framework bridges this gap by introducing a **Gray-box** approach (requiring only telemetry data) optimized for physical deployment via a localized "Sponge Patch".

---

## 2. Theoretical Background

### 2.1. The Non-Maximum Suppression (NMS) Bottleneck
Modern Object Detectors (like YOLO, MobileNet-SSD) rely on anchor boxes to predict object locations. A single image pass generates thousands of raw predictions. Most are discarded early due to low confidence scores. The surviving boxes are sent to the NMS algorithm to filter out overlapping predictions based on Intersection over Union (IoU). 

If an attacker can manipulate the input image so that thousands of raw boxes cross the confidence threshold simultaneously, the NMS algorithm is forced to compute the IoU for every pair of boxes. The complexity of this operation is $\mathcal{O}(N^2)$. On an Edge Server lacking a robust GPU, this quadratic matrix calculation instantly spikes CPU usage to 100%, causing significant latency.

### 2.2. Saliency Maps in Adversarial Context
Finding the optimal location to place an adversarial patch is an NP-Hard problem. Random placement often fails because it misses the network's receptive fields. We utilize Saliency Maps—a technique from computer vision that highlights regions of high gradient and spatial frequency—to deterministically anchor our Genetic Algorithm's search space, guaranteeing convergence up to 73% faster.

---

## 3. Attack Architecture & Threat Model

### Threat Model (Gray-Box with Telemetry)
*   **Knowledge:** The attacker does NOT know the internal weights or gradients of the target model.
*   **Access:** The attacker has access to internal telemetry streams, specifically the raw bounding box coordinates and their pre-NMS confidence scores.
*   **Goal:** Maximize CPU usage and minimize FPS (Visual DoS) to halt system availability.

### Attack Pipeline
1. **Target Localization:** The victim camera feed is analyzed to extract a Saliency Map. The most sensitive $5\%$ of the image becomes the patch constraint zone.
2. **Genetic Evolution:** A population of random patches is generated. Each patch is evaluated based on how many boxes it triggers. The best patches undergo crossover and mutation over generations.
3. **Physical Transformation (EOT):** During evolution, patches undergo Expectation Over Transformation (EOT)—including random rotations, blurring, and brightness shifts. This ensures they survive physical printing and camera capture noise.

---

## 4. Mathematical Formulation

### 4.1. NMS Complexity
Let $N$ be the number of active bounding boxes that surpass the confidence threshold $\tau$. The NMS algorithm evaluates every pair, executing combinations:

$$ C(N, 2) = \frac{N(N-1)}{2} $$

### 4.2. Sponge Fitness Function
The Genetic Algorithm maximizes the fitness function $F$, designed to exponentially reward the generation of active boxes while considering their mean confidence $\bar{c}$. $\lambda$ acts as a regularization parameter to balance box quantity versus confidence quality.

$$ F(\text{patch}) = N_{active} + \lambda \cdot \bar{c}_{active} $$

---

## 5. Hardware & Software Prerequisites

### Edge Server (Victim Environment)
To accurately reproduce the IoT bottleneck, avoid running the victim node on a MacBook or high-end Workstation.
*   **CPU:** Intel Core i5-2400 (or similar legacy x86 architectures common in old NVR setups) / ARM Cortex (Raspberry Pi).
*   **RAM:** 8GB DDR3.
*   **OS:** Ubuntu Server 20.04/22.04 LTS (Bare-metal, NO Docker virtualization for accurate hardware profiling).

### Attacker Workstation (Training Environment)
Evolution is computationally expensive.
*   **GPU:** NVIDIA RTX 3090 / 4090 / 5090.
*   **RAM:** 32GB+.
*   **OS:** Windows 11 or Ubuntu.

### Software Stack
*   Python 3.8 - 3.11
*   PyTorch 2.0+ (with CUDA for training)
*   OpenCV-Python
*   NumPy, Matplotlib

---

## 6. Installation Guide

Clone the repository and install the required dependencies. It is highly recommended to use a virtual environment.

```bash
# 1. Clone the repository
git clone https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI.git
cd Physical-Visual-DoS-EdgeAI

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# On Windows:
source venv/Scripts/activate
# On Linux/MacOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 7. Comprehensive Usage

### Step 1: Evolving the Patch (Training)
Run the Genetic Algorithm to evolve a patch. Three threat model variants are available:

```bash
# Gray-box GA (primary method — uses pre-NMS scores)
python main_train.py --pop 50 --gen 100 --size 64

# Strict Black-box GA (uses only latency/observable metrics)
python train_blackbox.py --pop 20 --gen 30 --seed 42

# White-box PGD (experimental — uses gradients, not primary method)
python fast_train.py
```

### Step 2: Multi-seed Statistical Evaluation
Run the GA over multiple seeds for statistical rigor (mean ± std):

```bash
# Saliency-Guided GA: 10 seeds
python experiments/multi_seed_experiment.py --n-seeds 10 --pop 20 --gen 25

# Standard GA (no saliency): 10 seeds
python experiments/multi_seed_experiment.py --n-seeds 10 --pop 20 --gen 25 --no-saliency --out-dir outputs/multi_seed_nosal

# Random Search baseline: 10 seeds, 500 evals each
python experiments/random_search.py --n-seeds 10 --n-evals 500
```

### Step 3: Baseline & Defense Evaluation
Compare Sponge Patch against control textures and evaluate defense strategies:

```bash
# Baseline comparison (random noise, checkerboard, solid, gaussian)
python experiments/baseline_comparison.py --patch outputs/sponge_patch.png --n-trials 30

# Defense evaluation (conf_thresh sweep, max_det cap)
python experiments/defense_evaluation.py --patch outputs/sponge_patch.png --n-trials 20
```

### Step 4: Evaluating the Patch Locally (Digital Injection)
Test the generated patch on your local webcam:

```bash
# Run baseline (clean stream)
python test_physical_dos.py --cam 0

# Run attack (inject patch)
python test_physical_dos.py --cam 0 --patch outputs/sponge_patch.png
```

### Step 5: Simulating the Headless Edge Server
Deploy `web_simulation.py` on your Ubuntu Edge server. It launches an HTTP dashboard on port 5000:

```bash
python web_simulation.py
# Access http://<server-ip>:5000 in your browser
```

---

## 8. Empirical Results

### 8.1. Performance Breakdown (Core i5-2400)
The following table demonstrates the catastrophic failure of the Edge Server when processing a single frame under attack.

| System State | Raw Boxes ($N_{active}$) | IoU Operations/Frame | NMS Latency (ms) | Core i5-2400 Load | 720p FPS Impact |
|---|---|---|---|---|---|
| **Clean Stream** | ~ 47 | ~ 1,081 | ~ 0.99 ms | 52% - 64% | 15 - 21 FPS |
| **Under Attack** | ~ 56 (Max 118) | ~ 1,540 (Max 6,903) | ~ 1.92 ms (Avg) | 67% - 78% (Max 100%) | 5 - 11 FPS |

*Note: In Edge-AI constraints, a jump from 1,081 to nearly 7,000 IoU matrix operations per frame creates an insurmountable I/O and memory bandwidth bottleneck, driving the FPS down to unplayable levels.*

### 8.2. Ablation Study: GA vs Saliency
Integrating the Saliency Map constraints yields significantly higher fitness compared to Standard GA and Random Search, proving that structural targeting is required for Visual DoS.

*   **Saliency-Guided GA:** `Best Fitness = 65.07 ± 4.67` | `Convergence Gen = 15.4 ± 3.8` (n=10 seeds)
*   **Standard GA:** `Best Fitness = 64.02 ± 5.17` | `Convergence Gen = 14.7 ± 2.3` (n=10 seeds)
*   **Random Search:** `Best Fitness = ~15.30 ± 0.08` | `N/A` (n=10 seeds)

---

## 9. Project Structure

Below is a detailed breakdown of the critical components within the repository:

### Core Modules
*   `attack/genetic_algo.py`: Core GA implementation. Handles crossover, centered mutation (unbiased), elite preservation, and Saliency Map masking.
*   `core/victim_model.py`: Wraps the PyTorch (MobileNet/YOLO) models. Exposes the Gray-box telemetry API (`get_raw_predictions`) and NMS profiling (`get_predictions_with_nms`).
*   `core/sponge_fitness.py`: Custom fitness function for GA optimization. Includes both `calculate_sponge_fitness` (gray-box) and `ObservableFitness` (strict black-box, latency-based).
*   `core/eot_transforms.py`: Expectation Over Transformation — rotations, blur, brightness shifts for physical robustness.

### Training Scripts
*   `main_train.py`: Primary gray-box GA training pipeline (64×64 patch on 320×320 input).
*   `train_blackbox.py`: **[NEW]** Strict black-box variant using `ObservableFitness` (latency-only, no internal scores).
*   `fast_train.py`: Experimental white-box PGD variant (uses gradients — separate threat model).

### Experiment Scripts
*   `experiments/multi_seed_experiment.py`: Multi-seed GA evaluation (10 seeds, mean ± std reporting).
*   `experiments/random_search.py`: Random search baseline for fair comparison.
*   `experiments/baseline_comparison.py`: Compares Sponge Patch vs control textures (random, checkerboard, solid, Gaussian).
*   `experiments/defense_evaluation.py`: Evaluates defense strategies (conf_thresh sweep, max_det cap).
*   `experiments/ablation_patch_size.py`: Patch size ablation (32, 48, 64, 96, 128 px).

### Deployment & Simulation
*   `proper_dos_sim.py`: Isolated $\mathcal{O}(N^2)$ NMS benchmarking tool with per-stage latency profiling.
*   `simulate_edge_server.py`: Full edge server simulation with hardware telemetry.
*   `web_simulation.py`: Flask web dashboard for remote monitoring.
*   `test_physical_dos.py`: Local webcam digital injection tester.
*   `test_headless_pi.py`: Headless Raspberry Pi / IP camera tester.

### Utilities
*   `utils/monitor.py`: Bare-metal hardware telemetry logger (`psutil` CPU/RAM).
*   `utils/plot_results.py`: Visualization tools (multi-seed convergence, latency breakdown, scenario comparison).
*   `utils/find_cam.py`: Camera device discovery utility.

---

## 10. Defense & Mitigation

While this repository demonstrates the attack, protecting Edge AI systems against Visual DoS is an ongoing research topic. Proposed mitigations include:

1.  **NMS-Free Architectures:** Transitioning to Transformer-based detectors like DETR or RT-DETR, which use bipartite matching instead of NMS, fundamentally removing the $\mathcal{O}(N^2)$ bottleneck.
2.  **Hardware-Level Limits:** Enforcing strict hard-caps on the number of proposals accepted per frame at the memory buffer level.
3.  **Frequency Analysis Filtering:** Pre-processing input frames to detect and blur out unnatural high-frequency spatial noise (Sponge Patches) before it hits the CNN backbone.

---

## 11. Cybersecurity & AI Ethics

### Responsible Disclosure & Dual-Use Nature
The intersection of Artificial Intelligence and physical cybersecurity creates dual-use technologies. The methodologies presented in this framework (Saliency-Guided GA, Visual DoS) possess the capability to temporarily disable critical physical infrastructure, including security cameras, autonomous vehicle sensors, and industrial visual monitors. 

**We strongly emphasize that this research is published strictly under the doctrine of Responsible Disclosure.** By openly discussing the mathematical vulnerability of the NMS algorithm and providing empirical evidence of its failure modes on legacy hardware, we aim to equip defense engineers with the knowledge necessary to build resilient AI systems. 

### Ethical Guidelines for Usage
1.  **Authorization:** You must obtain explicit, written authorization from the owners of any hardware, network, or physical surveillance system before deploying this software.
2.  **Containment:** All experiments must be conducted in isolated, non-production environments (e.g., local testbeds, sandboxed edge devices).
3.  **No Malicious Intent:** This tool shall not be weaponized to facilitate physical intrusions, bypass security checkpoints, or degrade public safety infrastructure.

---

## 12. Authors & Citation

This research and corresponding framework were developed by the following authors under the Scientific Research Program (NCKH) 2025-2026 at HUTECH University.

### 👤 First Author: Pham Tuan Anh (Reikage)
*   **Role:** Lead AI Security Researcher
*   **Contributions:** Conceptualized the Visual DoS attack vector, formulated the Saliency-Guided GA mathematical model, designed the Gray-box optimization pipeline, and developed the core adversarial evolution logic.
*   **Contact:** anh25807700004@hutech.edu.vn | [GitHub: @reikageisme](https://github.com/reikageisme)

### 👤 Co-Author: Mai Quoc Bao (BaoZ)
*   **Role:** Systems Architecture & Edge Deployment Lead
*   **Contributions:** Engineered the bare-metal hardware profiling mechanisms, designed the real-time webcam testing pipeline, developed the Headless Ubuntu web simulation, and verified empirical metrics.
*   **Contact:** bao2580770008@hutech.edu.vn

### How to Cite this work
If you use this framework in your academic research, please consider citing our work:

```bibtex
@misc{pham2025visualdos,
  author = {Pham, Tuan Anh and Mai, Quoc Bao},
  title = {Adversarial Sponge Attack: Visual Denial-of-Service on Edge Server via Saliency-Guided Genetic Algorithms},
  year = {2025},
  publisher = {HUTECH University},
  howpublished = {\url{https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI}},
  note = {Scientific Research Program 2025-2026}
}
```

---

## 13. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 

### MIT License Summary
Copyright (c) 2025 Pham Tuan Anh, Mai Quoc Bao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
