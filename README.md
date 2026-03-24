# 🧽 Physical Visual DoS: Saliency-Guided Edge-AI Attack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MobileNet%2FYOLO-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-C51A4A?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **A physical Visual Denial-of-Service (DoS) attack framework targeting Edge-AI object detectors via NMS overloading using Saliency-Guided Black-box Genetic Algorithms.**

![System Architecture](docs/Sponge_GA_Flowchart.png)

## 📖 Overview

While traditional physical adversarial attacks focus on **Misclassification** (tricking the AI into seeing the wrong object), this research explores a critical resource-depletion vulnerability: **Visual Denial-of-Service (Visual DoS)**.

By integrating a **Deterministic Saliency-Guided Mechanism** with a **Black-box Genetic Algorithm (GA)**, the system dynamically identifies the "most salient" regions of a video frame and optimizes an adversarial "Sponge Patch". This patch forces the AI to generate thousands of raw bounding boxes simultaneously, creating a severe computational bottleneck at the **Non-Maximum Suppression (NMS)** filtering stage. The result? 100% CPU overload and frozen real-time camera streams on resource-constrained Edge devices.

---

## ⚡ Key Innovations: Misclassification vs. Visual DoS

| Feature | Traditional Attacks (Legacy) | 🚀 Proposed Visual DoS (Current) |
| :--- | :--- | :--- |
| **Attack Goal** | Deceive the classifier (Integrity) | **Crash the hardware (Availability)** |
| **Optimization** | White-box Gradient Descent | **Black-box Genetic Algorithm (GA)** |
| **Space Search** | Random or Center-fixed | **Dynamic Saliency-Guided Targeting** |
| **Target Vulnerability** | Loss Function (Cross-Entropy) | **NMS Complexity Bottleneck** $\mathcal{O}(N^2)$ |
| **Deployment** | Virtualized / Abstract | **Direct Bare-Metal Execution (Native OS)** |

---

## 🧠 Core Logic: How It Works

The attack pipeline executes the following mathematical and algorithmic steps:

1. **Saliency-Guided Localization**: The system computes a Gradient-based Saliency Map $S(x)$ and applies Gaussian Blur to find the most sensitive region $(u^*, v^*)$ on the frame to place the patch, drastically reducing the GA search space.
2. **Black-box Evolution**: The Genetic Algorithm evolves the patch to maximize the number of generated raw boxes ($N_{active}$) with high confidence ($C_i > \tau$).
3. **EOT Integration**: Expectation Over Transformation (EOT) is applied (rotation, scaling, noise) to ensure the patch remains lethal in the physical world.
4. **NMS Overloading**: The explosion of raw boxes forces the CPU to calculate Intersection over Union (IoU) for every pair. The required operations scale quadratically:

$$ \text{Operations} = \frac{N_{active} \times (N_{active} - 1)}{2} $$

*This massive mathematical overhead instantly throttles the ARM CPU of the Edge device.*

---

## 🏗 Project Structure

Refactored for optimal physical hardware profiling (no virtualization overhead):

```text
Physical-Visual-DoS-EdgeAI/
├── requirements.txt        # Python dependencies
├── attack/                 # [CORE] Genetic Algorithm implementation
├── core/                   # [CORE] EOT transforms, Fitness functions, Victim Model
├── logs/                   # [DATA] Profiling metrics (CPU/FPS logs in CSV)
├── utils/                  # [TOOLS] Profiling chart generators & camera utils
├── docs/                   # Documentation & Flowcharts
├── outputs/                # Generated adversarial patches (.png)
├── main_train.py           # Entry point: Train the physical patch via GA
└── test_physical_dos.py    # Entry point: Local camera testing & Live HUD
```

---

## 🚀 Deployment Guide

### Prerequisites
* **Hardware:** Raspberry Pi 4 (8GB RAM recommended) or any Linux/Windows PC.
* **Camera:** IP Camera (MJPEG Stream), USB Webcam, or DroidCam.
* **Note on Profiling:** This project runs strictly on **bare-metal python** without Docker. Virtualization alters CPU load measurements and thermal constraints, making DoS research metrics unrealistic.

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI.git
cd Physical-Visual-DoS-EdgeAI

# 2. Create a virtual environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Generate the Sponge Patch
Run the training script to evolve the patch offline using the Genetic Algorithm. You can customize the population, generations, and patch size:
```bash
python main_train.py --pop 50 --gen 100 --size 64
```
*The optimally generated patch will be saved in `outputs/`.*

### 2. Test Physical Attack (Live HUD)
To observe the NMS bottleneck effect on a local machine via Webcam or DroidCam:

**A. Baseline Check (No Attack)**
```bash
python test_physical_dos.py --cam 0
```

**B. Digital Overlay Attack (Apply patch implicitly in the video stream)**
```bash
python test_physical_dos.py --cam 0 --patch outputs/sponge_patch_g100_p50.png
```

### 3. Generate Evaluation Charts
After running the DoS test, the system auto-logs hardware state via `utils/monitor.py`. Generate the final NCKH chart:
```bash
python utils/plot_results.py
```

---

## 📊 Performance Metrics (Raspberry Pi 4B - 8GB)
Empirical evidence of the physical attack's impact on Edge-AI hardware:

| System State | Raw Boxes ($N_{active}$) | IoU Operations/Frame | CPU Load | FPS Impact |
|---|---|---|---|---|
| Clean Stream | ~ 15 - 25 | ~ 300 | 20% - 25% | 30 FPS (Smooth) |
| Under Attack | 300 (Max) | ~ 44,850 | 100% (Overload) | < 2 FPS (Frozen) |

*The attack forces the system into an unstable state, effectively blinding the surveillance system.*

---

## 👨‍💻 Authors & Acknowledgments
*   **Reikage**: Core Algorithm (Saliency Logic), Genetic Optimization.
*   **BaoZ**: IoT Architecture, Dockerization, Web Interface.


*Project conducted under the Scientific Research Program (NCKH) 2025-2026.*

## ⚖️ Disclaimer
This project is for educational and academic research purposes only. The vulnerabilities demonstrated involve physical resource depletion. Do not use this framework against production surveillance systems without explicit authorization.