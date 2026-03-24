# 🧽 Physical Visual DoS: Saliency-Guided Edge-AI Attack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MobileNet%2FYOLO-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
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
| **Deployment** | Static Python Scripts | **Dockerized Microservice & Web Dashboard** |

---

## 🧠 Core Logic: How It Works

The attack pipeline executes the following mathematical and algorithmic steps:

1. **Saliency-Guided Localization**: The system computes a Gradient-based Saliency Map $S(x)$ and applies Gaussian Blur to find the most sensitive region $(u^*, v^*)$ on the frame to place the patch, drastically reducing the GA search space.
2. **Black-box Evolution**: The Genetic Algorithm evolves the patch to maximize the number of generated raw boxes ($N_{active}$) with high confidence ($C_i > \tau$).
3. **EOT Integration**: Expectation Over Transformation (EOT) is applied (rotation, scaling, noise) to ensure the patch remains lethal in the physical world.
4. **NMS Overloading**: The explosion of raw boxes forces the CPU to calculate Intersection over Union (IoU) for every pair. The required operations scale quadratically:
   $$Operations = \frac{N_{active} \times (N_{active} - 1)}{2}$$
   *This massive mathematical overhead instantly throttles the ARM CPU of the Edge device.*

---

## 🏗 Project Structure

Refactored for modularity, MVC architecture, and Docker support:

```text
Physical-Visual-DoS-EdgeAI/
├── Dockerfile              # Container configuration (ARM64 optimized)
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # Python dependencies
├── attack/                 # [CORE] Genetic Algorithm implementation
├── core/                   # [CORE] EOT transforms, Fitness functions, Victim Model
├── web/                    # [WEB] Flask Server & Interface
│   ├── app.py              # Flask entry point
│   └── templates/          # HTML Dashboard
├── utils/                  # [TOOLS] Camera diagnostic and monitoring scripts
├── docs/                   # Documentation & Flowcharts
├── outputs/                # Generated adversarial patches (.png)
├── main_train.py           # Entry point: Train the physical patch via GA
└── test_physical_dos.py    # Entry point: Local camera/dashboard testing
```

---

## 🚀 Deployment Guide

### Prerequisites
* **Hardware:** Raspberry Pi 4 (8GB RAM recommended) or any Linux/Windows PC.
* **Camera:** IP Camera (MJPEG Stream) or USB Webcam (V4L2).
* **Software:** Docker Engine & Docker Compose.

### Method 1: Docker Microservice (Recommended for Raspberry Pi)
This method automatically handles all dependencies and sets up the Web Dashboard.

```bash
# 1. Clone the repository
git clone https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI.git
cd Physical-Visual-DoS-EdgeAI

# 2. Build and Run the container
docker-compose up --build -d

# 3. View logs (Observe CPU throttling in real-time)
docker-compose logs -f
```

### Method 2: Manual Installation (For PC/Laptop Simulation)
Suitable for offline patch training and local testing.

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test Physical Attack (View HUD locally)
python test_physical_dos.py
```

---

## 🎮 Web Dashboard Interface
When deployed via Docker, access the control center at:
👉 `http://localhost:5000` (or `http://<RASPBERRY_PI_IP>:5000`)

**Dashboard Features:**
* **Live Surveillance Feed:** Displays the real-time stream from the Edge Camera.
* **Telemetry Panel:** Live tracking of FPS, CPU Load, and RAM Usage.
* **Attack Status:** Visual indicator turning RED when the NMS Bottleneck is triggered and $N_{active}$ exceeds the safe threshold.

---

## 📊 Performance Metrics (Raspberry Pi 4B - 8GB)
Empirical evidence of the physical attack's impact on Edge-AI hardware:

| System State | Raw Boxes ($N_{active}$) | IoU Operations/Frame | CPU Load | FPS Impact |
|---|---|---|---|---|
| Clean Stream | ~ 15 - 25 | ~ 300 | 20% - 25% | 30 FPS (Smooth) |
| Under Attack (Visual DoS) | 300 (Max) | ~ 44,850 | 100% (Overload) | < 2 FPS (Frozen) |

---

## 👨‍💻 Author
**ReiKage (Phạm Tuấn Anh)**
*Information Security Researcher | Proud member of CTF Team: 6h4T 9pT pR0*

*Project conducted under the Scientific Research Program (NCKH) 2025-2026.*

## ⚖️ Disclaimer
This project is for educational and academic research purposes only. The vulnerabilities demonstrated involve physical resource depletion. Do not use this framework against production surveillance systems without explicit authorization.