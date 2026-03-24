# 🧽 Physical Visual DoS on Edge-AI 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A physical Visual Denial-of-Service (DoS) attack framework targeting Edge-AI object detectors via NMS overloading using Black-box Genetic Algorithms.**

![System Architecture](docs/Sponge_GA_Flowchart.png)

## 📖 Overview
While most physical adversarial attacks focus on **Misclassification** (tricking the AI into seeing the wrong object), this research explores a critical resource-depletion vulnerability: **Visual Denial-of-Service (Visual DoS)**.

By generating a highly optimized "Sponge Patch", we force the target Object Detection model (e.g., MobileNet-SSD, YOLO) to generate thousands of raw bounding boxes simultaneously. This sudden explosion of data creates a severe computational bottleneck at the **Non-Maximum Suppression (NMS)** filtering stage (which has $\mathcal{O}(N^2)$ complexity), ultimately maxing out the CPU and freezing the real-time camera stream on resource-constrained Edge devices (like Raspberry Pi).

## ✨ Key Features
* **Strict Black-box Attack:** Does not require access to the model's architecture, weights, or gradients.
* **Saliency-Guided Evolution:** Uses Saliency Maps to locate the most vulnerable coordinates in the frame, drastically reducing the search space for the Genetic Algorithm.
* **Physical Deployment (EOT):** Incorporates Expectation Over Transformation (EOT) to ensure the patch remains lethal even when printed on paper and captured through real-world IP cameras (handling angle, lighting, and compression noise).
* **Hardware-Crashing Impact:** Proven to push ARM-based CPUs to 100% load, causing severe FPS drops ($\approx 1$ FPS) on Edge-AI systems.

## 📂 Repository Structure
```text
Physical-Visual-DoS-EdgeAI/
├── attack/                 # Core Genetic Algorithm implementation
├── core/                   # EOT transforms, Fitness functions, Victim Model
├── docs/                   # System architecture flowcharts and diagrams
├── logs/                   # Profiling metrics (CPU/FPS logs)
├── outputs/                # Generated adversarial patches (.png)
├── utils/                  # Camera diagnostic and monitoring scripts
├── main_train.py           # Entry point: Train the physical patch via GA
├── test_physical_dos.py    # Entry point: Test attack on PC/Laptop
├── test_headless_pi.py     # Entry point: Deploy attack on Raspberry Pi
└── requirements.txt        # Python dependencies
```

## 🚀 Installation
Clone this repository:
```bash
git clone https://github.com/reikageisme/Physical-Visual-DoS-EdgeAI.git
cd Physical-Visual-DoS-EdgeAI
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage
### 1. Generate the Sponge Patch
Run the training script to evolve the patch offline using the Saliency-guided Genetic Algorithm.
```bash
python main_train.py
```
*The optimized patch will be saved in `outputs/sponge_patch_final.png`.*

### 2. Test Physical Attack (PC/Laptop Simulator)
To observe the NMS bottleneck effect on a local machine via Webcam or DroidCam:
```bash
python test_physical_dos.py
```
*Note: Point your camera at a blank wall to measure the Baseline, then point it at the printed/displayed patch to observe the explosion of Raw Boxes.*

### 3. Edge-AI Hardware Profiling (Raspberry Pi)
Deploy the headless script on a Raspberry Pi to monitor real-time CPU throttling and FPS drops:
```bash
python test_headless_pi.py
```

## 📊 Experimental Results
When tested on a Raspberry Pi 4B (8GB) processing a live IP Camera stream, the physical Sponge Patch successfully triggered the NMS vulnerability:

| State | Raw Boxes ($N_{active}$) | IoU Operations/Frame | CPU Load | FPS |
|---|---|---|---|---|
| Clean Stream | ~ 15 | ~ 105 | 20% - 25% | 30 FPS |
| Under Attack | 300 (Max) | ~ 44,850 | 100% (Overload) | < 2 FPS |

*The attack forces the system into an unstable state, effectively blinding the surveillance system.*

## 🤝 Author
**ReiKage (Phạm Tuấn Anh)** 
* *Information Security Researcher* | *Proud member of CTF Team: 6h4T 9pT pR0*

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

**Disclaimer:** This project is strictly for educational and academic research purposes. Do not use this framework against production surveillance systems without explicit authorization.