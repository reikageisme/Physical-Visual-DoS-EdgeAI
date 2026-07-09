# Sponge Patch — Physical Visual DoS on Edge-AI Systems

🌍 *[English Version below](#english-version)* | 🇻🇳 *[Phiên bản Tiếng Việt bên dưới](#phiên-bản-tiếng-việt)*

---

## 🇻🇳 Phiên bản Tiếng Việt

### Tổng quan (Overview)
> **Mô hình Mối đe dọa (Threat Model):** Gray-box (truy cập điểm số trước NMS - pre-NMS score access)  
> **Phương pháp Tấn công (Attack):** Thuật toán Di truyền (Genetic Algorithm) + Tối ưu hóa Patch dựa trên độ nổi bật (Saliency-guided)  
> **Mục tiêu (Target):** YOLOv8n trên Edge Server (mô phỏng / thực tế)

### 🚀 Bắt đầu nhanh (Quick Start)

#### 1. Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

#### 2. Chạy mô phỏng toàn bộ (Không cần camera)
```bash
# Demo nhanh (~2 phút)
python simulate_edge_server.py --quick

# Thử nghiệm đầy đủ: Huấn luyện GA patch + test cả 2 kịch bản (sạch và bị tấn công)
python simulate_edge_server.py --frames 200 --train --pop 20 --gen 30

# Mô phỏng giới hạn phần cứng của Raspberry Pi
python simulate_edge_server.py --frames 200 --train --edge-profile raspberry_pi

# Sử dụng patch đã huấn luyện từ trước
python simulate_edge_server.py --frames 200 --patch outputs/sponge_patch.png
```

#### 3. Chạy các thử nghiệm độc lập
```bash
# Đánh giá thống kê đa hạt giống (10 seeds, trung bình ± độ lệch chuẩn)
python experiments/multi_seed_experiment.py --n-seeds 10 --pop 20 --gen 30

# Ablation study: Kích thước patch từ 1% đến 16%
python experiments/ablation_patch_size.py

# So sánh với các baseline (random / checkerboard / sponge)
python experiments/baseline_comparison.py --patch outputs/sponge_patch.png

# Đánh giá phòng thủ (Ngưỡng confidence / Giới hạn max_det)
python experiments/defense_evaluation.py --patch outputs/sponge_patch.png
```

#### 4. Chỉ huấn luyện Patch (Lưu patch để sử dụng sau)
```bash
# Huấn luyện GA patch (64×64px, ~4% diện tích khung hình 320×320)
python main_train.py --size 64 --pop 30 --gen 50 --seed 42

# Huấn luyện theo tỷ lệ phần trăm diện tích
python main_train.py --area-pct 4.0 --resolution 320 --seed 42

# Huấn luyện với thuật toán lai ghép khác
python main_train.py --size 64 --pop 30 --gen 50 --crossover uniform
```

#### 5. Thử nghiệm với Camera thực tế
```bash
# Baseline sạch (không bị tấn công)
python test_physical_dos.py --scenario clean

# Tấn công bằng cách chèn patch kỹ thuật số
python test_physical_dos.py --patch outputs/sponge_patch.png --scenario digital_attack

# Mô phỏng không giao diện (headless, không cần camera)
python test_physical_dos.py --simulate --frames 200 --scenario digital_attack
```

#### 6. Vẽ biểu đồ kết quả
```bash
# Vẽ biểu đồ từ log mới nhất
python utils/plot_results.py

# Vẽ biểu đồ với phân tích chi tiết độ trễ (latency breakdown)
python utils/plot_results.py --file logs/resource_log_xxx.csv --breakdown

# Vẽ biểu đồ hội tụ đa hạt giống
python utils/plot_results.py --multi-seed outputs/multi_seed/
```

### 💻 Cấu hình Phần cứng Edge (Edge Hardware Profiles)

| Profile | Threads (Luồng) | GPU | Thiết bị mô phỏng |
|---|---|---|---|
| `raspberry_pi` | 1 | ❌ | Raspberry Pi 4 |
| `intel_nuc` | 2 | ❌ | Intel NUC |
| `jetson_nano` | 2 | ✅ | Jetson Nano |
| `full_server` | all | ✅ | Full Edge Server |

```bash
# Ví dụ chạy với profile raspberry_pi
python simulate_edge_server.py --edge-profile raspberry_pi --frames 200 --train
```

### 📁 Cấu trúc Thư mục (Project Structure)

```text
Physical-Visual-DoS-EdgeAI/
├── simulate_edge_server.py       # ← MAIN: Mô phỏng toàn bộ pipeline
├── main_train.py                 # Huấn luyện Patch (chỉ dùng GA)
├── fast_train.py                 # Huấn luyện Patch (dùng PGD gradient)
├── test_physical_dos.py          # Test camera / mô phỏng headless
│
├── core/
│   ├── victim_model.py           # Wrapper YOLOv8 (gray-box + NMS profiling)
│   ├── sponge_fitness.py         # Hàm Fitness: GrayBox + ObservableFitness
│   └── eot_transforms.py         # Kỹ thuật Data Augmentation EOT (Kornia)
│
├── attack/
│   └── genetic_algo.py           # Saliency-Guided GA (seed + hội tụ)
│
├── experiments/
│   ├── multi_seed_experiment.py  # Đánh giá tính thống kê (Major Issue 5)
│   ├── ablation_patch_size.py    # Thử nghiệm kích thước patch (Major Issue 4)
│   ├── baseline_comparison.py    # So sánh baseline (Major Issue 9)
│   └── defense_evaluation.py     # Đánh giá phòng thủ (Major Issue 10)
│
├── utils/
│   ├── monitor.py                # Theo dõi tài nguyên + phân tích độ trễ
│   └── plot_results.py           # Vẽ biểu đồ hiệu năng / độ trễ / multi-seed
│
├── docs/
│   ├── REVIEWED.md               # Chi tiết phản biện peer review
│   └── BaoCaoKhoaHocAdversarialAttacks2.docx
│
└── outputs/                      # Thư mục lưu patches + kết quả chạy
```

### 🛡️ Làm rõ Mô hình Mối đe dọa (Threat Model Clarification)

| Chế độ (Mode) | Quyền truy cập (Access) | Dùng bởi (Used by) |
|---|---|---|
| **Gray-box** | Lấy điểm confidence score gốc trước NMS (cần bản copy local của model) | `main_train.py`, Thuật toán Di truyền (GA) |
| **Observable Black-box** | Chỉ nhận bounding box sau NMS + độ trễ | `ObservableFitness` trong `sponge_fitness.py` |
| **White-box** | Truy cập toàn bộ gradient của model | `fast_train.py` (PGD) |

### 📂 Kết quả Đầu ra (Outputs)

Sau khi chạy mô phỏng, các file sau sẽ được tạo ra:

```text
outputs/
├── sponge_patch_*.png            # Patch đã tối ưu (bản nhỏ + bản in A4)
├── sponge_patch_A4_*.png         # Bản in A4 độ phân giải 300DPI
├── ga_summary_*.json             # Thống kê GA sau mỗi lần chạy
├── plot_clean.png                # Đồ thị hiệu suất lúc bình thường (clean)
├── plot_attack.png               # Đồ thị hiệu suất lúc bị tấn công
├── latency_clean.png             # Phân tách thời gian NMS/Forward/Preproc (clean)
├── latency_attack.png            # Phân tách thời gian NMS/Forward/Preproc (attack)
└── scenario_comparison.png       # Đồ thị so sánh 2 kịch bản

logs/
├── resource_log_clean_*.csv      # Log từng frame: FPS/CPU/RAM/NMS_ms
└── resource_log_digital_attack_*.csv
```

---
<br>

## 🇬🇧 English Version

### Overview
> **Threat Model:** Gray-box (pre-NMS score access)  
> **Attack:** Genetic Algorithm + Saliency-guided Patch Optimization  
> **Target:** YOLOv8n on Edge Server (simulated / physical)

### 🚀 Quick Start

#### 1. Install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run the full simulation (no camera needed)
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

#### 3. Run individual experiments
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

#### 4. Train only (save patch for later)
```bash
# Train GA patch (64×64px, ~4% of 320×320 frame)
python main_train.py --size 64 --pop 30 --gen 50 --seed 42

# Train with area percentage
python main_train.py --area-pct 4.0 --resolution 320 --seed 42

# Train with different crossover
python main_train.py --size 64 --pop 30 --gen 50 --crossover uniform
```

#### 5. Test with real camera
```bash
# Clean baseline
python test_physical_dos.py --scenario clean

# Digital attack overlay
python test_physical_dos.py --patch outputs/sponge_patch.png --scenario digital_attack

# Headless simulation (no camera)
python test_physical_dos.py --simulate --frames 200 --scenario digital_attack
```

#### 6. Plot results
```bash
# Plot latest log
python utils/plot_results.py

# Plot with latency breakdown
python utils/plot_results.py --file logs/resource_log_xxx.csv --breakdown

# Multi-seed convergence plot
python utils/plot_results.py --multi-seed outputs/multi_seed/
```

### 💻 Edge Hardware Profiles

| Profile | Threads | GPU | Simulates |
|---|---|---|---|
| `raspberry_pi` | 1 | ❌ | Raspberry Pi 4 |
| `intel_nuc` | 2 | ❌ | Intel NUC |
| `jetson_nano` | 2 | ✅ | Jetson Nano |
| `full_server` | all | ✅ | Full Edge Server |

```bash
# Example running with raspberry_pi profile
python simulate_edge_server.py --edge-profile raspberry_pi --frames 200 --train
```

### 📁 Project Structure

```text
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
│   └── defense_evaluation.py     # Defense eval (Major Issue 10)
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

### 🛡️ Threat Model Clarification

| Mode | Access | Used by |
|---|---|---|
| **Gray-box** | Pre-NMS raw confidence scores (local model copy) | `main_train.py`, GA |
| **Observable Black-box** | Post-NMS detections + latency only | `ObservableFitness` in `sponge_fitness.py` |
| **White-box** | Full gradient access | `fast_train.py` (PGD) |

### 📂 Outputs

After running the simulation:

```text
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
