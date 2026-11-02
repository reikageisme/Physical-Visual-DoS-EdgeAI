# Q1/Q2 Research Roadmap — Physical Visual DoS / Sponge Patch for Edge Object Detectors

> **Mục đích của tài liệu:** Đây là “master roadmap” để đặt trực tiếp trong repository và dùng làm chuẩn làm việc cho toàn bộ quá trình nâng cấp bài báo từ bản hiện tại lên mức **Q2 khả thi** và hướng tới **Q1 có cơ sở**.  
> Tài liệu này tổng hợp các vấn đề đã phát hiện trong bản thảo hiện tại, các bất nhất giữa paper và repository, chiến lược Laptop → VM → Real Edge Server, protocol thực nghiệm, thống kê, quản lý log, cách viết lại claim, và checklist trước khi nộp.
>
> **Nguyên tắc cứng:** `NO RAW LOG = NO RESULT IN PAPER`.

---

## 0. Executive Summary

Ý tưởng nghiên cứu có tiềm năng: thay vì chỉ đánh lừa classification/integrity, Sponge Patch nhắm vào **availability** của object detection pipeline bằng cách làm tăng workload đi vào bước post-processing, đặc biệt là NMS, từ đó gây tăng latency, giảm FPS, tăng resource pressure trên edge systems.

Tuy nhiên, bản hiện tại **chưa nên gửi Q1/Q2 ngay** vì bằng chứng thực nghiệm chưa đủ nhất quán để bảo vệ các claim mạnh. Vấn đề lớn nhất không nằm ở ý tưởng, mà nằm ở:

1. Số liệu trong paper chưa khớp hoàn toàn với raw logs/JSON trong repository.
2. Một số kết quả là simulation nhưng đang được diễn đạt gần như real hardware result.
3. “Raw boxes” và “post-confidence candidates” chưa được phân biệt chặt.
4. Claim NMS `O(N²)` đang viết quá tuyệt đối so với implementation thực tế.
5. Transferability hiện tại chưa chứng minh “systemic vulnerability” trên nhiều kiến trúc.
6. Threat model gray-box cần formalize rõ hơn.
7. Physical evaluation hiện mới ở mức proof-of-concept.
8. CPU%/RAM% tuyệt đối không phải metric tốt nhất để so giữa laptop, VM và máy edge cũ.
9. Statistical protocol hiện chưa đủ chặt để đáp ứng reviewer Q1/Q2.
10. Repo đã có nhiều script tốt, nhưng chưa có một **reproducible experiment engine** thống nhất.

### Đánh giá làm việc hiện tại

| Hạng mục | Điểm hiện tại ước lượng |
|---|---:|
| Research problem | 8/10 |
| Novelty | 6.5/10 |
| Threat model | 6.5/10 |
| Method | 6/10 |
| Mathematical rigor | 5.5/10 |
| Experiments | 3.5/10 |
| Statistical rigor | 4/10 |
| Reproducibility | 5/10 |
| Physical-world validation | 3.5/10 |
| Baseline/SOTA comparison | 4/10 |
| Writing/presentation | 6/10 |
| **Tổng thể** | **~52–58/100** |

### Mục tiêu sau khi nâng cấp

- **Q2 khả thi:** ~72–78/100 nếu làm lại benchmark VM có kiểm soát, real-i5 confirmation, baseline, negative control, defense, statistics.
- **Q2 mạnh / có cửa Q1:** ~80–86/100 nếu thêm cross-architecture characterization, physical robustness tốt hơn, causal NMS evidence, defense trade-off.
- **Q1 nghiêm túc:** ~85–90/100 nếu thêm contribution thuật toán rõ ràng (query efficiency hoặc hardware-adaptive characterization) và benchmark đủ rộng.

---

# 1. Trạng thái hiện tại của repository

Repository hiện đã có nền tảng tốt:

```text
attack/
    genetic_algo.py

core/
    eot_transforms.py
    sponge_fitness.py
    victim_model.py
    victim_model_zoo.py

experiments/
    ablation_patch_size.py
    baseline_comparison.py
    benchmark_hardware.py
    defense_evaluation.py
    defense_tradeoff.py
    multi_seed_experiment.py
    random_search.py
    transferability.py

utils/
    monitor.py
    physical_test_logger.py
    plot_results.py

outputs/
    ablation_size/
    baseline_comparison/
    benchmark/
    blackbox/
    defense_eval/
    defense_tradeoff/
    multi_seed/
    multi_seed_nosal/
    proper_sim/
    random_search/
    transferability/
```

Đây là điểm mạnh. Vấn đề không phải “thiếu code attack”, mà là thiếu một tầng orchestration/reproducibility để mọi kết quả được sinh theo một protocol duy nhất.

---

# 2. Audit dữ liệu hiện tại: các bất nhất cần xử lý trước

## 2.1. Table 2 trong paper vs `outputs/proper_sim/table2_results.json`

Kết quả hiện có trong `table2_results.json`:

### Clean

- FPS mean: `23.64`
- NMS mean: `0.991 ms`
- raw boxes mean: `47.3`
- raw boxes max: `95`
- forward mean: `42.0 ms`

### Digital

- FPS mean: `19.17`
- NMS mean: `1.919 ms`
- raw boxes mean: `55.7`
- raw boxes max: `118`
- forward mean: `65.48 ms`

### Physical-simulated

- FPS mean: `6.36`
- NMS mean: `5.273 ms`
- raw boxes mean: `53.4`
- raw boxes max: `105`
- forward mean: `157.71 ms`

### O² analysis currently stored

- box ratio: `1.178`
- NMS time ratio: `1.936`
- O² predicted ratio: `1.387`
- percentage error: `39.6%`

### Vấn đề

Paper hiện từng mô tả attack có `3000+` boxes. Dữ liệu JSON này **không hỗ trợ claim đó**.

**Hành động:**  
Không đưa `3000+` vào paper trừ khi có raw log thật từ experiment mới chứng minh rõ:

```text
Num_Conf_Candidates >= 3000
```

và phải biết chính xác đó là:

- raw detector outputs,
- post-confidence candidates,
- pre-NMS candidates,
- hay post-NMS detections.

---

## 2.2. Transferability hiện tại

`outputs/transferability/cross_arch_results.json`:

| Model | NMS? | Clean ms | Patched ms | Ratio |
|---|---:|---:|---:|---:|
| YOLOv8n | Yes | 97.22 | 114.09 | **1.173×** |
| Faster R-CNN MobileNetV3 | Yes | 387.44 | 388.05 | **1.002×** |
| RetinaNet ResNet50 | Yes | 3188.58 | 2960.76 | **0.929×** |
| DETR ResNet50 | No | 412.69 | 417.56 | **1.012×** |

### Kết luận đúng hiện tại

Không được viết:

> “Patch transfer mạnh sang mọi architecture có NMS.”

Kết quả hiện tại chỉ cho thấy:

- YOLOv8n: effect đáng chú ý.
- Faster R-CNN: gần như không đổi.
- RetinaNet: patched nhanh hơn trong run này.
- DETR: gần như không đổi.

### Cách biến “kết quả xấu” thành contribution tốt

Thay vì che đi, phát triển thành:

> **Attack effectiveness is architecture- and post-processing-dependent, rather than universally transferable.**

Đây là insight có giá trị khoa học hơn claim “universal vulnerability”.

---

## 2.3. Multi-seed hiện tại

`outputs/multi_seed/aggregate_stats.json`:

- best fitness mean: `65.073`
- std: `4.666`
- n: `10`
- convergence generation mean: `15.4`
- std: `3.806`

Không saliency:

- best fitness mean: `64.021`
- std: `5.169`
- convergence mean: `14.7`
- std: `2.263`

Random search:

- best fitness mean: `15.179`
- std: `0.508`
- n_seeds: `10`
- evals/seed: `500`

### Vấn đề quan trọng

Hiện tại **Saliency-Guided GA chưa thể hiện superiority rõ ràng so với GA không saliency** theo aggregate numbers đang có.

Do đó không nên viết:

> “Saliency giúp hội tụ nhanh vượt trội.”

Phải chạy lại theo **equal query budget** và statistical testing.

---

## 2.4. Resource logs hiện tại cho thấy machine/config mixing

Một số log hiện có:

### Clean run A

- CPU ~76.57%
- RAM ~13.48 GB
- FPS ~5.55
- raw boxes = 0

### Attack run A

- CPU ~76.71%
- RAM ~13.53 GB
- FPS ~5.29
- raw boxes = 0

### Clean run B

- CPU ~28.68%
- RAM ~13.53 GB
- FPS ~3.53
- raw boxes ~3.0

### Attack run B

- CPU ~19.95%
- RAM ~13.51 GB
- FPS ~3.98
- raw boxes ~3.29

### Kết luận

Các log này cho thấy:

- likely different machine/config/state,
- system-wide RAM đang được ghi,
- CPU% tuyệt đối không đủ đáng tin,
- một số run attack không làm tăng CPU,
- temperature = 0 cho thấy telemetry chưa hoạt động đúng.

**Hành động:**  
Không dùng các log này làm final Q1/Q2 evidence. Giữ lại như exploratory data.

---

# 3. Research Narrative mới

Thay vì paper có narrative:

> “Patch làm CPU lên 100%, gây DoS.”

Đổi thành:

> **Sponge Patch induces adversarial workload amplification in object-detection post-processing under constrained compute budgets.**

Chuỗi cơ chế cần chứng minh:

```text
Patch
  ↓
increases confidence-surviving candidates
  ↓
increases post-processing workload
  ↓
increases NMS / end-to-end latency
  ↓
increases deadline violations
  ↓
degrades service availability
```

Đây là narrative phù hợp hơn cho systems/security reviewer.

---

# 4. Terminology phải sửa

Không dùng “raw boxes” cho mọi thứ.

Chuẩn hóa:

```text
N_raw
    = số prediction candidates model tạo ra trước confidence filtering

N_conf
    = số candidates sống sót sau confidence threshold

N_nms_in
    = số candidates thực sự đi vào NMS

N_final
    = số detections sau NMS
```

Pipeline:

```text
Model Output
   ↓
N_raw
   ↓ confidence filtering
N_conf
   ↓ optional top-K / max_nms
N_nms_in
   ↓ NMS
N_final
```

### Claim nên viết

Không:

> “Patch ép mô hình sinh ra 3000 raw boxes.”

Nên:

> “The attack increases the number of confidence-surviving predictions entering the post-processing stage.”

---

# 5. NMS Complexity: sửa lý thuyết cho chặt

Không nói:

> “NMS bắt buộc tính IoU giữa mọi cặp hộp.”

Cách viết tốt hơn:

- Naive greedy NMS có worst-case quadratic behavior.
- Actual latency phụ thuộc implementation, vectorization, top-K, score distribution, overlap distribution và hardware.

Mô hình:

\[
T_{\text{NMS}}
=
f(N_{\text{nms-in}}, D, \tau, \mathcal{I}, H)
\]

Trong đó:

- \(N_{\text{nms-in}}\): số candidates đi vào NMS.
- \(D\): spatial overlap distribution.
- \(\tau\): IoU threshold.
- \(\mathcal{I}\): NMS implementation.
- \(H\): hardware/runtime.

Thực nghiệm cần fit:

\[
T_{\text{NMS}} \approx aN^\alpha + b
\]

và **đo \(\alpha\) thật**, không mặc định \(\alpha = 2\).

---

# 6. Threat Model mới

## 6.1. Không gọi “strict black-box” nếu có telemetry

Threat model nên là:

> **Gray-box / score-based black-box with limited internal telemetry.**

Bảng attacker knowledge:

| Information | Attacker access |
|---|---|
| Model weights | No |
| Exact gradients | No |
| Full architecture | No / Partial |
| Confidence scores | Yes |
| Pre-NMS candidate count | Yes |
| Post-NMS detections | Optional |
| Runtime latency | Yes |
| CPU/RAM telemetry | Optional |
| Query access | Yes |
| Stream modification | Digital setting only |
| Physical FoV access | Physical setting |

## 6.2. Giải thích tại sao attacker có telemetry

Các scenario hợp lý:

- compromised observability API,
- exposed debugging endpoint,
- shared monitoring service,
- edge management telemetry,
- side-channel-like latency feedback.

Không được để reviewer hỏi:

> “Nếu attacker không biết model thì tại sao lại đọc được pre-NMS anchors?”

Phải formalize ngay từ đầu.

---

# 7. Tách rõ 3 môi trường thực nghiệm

## Layer A — Laptop Development

Mục đích:

- code,
- debug,
- optimizer,
- short runs,
- unit tests,
- patch generation.

**Không dùng CPU% tuyệt đối trên laptop làm final claim.**

---

## Layer B — Controlled VM Benchmark

Đây là nơi chạy:

- multi-seed,
- optimizer comparisons,
- resource scaling,
- transferability,
- defense,
- microbenchmark,
- statistics.

VM chỉ đại diện cho **compute budget**, không giả lập microarchitecture.

Ví dụ:

| Profile | vCPU | RAM | Vai trò |
|---|---:|---:|---|
| VM-S | 1 | 2 GB | highly constrained |
| VM-M | 2 | 4 GB | low-end edge budget |
| VM-L | 4 | 8 GB | moderate edge budget |
| VM-XL | 8 | 16 GB | reference |

Claim:

> “We emulate constrained CPU budgets using isolated virtual machines.”

Không claim:

> “VM emulates Intel i5 exactly.”

---

## Layer C — Real Legacy Edge Server

Máy i5 cũ chỉ dùng để:

- final confirmatory experiments,
- clean vs attack,
- defense validation,
- possibly physical test.

Không cần train GA trên máy này.

### Workflow

```text
Laptop development
  ↓
freeze Git commit
  ↓
VM benchmark complete
  ↓
tag release
  ↓
power on old i5
  ↓
git pull exact tag
  ↓
run automated edge suite
  ↓
collect logs
  ↓
shutdown
```

SSH có rớt cũng không làm mất experiment nếu dùng:

```bash
tmux
```

hoặc:

```bash
nohup ./run_edge_suite.sh > edge_run.log 2>&1 &
```

---

# 8. Reproducible Experiment Engine

Nên tạo cấu trúc mới:

```text
configs/
    vm_s.yaml
    vm_m.yaml
    vm_l.yaml
    vm_xl.yaml
    edge_i5.yaml

runner/
    run_experiment.py
    run_suite.py
    collect_metadata.py

analysis/
    aggregate.py
    statistics.py
    plot_latency.py
    plot_resource_scaling.py
    plot_transferability.py
    plot_defense_tradeoff.py

results/
    run_YYYYMMDD_HHMMSS_<uuid>/
        manifest.json
        metrics.csv
        summary.json
        stdout.log
```

---

# 9. Manifest bắt buộc cho mọi run

Mỗi run tự ghi:

```json
{
  "run_id": "2026-08-07T12-30-00Z_ab12cd",
  "git_commit": "abc123...",
  "git_dirty": false,
  "scenario": "attack",
  "seed": 42,
  "hostname": "vm-edge-m",
  "environment": "vm",
  "cpu_model": "...",
  "physical_cores": 4,
  "logical_cores": 4,
  "ram_total_mb": 8192,
  "os": "Ubuntu ...",
  "python": "...",
  "torch": "...",
  "torchvision": "...",
  "opencv": "...",
  "model_name": "yolov8n",
  "model_hash": "...",
  "patch_hash": "...",
  "input_resolution": [640, 640],
  "confidence_threshold": 0.01,
  "iou_threshold": 0.45,
  "max_det": 300,
  "omp_num_threads": 2,
  "mkl_num_threads": 2,
  "cpu_affinity": [0, 1],
  "warmup_frames": 50,
  "measurement_frames": 500,
  "notes": ""
}
```

### Rule

Nếu run thiếu manifest → không được dùng trong paper.

---

# 10. Metrics mới: tránh phụ thuộc hardware tuyệt đối

## 10.1. End-to-End Latency Amplification Factor

\[
LAF
=
\frac{T_{\text{attack}}}
{T_{\text{clean}}}
\]

Ưu tiên median hoặc p95.

---

## 10.2. NMS Amplification Factor

\[
NAF
=
\frac{T_{\text{NMS,attack}}}
{T_{\text{NMS,clean}}}
\]

---

## 10.3. Candidate Amplification Factor

\[
CAF
=
\frac{N_{\text{nms-in,attack}}}
{N_{\text{nms-in,clean}}}
\]

---

## 10.4. FPS Degradation Ratio

\[
FDR
=
1 -
\frac{FPS_{\text{attack}}}
{FPS_{\text{clean}}}
\]

---

## 10.5. QoS Violation Rate

Nếu deadline là \(D\):

\[
QVR
=
\frac{\#\{T_{e2e} > D\}}
{N}
\]

Ví dụ nếu system requirement là 10 FPS:

\[
D = 100 ms
\]

Đây là metric quan trọng hơn CPU 100%.

---

## 10.6. Process Memory Amplification

Dùng process RSS:

\[
MAF
=
\frac{RSS_{\text{attack}}}
{RSS_{\text{clean}}}
\]

Không dùng system-wide RAM làm bằng chứng chính.

---

## 10.7. Query Efficiency

\[
QE(Q)
=
\max_{q \le Q}
Fitness(q)
\]

So optimizer dưới **cùng query budget**.

---

# 11. Resource Monitoring phải sửa

Đo riêng:

```text
Process CPU%
System CPU%
Per-core CPU%
Process RSS
System RAM
CPU frequency
CPU temperature
Load average
```

Nếu telemetry không đọc được:

```text
Temperature_C = null
```

Không ghi `0.0` vì reviewer có thể hiểu là nhiệt độ thật.

---

# 12. Statistical Protocol

## 12.1. Unit of analysis

Không coi từng frame là independent sample.

Ví dụ:

```text
Trial 1 = 500 frames
Trial 2 = 500 frames
...
Trial 10 = 500 frames
```

Và:

\[
n = 10 \text{ independent trials}
\]

không phải `n = 5000 frames`.

---

## 12.2. Paired randomized design

Không chạy:

```text
Clean x10
Attack x10
```

Nên random order:

```text
C A
A C
C A
A C
...
```

hoặc Latin-style counterbalancing.

Mục tiêu:

- giảm thermal bias,
- giảm background-load bias,
- giảm time-order bias.

---

## 12.3. Warm-up

Trước mỗi measured trial:

```text
warmup = 30–100 frames
```

Không đưa warmup vào thống kê.

---

## 12.4. Report

Mỗi metric nên có:

- n independent trials,
- median,
- IQR,
- mean ± SD nếu phù hợp,
- p95 latency,
- 95% bootstrap CI,
- paired effect size.

Nếu distribution lệch:

- Wilcoxon signed-rank cho paired design.

Nếu cần nhiều conditions:

- Friedman test / aligned rank approach,
- multiple-comparison correction.

Không chạy significance test trên từng frame liên tiếp.

---

# 13. Experiment Matrix — Q2 Minimum

## E0 — Environment Sanity Check

Mục tiêu:

- xác nhận deterministic configuration,
- check thread count,
- check CPU affinity,
- check warmup,
- check model checksum,
- check logging.

Pass criteria:

```text
same config + same seed
→ comparable clean latency distribution
```

---

## E1 — NMS Microbenchmark

Tạo synthetic candidate sets:

```text
N = 10
50
100
250
500
1000
2000
4000
8000
10000
```

Đo:

```text
NMS latency
CPU time
RSS delta
```

Lặp:

```text
n >= 30 calls per N
```

Fit:

\[
T(N)=aN^\alpha+b
\]

Không kết luận `O(N²)` chỉ từ lý thuyết; đo exponent thật.

---

## E2 — Clean vs Attack on VM

Cho mỗi VM profile:

```text
Clean
Random patch
Gaussian noise
Checkerboard
Proposed Sponge Patch
```

Mỗi condition:

```text
10 independent trials
```

Đo:

- N_raw
- N_conf
- N_nms_in
- N_final
- preproc latency
- forward latency
- confidence-filter latency
- NMS latency
- total latency
- FPS
- process CPU
- process RSS
- deadline violation rate

---

## E3 — Resource Scaling

VM profiles:

```text
1 vCPU / 2 GB
2 vCPU / 4 GB
4 vCPU / 8 GB
8 vCPU / 16 GB
```

Research question:

> Does attack amplification grow as compute budget decreases?

Main plot:

```text
x = compute budget
y = LAF / NAF / QVR
```

Đây có thể trở thành một contribution systems-oriented rất tốt.

---

## E4 — Optimizer Query Efficiency

Methods:

```text
Random Search
Standard GA
Saliency-Guided GA
PSO or CMA-ES
Optional: NES / Bayesian optimization
```

Bắt buộc equal budget:

```text
Q = 500
Q = 1000
Q = 2500
Q = 5000
Q = 10000
```

Report:

```text
fitness vs queries
CAF vs queries
NAF vs queries
wall-clock chỉ là secondary metric
```

Nếu Saliency-Guided GA không thắng rõ:

- không overclaim,
- đổi contribution từ optimizer superiority sang system characterization.

---

## E5 — Patch Size Ablation

Patch area:

```text
1%
2%
4%
8%
16%
```

Report:

- attack effectiveness,
- query efficiency,
- physical visibility,
- robustness.

Không chỉ report best fitness.

---

## E6 — Cross-Architecture Evaluation

Minimum:

```text
YOLOv5n
YOLOv8n
SSD/MobileNet
Faster R-CNN
RetinaNet
DETR or RT-DETR
```

Group by post-processing:

```text
NMS-based
NMS-free
```

### Negative control

DETR/RT-DETR rất quan trọng.

Nếu effect gần như biến mất ở NMS-free model, đó là evidence hỗ trợ NMS hypothesis.

Nếu effect vẫn còn mạnh, cần kiểm tra forward-path amplification.

---

## E7 — Compression / Video Pipeline

Test:

```text
raw frame
MJPEG/JPEG quality levels
H.264 if feasible
different resize pipelines
```

Mục tiêu:

- xem high-frequency patch có bị codec phá hủy không.

---

## E8 — Defense Evaluation

Minimum defense candidates:

```text
pre-NMS top-K
adaptive confidence threshold
candidate-count cap
latency watchdog
```

Đo cả:

```text
attack latency reduction
mAP/precision/recall impact
false positive/false negative effect
```

Không chỉ đo “attack bị yếu”.

Main trade-off:

\[
\text{Security gain}
\quad vs \quad
\Delta Accuracy
\]

---

## E9 — Real Old-i5 Confirmatory Validation

Chỉ chạy sau khi VM pipeline freeze.

Conditions:

```text
Clean
Random patch
Proposed patch
Proposed patch + defense
```

Minimum:

```text
10 paired independent trials
```

Không train optimizer trên máy i5.

Điều này còn tạo setup đẹp:

```text
Optimize elsewhere
→ deploy on constrained victim
```

---

# 14. Physical Evaluation

Nếu chỉ đủ điều kiện cơ bản:

Gọi là:

> **Preliminary Physical Feasibility Evaluation**

Không gọi:

> robust physical-world attack.

Minimum matrix nên hướng tới:

| Variable | Levels |
|---|---|
| Distance | 0.5 / 1 / 2 / 3 m |
| Angle | 0 / 15 / 30 / 45 / 60° |
| Illumination | low / medium / office / bright |
| Patch scale | 2 / 4 / 8 / 16% FoV |
| Material | matte paper / screen display |

Nếu chưa có lux meter:

- ghi rõ approximate lighting category,
- hoặc dùng sensor/lux app như auxiliary measurement,
- không claim chính xác 500 lux nếu chưa đo.

Mỗi condition:

```text
n >= 5 independent trials
```

Q1 physical-heavy paper nên nhiều hơn.

---

# 15. Baselines bắt buộc

Không chỉ so với “clean”.

## Input baselines

```text
Random patch
Gaussian noise
Checkerboard
Solid color
Natural texture patch
```

## Optimization baselines

```text
Random Search
Standard GA
PSO/CMA-ES
```

## Research attack baselines

Nên tái hiện hoặc so theo protocol với:

```text
Sponge Examples
Phantom Sponges
Overload
SlowTrack-style latency objective
```

Không nhất thiết mọi baseline đều cùng threat model, nhưng phải giải thích fairness.

---

# 16. Defense Section: cơ hội nâng bài mạnh

Một paper attack-only thường bị hỏi:

> “What can a system designer do?”

Nên biến defense thành một section chính.

Ví dụ:

## Defense A — Pre-NMS Top-K

Giới hạn:

\[
N_{\text{nms-in}} \le K
\]

Test:

```text
K = 50
100
200
300
500
1000
```

Đo:

- latency,
- mAP/recall,
- attack survival.

---

## Defense B — Adaptive Confidence Threshold

Nếu candidate count tăng bất thường:

\[
N_{\text{conf}} > \theta_N
\]

thì tăng confidence threshold.

Đánh giá:

- attack mitigation,
- legitimate crowded scenes,
- recall degradation.

---

## Defense C — Candidate Budget Watchdog

Runtime monitor:

```text
if N_conf > threshold:
    cap / drop / degrade gracefully
```

Quan trọng:

- test false alarms trên crowded benign images.

---

# 17. Causal NMS Study — experiment có giá trị cao

Đây là một trong các thí nghiệm nên ưu tiên nhất.

Mục tiêu:

> Không chỉ nói attack làm chậm hệ thống, mà chứng minh **vì sao**.

Chuỗi bằng chứng:

```text
Attack strength
  ↓
N_nms_in increases
  ↓
T_NMS increases
  ↓
T_total increases
  ↓
QoS violations increase
```

Phân tích:

- Spearman correlation giữa `N_nms_in` và `T_NMS`,
- partial correlation nếu cần,
- regression/log-log slope,
- compare NMS-based vs NMS-free models.

Main plots:

```text
N_nms_in vs T_NMS
CAF vs NAF
NAF vs QVR
```

Đây mạnh hơn biểu đồ CPU đơn thuần.

---

# 18. CPU 100% không phải mục tiêu chính

Không đặt success criterion là:

```text
CPU == 100%
```

DoS/availability nên định nghĩa bằng service-level objective.

Ví dụ:

```text
Target ≥ 10 FPS
```

thì:

\[
D = 100 ms
\]

Attack thành công nếu:

```text
p95 latency > deadline
```

hoặc:

```text
QoS violation rate increases materially
```

Đây là formulation gần với systems/security hơn.

---

# 19. Laptop vẫn dùng được thế nào?

Laptop dùng tốt cho:

```text
optimizer
patch generation
unit tests
model download/cache
debug
short benchmark
VM hosting
analysis
plots
```

Nhưng hạn chế:

- turbo boost,
- heterogeneous cores,
- thermal throttling,
- background workloads.

Do đó final comparisons trên laptop phải:

- fixed power mode nếu có,
- fixed affinity,
- fixed thread count,
- report CPU frequency,
- report temperature nếu có,
- dùng normalized metrics.

---

# 20. Cách dùng máy edge cũ mà không phụ thuộc agent/SSH liên tục

## Recommended

```bash
tmux new -s edge_eval
```

Sau đó:

```bash
python runner/run_suite.py --config configs/edge_i5.yaml
```

Detach:

```text
Ctrl+B
D
```

Hoặc:

```bash
nohup python runner/run_suite.py \
  --config configs/edge_i5.yaml \
  > edge_run.log 2>&1 &
```

Sau khi xong:

```bash
tar -czf edge_results.tar.gz results/
```

Chỉ cần SSH lại để kéo file.

---

# 21. Data Integrity Rules

## Rule 1

```text
NO RAW LOG = NO RESULT IN PAPER
```

## Rule 2

Mọi table/figure phải sinh tự động:

```text
raw logs
  ↓
analysis script
  ↓
CSV summary
  ↓
figure/table
```

Không nhập tay.

## Rule 3

Không trộn:

```text
REAL-HW
VM
SIMULATION
```

trong cùng claim nếu không label.

## Rule 4

Mọi run phải có:

```text
manifest.json
metrics.csv
summary.json
stdout.log
```

## Rule 5

Không overwrite results.

Mỗi run có immutable ID.

## Rule 6

Paper phải ghi đúng số liệu từ frozen artifact.

---

# 22. Suggested Result Directory

```text
results/
  vm/
    vm_s/
      yolov8n/
        clean/
        attack/
        defense/
    vm_m/
    vm_l/
    vm_xl/

  real_edge/
    intel_i5/
      clean/
      attack/
      defense/

  physical/
    camera_01/
      distance_0.5m/
      distance_1m/
      distance_2m/

  optimizer/
    random/
    ga/
    saliency_ga/
    cmaes/

  microbenchmark/
    nms/
```

---

# 23. Research Questions mới

## RQ1 — Workload Amplification

> Can a localized adversarial patch increase the number of confidence-surviving candidates entering object-detection post-processing?

Metrics:

```text
CAF
N_conf
N_nms_in
```

---

## RQ2 — Mechanism

> To what extent does candidate amplification translate into NMS and end-to-end latency amplification?

Metrics:

```text
NAF
LAF
correlation/regression
```

---

## RQ3 — Compute-Budget Sensitivity

> Are resource-constrained deployments disproportionately affected by the same adversarial workload?

Metrics:

```text
LAF across 1/2/4/8 vCPU
QVR across budgets
```

---

## RQ4 — Query Efficiency

> Does saliency guidance improve attack effectiveness under equal query budgets?

Metrics:

```text
fitness(Q)
CAF(Q)
NAF(Q)
```

---

## RQ5 — Cross-Architecture Generalization

> Is the attack architecture-specific, NMS-specific, or broadly transferable?

Metrics:

```text
model-wise LAF
NMS vs NMS-free comparison
```

---

## RQ6 — Physical Feasibility

> How much effectiveness survives camera capture, scaling, angle, lighting and compression?

Metrics:

```text
physical/digital ratio
LAF
CAF
QVR
```

---

## RQ7 — Defense Trade-off

> Can lightweight pre-NMS safeguards reduce latency amplification without unacceptable detection degradation?

Metrics:

```text
mitigation %
ΔmAP
ΔRecall
```

---

# 24. Contribution Statement đề xuất

Không viết:

> “We propose a completely novel revolutionary attack.”

Nên hướng tới:

1. **A gray-box adversarial workload amplification framework** for resource-constrained object detectors.
2. **A controlled compute-budget characterization** across virtualized edge profiles and a physical legacy edge server.
3. **A mechanistic study** linking candidate amplification to NMS/post-processing latency.
4. **A query-efficiency analysis** of saliency-guided gradient-free optimization.
5. **A defense trade-off evaluation** across lightweight candidate-limiting strategies.
6. **A preliminary physical-world feasibility study**, nếu physical data chưa đủ rộng.

---

# 25. Title đề xuất

Nếu physical chưa mạnh:

> **SpongePatch: Adversarial Workload Amplification Against Resource-Constrained Edge Object Detectors**

Nếu query-efficiency mạnh:

> **SpongePatch: Query-Efficient Adversarial Workload Amplification for Edge Object Detectors**

Nếu physical đủ mạnh:

> **SpongePatch: Query-Efficient Physical Resource-Exhaustion Attacks on Edge Object Detectors**

Không nên để “physical” trong title nếu chỉ có một proof-of-concept nhỏ.

---

# 26. Paper Structure đề xuất

```text
1. Introduction
2. Background and Related Work
3. Threat Model and Problem Formulation
4. SpongePatch Method
   4.1 Candidate-amplification objective
   4.2 Saliency prior
   4.3 Gradient-free optimizer
   4.4 EOT
5. Experimental Methodology
   5.1 Models
   5.2 VM profiles
   5.3 Real edge server
   5.4 Metrics
   5.5 Statistical protocol
6. Results
   6.1 NMS microbenchmark
   6.2 Clean vs attack
   6.3 Resource scaling
   6.4 Query efficiency
   6.5 Cross-architecture transfer
   6.6 Physical feasibility
7. Defense Evaluation
8. Discussion
9. Limitations and Ethics
10. Conclusion
```

---

# 27. Claims phải hạ đúng mức

## Không nên

- “mang tính đột phá”
- “tử huyệt”
- “bóp nghẹt hoàn toàn”
- “chính thức chứng minh”
- “systemic flaw of the entire industry”
- “universally transferable”
- “robust physical attack” nếu chưa test đủ.

## Nên

- “we observe”
- “our results suggest”
- “under the tested configurations”
- “architecture-dependent”
- “preliminary physical feasibility”
- “consistent with a post-processing bottleneck”
- “requires broader validation”

---

# 28. Q2 Minimum Checklist

## Methodology

- [ ] Threat model consistent.
- [ ] N_raw/N_conf/N_nms_in/N_final separated.
- [ ] Exact objective clearly defined.
- [ ] Query budget explicit.

## Experiments

- [ ] 3+ VM profiles.
- [ ] 4+ detector architectures.
- [ ] 1 NMS-free negative control.
- [ ] 3+ attack/input baselines.
- [ ] 10 independent seeds/runs.
- [ ] NMS microbenchmark.
- [ ] Defense evaluation.
- [ ] Real old-i5 confirmation.
- [ ] Physical PoC clearly labeled.

## Statistics

- [ ] Paired randomized design.
- [ ] Warmup excluded.
- [ ] Median/IQR.
- [ ] p95 latency.
- [ ] 95% bootstrap CI.
- [ ] Paired significance test/effect size.

## Reproducibility

- [ ] Frozen git commit.
- [ ] Manifest per run.
- [ ] Model/patch hash.
- [ ] No manual table entry.
- [ ] Raw logs archived.

---

# 29. Q1 Target Checklist

Ngoài toàn bộ Q2 checklist:

- [ ] Strong query-efficiency contribution.
- [ ] PSO/CMA-ES or comparable optimizer baseline.
- [ ] Overload/Phantom-Sponges/SlowTrack-class comparison.
- [ ] NMS causal characterization.
- [ ] Compute-budget scaling study.
- [ ] Defense–accuracy trade-off.
- [ ] Multiple NMS implementations or NMS variants.
- [ ] NMS-free detector control.
- [ ] More comprehensive physical matrix **hoặc** stronger systems contribution.
- [ ] Reproducibility package runnable end-to-end.

---

# 30. Reviewer Questions cần chuẩn bị trước

## “Why use a patch if digital MITM already controls the stream?”

Trả lời bằng design:

- digital injection = upper-bound / optimization environment,
- physical patch = restricted deployment vector,
- focus = workload amplification mechanism, not MITM itself.

---

## “Why call it gray-box?”

Vì attacker không có weights/gradients nhưng có limited telemetry or score feedback.

---

## “Is NMS always O(N²)?”

Không claim tuyệt đối. Dùng empirical microbenchmark và implementation-specific characterization.

---

## “Why not just increase confidence threshold?”

Defense trade-off phải chứng minh cost về recall/mAP hoặc crowded-scene robustness.

---

## “Could CPU increase be thermal noise?”

Dùng paired randomized trials, temperature/frequency logs, normalized latency metrics.

---

## “Does it generalize?”

Không hứa universal. Báo đúng architecture-dependent behavior.

---

## “Why use VM?”

VM dùng để control compute budget và reproducibility. Real legacy edge server dùng để confirm external validity.

---

# 31. Priority Order — việc cần làm theo thứ tự

## Phase 1 — Clean the evidence

1. Dừng chỉnh số trong paper.
2. Label outputs hiện tại:
   - exploratory,
   - simulated,
   - real.
3. Không dùng các log chưa xác định machine/config.
4. Chuẩn hóa terminology.

---

## Phase 2 — Build Experiment Engine

1. `run_experiment.py`
2. `manifest.json`
3. `metrics.csv`
4. `summary.json`
5. auto plots/tables.

---

## Phase 3 — VM benchmark

1. VM-S/M/L.
2. NMS microbenchmark.
3. Clean vs attack.
4. Baselines.
5. Resource scaling.
6. Multi-seed.

---

## Phase 4 — Architecture + Defense

1. YOLO family.
2. SSD/Faster R-CNN/RetinaNet.
3. DETR/RT-DETR.
4. Defense trade-off.

---

## Phase 5 — Real Edge Validation

1. Freeze git tag.
2. Power on old i5.
3. Run automated suite.
4. Collect immutable logs.
5. Shut down.

---

## Phase 6 — Physical

1. Printed patch.
2. Angle/distance/scale.
3. Compression/camera.
4. Report domain gap.

---

## Phase 7 — Rewrite paper

Chỉ sau khi dataset mới complete.

---

# 32. Kill Criteria / Pivot Rules

Để tránh “ép số đẹp”, đặt sẵn rule:

## Nếu Saliency-Guided GA không thắng Standard GA

Không fake advantage.

Pivot contribution sang:

> constrained workload characterization.

---

## Nếu attack chỉ hiệu quả trên YOLO

Không gọi universal.

Pivot:

> YOLO/post-processing-specific vulnerability characterization.

---

## Nếu NMS-free model cũng chậm mạnh

Kiểm tra:

- forward path,
- memory behavior,
- preprocessing,
- codec,
- patch-induced convolutional activation.

Có thể mở ra contribution mới.

---

## Nếu physical effect yếu

Không cố claim robust physical attack.

Giữ:

> preliminary feasibility / domain-gap study.

---

## Nếu old i5 không reproduce VM effect

Đây vẫn là result có giá trị:

> virtualization-based compute budgets do not fully predict microarchitectural behavior.

Phân tích nguyên nhân.

---

# 33. Suggested Automation Commands

Ví dụ target workflow:

```bash
python runner/run_suite.py \
  --config configs/vm_m.yaml \
  --suite clean_attack_baselines \
  --trials 10 \
  --seed-base 1000
```

Optimizer:

```bash
python runner/run_suite.py \
  --config configs/vm_m.yaml \
  --suite optimizer_query_budget \
  --budgets 500 1000 2500 5000 \
  --trials 10
```

Edge:

```bash
nohup python runner/run_suite.py \
  --config configs/edge_i5.yaml \
  --suite final_edge_validation \
  --trials 10 \
  > edge_run.log 2>&1 &
```

---

# 34. Recommended Frozen Artifacts Before Submission

```text
release/
  code/
  configs/
  patches/
  models_manifest.json
  raw_results/
  processed_results/
  figures/
  tables/
  environment.yml
  requirements-lock.txt
  REPRODUCE.md
```

Paper tables phải có mapping:

```text
Table 2
→ analysis/table2.py
→ processed_results/table2.csv
→ raw_results/<run_ids>
```

---

# 35. Literature Baseline / Reviewer Context

Các công trình nên được xem như minimum related-work backbone:

1. *Brown, T. B., Mané, D., Roy, A., Abadi, M., & Gilmer, J. (2017). Adversarial Patch. arXiv.*  
   https://arxiv.org/abs/1712.09665

2. *Athalye, A., Engstrom, L., Ilyas, A., & Kwok, K. (2018). Synthesizing Robust Adversarial Examples. ICML.*  
   http://proceedings.mlr.press/v80/athalye18b.html

3. *Shumailov, I., Zhao, Y., Bates, D., et al. (2021). Sponge Examples: Energy-Latency Attacks on Neural Networks. IEEE European Symposium on Security and Privacy.*  
   https://ieeexplore.ieee.org/abstract/document/9581273/

4. *Shapira, A., Zolfi, A., Demetrio, L., Biggio, B., et al. (2023). Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors. IEEE/CVF WACV.*  
   https://ieeexplore.ieee.org/abstract/document/10030526/

5. *Chen, E. C., Chen, P.-Y., Chung, I., et al. (2024). Overload: Latency Attacks on Object Detection for Edge Devices. CVPR.*  
   https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Overload_Latency_Attacks_on_Object_Detection_for_Edge_Devices_CVPR_2024_paper.html

6. *Ma, C., Wang, N., Chen, Q. A., & Shen, C. (2024). SlowTrack: Increasing the Latency of Camera-Based Perception in Autonomous Driving Using Adversarial Examples. AAAI.*  
   https://ojs.aaai.org/index.php/AAAI/article/view/28200

7. *Chen, E. C., Chen, P.-Y., Chung, I., et al. (2024). Latency Attack Resilience in Object Detectors: Insights from Computing Architecture. ACCV.*  
   https://openaccess.thecvf.com/content/ACCV2024/html/Chen_Latency_Attack_Resilience_in_Object_Detectors_Insights_from_Computing_Architecture_ACCV_2024_paper.html

8. *Wang, T., Wang, Z., Wang, C., Shu, Y., et al. (2025). Can't Slow Me Down: Learning Robust and Hardware-Adaptive Object Detectors Against Latency Attacks for Edge Devices. CVPR.*  
   https://ieeexplore.ieee.org/abstract/document/11094991/

9. *Rathnasuriya, R., Li, T., Xu, Z., Song, Z., Haque, M., et al. (2025). SoK: Efficiency Robustness of Dynamic Deep Learning Systems. USENIX Security.*  
   https://www.usenix.org/conference/usenixsecurity25/presentation/rathnasuriya

### Ý nghĩa cho bài hiện tại

- Brown/EOT: physical patch robustness foundation.
- Sponge Examples: availability/resource-exhaustion foundation.
- Phantom Sponges: NMS-targeted latency attacks.
- Overload/SlowTrack: strong modern latency attack baselines.
- ACCV 2024: hardware/runtime perspective.
- CVPR 2025 defense: reviewer benchmark cho defense/hardware adaptivity.
- SoK 2025: broad efficiency robustness context.

---

# 36. Final Research Positioning

## Bản hiện tại

> “A Sponge Patch raises CPU and lowers FPS on an old edge server.”

Điểm yếu:

- hardware-specific,
- dễ bị reviewer hỏi về measurement noise,
- dễ overclaim,
- novelty chưa rõ.

## Bản Q2 nên hướng tới

> **“A controlled characterization of adversarial workload amplification in NMS-based edge object detectors under constrained compute budgets, with real-edge confirmation and lightweight defenses.”**

## Bản Q1 nên hướng tới

> **“A query-efficient adversarial workload amplification framework with mechanistic NMS characterization, compute-budget sensitivity analysis, cross-architecture evaluation, and defense–accuracy trade-offs.”**

---

# 37. Quy tắc cuối cùng

1. **Không sửa số thủ công để khớp narrative.**
2. **Không dùng output simulation như real-hardware evidence.**
3. **Không overclaim transferability.**
4. **Không dùng CPU 100% làm định nghĩa duy nhất của DoS.**
5. **Không gọi NMS `O(N²)` tuyệt đối nếu chưa nêu worst-case/implementation assumptions.**
6. **Không dùng “raw boxes” khi thực chất là post-confidence candidates.**
7. **Không coi frame liên tiếp là independent samples.**
8. **Không dùng log thiếu metadata.**
9. **Không viết Results trước khi frozen dataset hoàn tất.**
10. **Mọi claim phải trace được về một hoặc nhiều run IDs.**

---

# 38. One-Line Project Rule

```text
Develop on laptop → benchmark in controlled VMs → confirm on real old edge hardware → generate paper tables/figures automatically from immutable logs.
```

---

# 39. Definition of Done

Bài được coi là sẵn sàng để viết lại final version khi:

```text
[ ] VM benchmark complete
[ ] real-edge confirmation complete
[ ] all runs have manifests
[ ] all metrics generated automatically
[ ] no paper number is manually entered
[ ] NMS mechanism validated empirically
[ ] optimizer comparison uses equal query budgets
[ ] cross-architecture behavior reported honestly
[ ] defense trade-off measured
[ ] physical claim matches actual evidence level
[ ] statistics are trial-level, not frame-level
[ ] figures/tables trace back to raw logs
```

Khi toàn bộ checklist này pass, mới freeze dataset và viết lại Results/Discussion/Abstract/Conclusion.

---

## Chốt hướng chiến lược

Máy i5 cũ **không phải trở ngại cần loại bỏ**; nó nên được dùng như một **external validation node**. Laptop đảm nhiệm development và optimization. VM đảm nhiệm reproducibility, resource scaling, multi-seed và benchmarking. Máy i5 đảm nhiệm final reality check.

Điểm “bứt phá” không nằm ở việc cố chứng minh CPU lên 100%. Điểm mạnh hơn là chứng minh một cách có cơ chế:

\[
\boxed{
\text{Patch}
\rightarrow
\text{Candidate Amplification}
\rightarrow
\text{Post-processing Amplification}
\rightarrow
\text{QoS Violation}
}
\]

và chỉ ra:

- khi nào attack mạnh,
- khi nào attack yếu,
- architecture nào dễ tổn thương,
- compute budget ảnh hưởng thế nào,
- defense nào hiệu quả,
- accuracy phải trả giá bao nhiêu.

Đó là hướng giúp dự án từ một demo tấn công trở thành một nghiên cứu systems/security đủ độ sâu cho Q2 và có nền để nhắm Q1.
