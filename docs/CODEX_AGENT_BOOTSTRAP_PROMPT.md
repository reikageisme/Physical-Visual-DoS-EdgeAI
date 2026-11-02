# Codex Bootstrap Prompt — Q1/Q2 Experimental Refactor

> Copy toàn bộ nội dung phần **MASTER PROMPT** bên dưới vào Codex Agent tại root của repository.
> Agent phải đọc `Q1_Q2_RESEARCH_ROADMAP.md` trước khi sửa bất kỳ code nào.

---

# MASTER PROMPT

You are acting as the implementation and reproducibility engineer for this research repository.

Your primary objective is **NOT** to make the reported numbers look better. Your objective is to rebuild the experimental pipeline so that every result used in the paper is traceable, reproducible, correctly labeled, and generated from real execution logs.

## 0. Mandatory research-integrity rules

Treat the following as hard constraints:

1. **NO RAW LOG = NO RESULT IN PAPER.**
2. Never fabricate, synthesize, interpolate, or manually edit experimental results to match the paper.
3. Never copy laptop results and label them as VM or physical-edge results.
4. Keep these environments strictly separated:
   - `DEV-LAPTOP`
   - `VM-BENCHMARK`
   - `REAL-EDGE`
   - `SIMULATION`
5. Existing outputs are historical/exploratory evidence. Do not overwrite them.
6. Every new run must have a unique immutable run ID.
7. Every new run must save:
   - `manifest.json`
   - `metrics.csv`
   - `summary.json`
   - `stdout.log`
8. Every paper table and figure must eventually be generated from raw logs by scripts.
9. If an experiment fails or contradicts the hypothesis, preserve the result and report it.
10. Do not claim:
   - universal transferability,
   - robust physical attack,
   - CPU exhaustion,
   - quadratic NMS behavior,
   unless the measured evidence actually supports that claim.
11. Do not make hidden changes to thresholds, model versions, patch sizes, seeds, or hardware configuration between compared conditions.
12. Clean and attack trials must use matched configurations.
13. Do not execute destructive commands without explicit user approval.
14. Do not delete historical data.
15. Before installing system-level packages, creating VMs, changing BIOS/virtualization settings, or changing host networking, explain exactly what will change and wait for approval.

---

## 1. Read project context before touching code

First, inspect the repository and read completely:

- `Q1_Q2_RESEARCH_ROADMAP.md`
- the current paper/documentation if present
- `README.md`
- `requirements*.txt`
- environment files
- all scripts under:
  - `attack/`
  - `core/`
  - `experiments/`
  - `utils/`
- representative output files under:
  - `outputs/`
  - `logs/`
  - `results/` if it exists

Pay particular attention to:

- model loading,
- confidence filtering,
- NMS timing,
- raw/post-threshold/post-NMS box terminology,
- resource monitoring,
- seed handling,
- patch loading,
- VM/hardware assumptions,
- code that writes CSV/JSON,
- any manually hard-coded result values.

Do not modify anything yet.

Then produce a concise audit with:

1. current repository structure,
2. current experiment entry points,
3. data-quality/reproducibility risks,
4. code paths that appear to mix simulation and real measurement,
5. places where “raw boxes” terminology is ambiguous,
6. current monitoring limitations,
7. the minimum safe refactor plan.

STOP and show the audit to the user before implementing.

---

## 2. Work in phases and stop at checkpoints

You must work incrementally.

Use this order:

### Phase A — Repository hygiene and metadata
### Phase B — Unified experiment runner
### Phase C — Monitoring and metrics
### Phase D — Analysis/statistics
### Phase E — VM infrastructure
### Phase F — VM sanity benchmark
### Phase G — Full VM experiment matrix
### Phase H — Real legacy-edge validation
### Phase I — Physical experiment support
### Phase J — Paper artifact generation

At the end of every phase:

- run tests/sanity checks,
- show changed files,
- explain what was implemented,
- show example output,
- state remaining risks,
- STOP and wait for user confirmation before starting the next phase.

Do not perform the entire project in one autonomous run.

---

# PHASE A — Repository hygiene and metadata

Create or propose the following layout without deleting existing code:

```text
configs/
runner/
analysis/
results/
docs/
tests/
scripts/
```

Recommended configuration files:

```text
configs/
    dev_laptop.yaml
    vm_s.yaml
    vm_m.yaml
    vm_l.yaml
    vm_xl.yaml
    edge_i5.yaml
```

Do not fill hardware-specific values by guessing. Use placeholders or auto-detection where appropriate.

Create a metadata collector that records at least:

```text
run_id
timestamp UTC
git commit
git dirty state
hostname
environment label
OS
kernel
CPU model
physical cores
logical cores
total RAM
Python version
PyTorch version
Torchvision version
OpenCV version
model name
model file hash
patch file hash
seed
input resolution
confidence threshold
IoU threshold
max_det/top-K
OMP_NUM_THREADS
MKL_NUM_THREADS
CPU affinity
warm-up frames
measurement frames
scenario
VM profile
notes
```

If temperature or CPU-frequency telemetry is unavailable, record `null`, not `0`.

Expected run structure:

```text
results/
  <environment>/
    <run_id>/
      manifest.json
      metrics.csv
      summary.json
      stdout.log
```

Historical outputs must remain untouched.

Checkpoint after Phase A.

---

# PHASE B — Unified experiment runner

Implement a single experiment entry point, preferably:

```bash
python runner/run_experiment.py --config configs/<profile>.yaml
```

and a suite-level orchestrator:

```bash
python runner/run_suite.py ...
```

Requirements:

1. no manual result values,
2. deterministic seed setup,
3. warm-up separated from measurement,
4. clean/attack configurations matched,
5. failure status written to manifest,
6. graceful interruption,
7. no result overwrite,
8. timestamps from a monotonic timer for latency,
9. per-stage timing where technically possible.

Design the runner so existing attack/model code is reused rather than rewritten unnecessarily.

Checkpoint after Phase B.

---

# PHASE C — Monitoring and metric definitions

Refactor monitoring so the following are separate:

```text
process_cpu_percent
system_cpu_percent
per_core_cpu_percent
process_rss_mb
system_ram_used_mb
cpu_frequency_mhz
cpu_temperature_c
load_average
```

Do not use system-wide RAM as evidence of model-process memory growth.

Record per-frame/per-sample stages where possible:

```text
preprocess_ms
forward_ms
confidence_filter_ms
nms_ms
postprocess_ms
total_ms
fps
N_raw
N_conf
N_nms_in
N_final
```

If the detector API does not expose one of these directly, do not invent it. Mark unavailable and document the instrumentation needed.

Implement normalized metrics:

```text
LAF = attack_total_latency / clean_total_latency
NAF = attack_nms_latency / clean_nms_latency
CAF = attack_N_nms_in / clean_N_nms_in
FDR = 1 - attack_fps / clean_fps
MAF = attack_process_rss / clean_process_rss
```

Implement configurable QoS deadline:

```text
QoSViolationRate = frames(total_ms > deadline_ms) / measured_frames
```

Checkpoint after Phase C.

---

# PHASE D — Analysis and statistics

Create analysis scripts that read only raw run folders.

Minimum outputs:

```text
analysis/aggregate.py
analysis/statistics.py
analysis/plot_latency.py
analysis/plot_resource_scaling.py
analysis/plot_optimizer_query_efficiency.py
analysis/plot_transferability.py
analysis/plot_defense_tradeoff.py
```

Statistical unit:

- independent trial/run, not individual sequential video frame.

For paired clean-vs-attack trials report:

```text
n
median
IQR
mean
standard deviation
p95
95% bootstrap confidence interval
paired effect size
paired non-parametric test when appropriate
```

Do not automatically declare “significant” only from p-values; report effect size and CI.

Checkpoint after Phase D.

---

# PHASE E — VM infrastructure

Before creating any VM, detect the host OS and available virtualization stack.

## If host is Windows

Prefer in this order:

1. Hyper-V if available and appropriate.
2. VirtualBox if Hyper-V is unavailable/incompatible.
3. Do not silently use WSL2 as the benchmark VM unless the user explicitly accepts it as a different virtualization environment.

## If host is Linux

Prefer:

```text
KVM/QEMU + libvirt
```

## If host is macOS

Detect:

```text
Intel vs Apple Silicon
```

On Apple Silicon, explicitly warn that ARM virtualization and x86 emulation are not equivalent to the legacy x86 target. It can still be used for compute-budget experiments if labeled correctly.

Before installing anything, report:

```text
host OS
CPU architecture
RAM
available disk
virtualization support
available hypervisor
recommended hypervisor
expected limitations
```

STOP for user approval.

After approval, create reproducible VM profiles:

```text
VM-S:  1 vCPU / 2 GB RAM
VM-M:  2 vCPU / 4 GB RAM
VM-L:  4 vCPU / 8 GB RAM
VM-XL: 8 vCPU / 16 GB RAM
```

If the laptop cannot safely host VM-XL, do not create it. Adjust to available hardware and document the change.

Use the same guest OS version for all VMs.

Recommended guest:

```text
Ubuntu LTS minimal/server
```

Use one controlled software stack across profiles.

Record hypervisor name/version in every VM manifest.

Do not describe VM profiles as emulating an exact Intel CPU. They emulate **resource budgets**, not microarchitectures.

Checkpoint after VM creation but before research experiments.

---

# PHASE F — VM sanity benchmark

Before adversarial experiments, perform only sanity tests:

1. VM boots reliably.
2. SSH works.
3. repository checkout is reproducible.
4. Python environment installs.
5. model loads.
6. clean inference works.
7. metadata is correct.
8. CPU/RAM limits match profile.
9. results survive SSH disconnect.
10. repeated clean runs have reasonable variance.

Use `tmux`, `systemd-run`, or `nohup` for long jobs so the experiment is not tied to an interactive SSH session.

Do not run attack experiments until clean variance is understood.

Report clean-run coefficient of variation / distribution for key latency metrics.

Checkpoint after Phase F.

---

# PHASE G — VM experiment matrix

Implement, but run progressively.

## E1 — NMS microbenchmark

Synthetic candidate counts:

```text
10
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

Use the exact NMS implementation used by the victim path when possible.

Measure:

```text
N
NMS latency
CPU time
RSS delta
```

Fit empirically:

```text
T_NMS ≈ a * N^alpha + b
```

Never force `alpha = 2`.

## E2 — Clean vs input baselines vs proposed attack

Conditions:

```text
Clean
Random patch
Gaussian noise
Checkerboard
Solid-color/natural-texture control if available
Proposed Sponge Patch
```

Minimum 10 independent trials per condition.

## E3 — Resource-scaling

Run across available VM-S/M/L/XL profiles.

Main question:

> Does attack amplification increase as compute budget decreases?

## E4 — Query-efficiency optimizer comparison

Compare under exactly equal query budgets:

```text
Random Search
Standard GA
Saliency-Guided GA
PSO or CMA-ES
```

Budgets:

```text
500
1000
2500
5000
10000
```

Primary output:

```text
fitness vs queries
CAF vs queries
NAF vs queries
```

Do not claim saliency superiority if it is not statistically supported.

## E5 — Patch-size ablation

Suggested patch-area ratios:

```text
1%
2%
4%
8%
16%
```

## E6 — Cross-architecture

Target a mix such as:

```text
YOLOv5n
YOLOv8n
SSD/MobileNet
Faster R-CNN
RetinaNet
DETR or RT-DETR
```

Explicitly separate:

```text
NMS-based
NMS-free
```

Treat NMS-free detector as a negative/control condition for the NMS mechanism hypothesis.

## E7 — Video/compression pipeline

When feasible:

```text
raw
JPEG/MJPEG quality levels
H.264
different resize stages
```

## E8 — Defense

At minimum evaluate:

```text
pre-NMS top-K
adaptive confidence threshold
candidate-count cap
latency/candidate watchdog
```

For every defense report both:

```text
attack mitigation
accuracy/recall cost
```

Checkpoint after each experiment family. Do not launch every matrix cell at once.

---

# PHASE H — Real legacy-edge validation

The old Intel i5 machine is a **validation node**, not the development machine.

Do not optimize/train the patch on it unless explicitly required by an experiment.

Workflow:

```text
develop on laptop
→ complete VM tests
→ freeze git commit/tag
→ power on old i5
→ pull exact frozen tag
→ run automated validation suite
→ collect immutable raw logs
→ power off
```

Use detached execution:

```bash
tmux new -s edge_eval
```

or:

```bash
nohup python runner/run_suite.py \
  --config configs/edge_i5.yaml \
  --suite final_edge_validation \
  > edge_run.log 2>&1 &
```

Required final conditions:

```text
Clean
Random/control patch
Proposed patch
Proposed patch + defense
```

Prefer at least 10 paired independent trials if practical.

Never merge real-edge values with VM values without labels.

Checkpoint after raw edge logs are collected.

---

# PHASE I — Physical support

Treat physical evaluation conservatively.

If current evidence is limited, label it:

```text
Preliminary Physical Feasibility Evaluation
```

Support metadata for:

```text
camera model
resolution
codec
distance
angle
illumination measurement/method
patch physical size
patch FoV ratio
print/display medium
trial ID
```

Never claim exact lux if it was not measured.

Suggested future matrix:

```text
distance: 0.5 / 1 / 2 / 3 m
angle: 0 / 15 / 30 / 45 / 60 deg
patch FoV: 2 / 4 / 8 / 16%
lighting: several controlled levels
```

Checkpoint after physical instrumentation.

---

# PHASE J — Paper artifact generation

Only after the new dataset is frozen:

Create scripts such that:

```text
raw run IDs
→ aggregate CSV
→ table
→ figure
```

For every paper table create a provenance mapping, e.g.:

```text
Table 2
→ analysis/build_table2.py
→ processed/table2.csv
→ source run IDs
```

Do not edit paper numbers manually.

---

# 3. Research claims to protect

Preferred research story:

```text
Patch
→ candidate amplification
→ post-processing workload amplification
→ latency amplification
→ QoS/deadline violation
```

The goal is not to maximize CPU percentage.

Use CPU utilization as a secondary systems metric.

Primary availability metrics should be:

```text
end-to-end latency
p95 latency
FPS
QoS violation rate
NMS latency
candidate amplification
```

---

# 4. Coding quality requirements

- Keep changes small and reviewable.
- Add docstrings/types where useful.
- Avoid breaking existing scripts unless necessary.
- Prefer adapters around current model code.
- Add tests for:
  - manifest generation,
  - unique run IDs,
  - no overwrite,
  - seed reproducibility,
  - metric aggregation,
  - clean/attack config matching.
- Never hide failed trials.
- Write errors into run metadata.
- Use UTC timestamps in machine-readable files.
- Use monotonic/performance timers for latency.
- Use SHA-256 for patch/model hashes where practical.

---

# 5. Safety against accidental paper-data corruption

Before changing any historical output:

STOP.

Historical directories are read-only for this refactor unless the user explicitly authorizes migration.

If a cleaner schema is needed:

```text
create new results_v2/
```

or another new directory rather than rewriting old evidence.

---

# 6. First task right now

Do only this:

1. Read the roadmap and repository.
2. Produce the audit requested in Section 1.
3. Propose the exact Phase A file changes.
4. Detect host OS/hardware only if shell access permits.
5. Do **not** install a hypervisor yet.
6. Do **not** create VMs yet.
7. Do **not** run long experiments.
8. Do **not** rewrite the paper.

Then STOP and wait for approval.

# END MASTER PROMPT

---

# Cách sử dụng prompt này

## Bước 1

Đặt file này và roadmap ở repo:

```text
docs/
  Q1_Q2_RESEARCH_ROADMAP.md
  CODEX_AGENT_BOOTSTRAP_PROMPT.md
```

Nếu roadmap hiện ở root thì cũng được; sửa đường dẫn trong prompt cho đúng.

## Bước 2

Mở Codex tại **root repository**.

Prompt ngắn đầu tiên có thể là:

```text
Read docs/CODEX_AGENT_BOOTSTRAP_PROMPT.md and docs/Q1_Q2_RESEARCH_ROADMAP.md completely. Follow the MASTER PROMPT exactly. Start only with the initial repository audit. Do not modify files, install software, create VMs, or run long experiments yet.
```

## Bước 3

Sau khi Codex đưa audit, chỉ khi audit hợp lý mới gửi:

```text
Proceed with Phase A only. Make the repository-hygiene and metadata changes you proposed. Preserve all historical outputs. Run only fast sanity tests. At the end, show changed files and sample manifest output, then stop.
```

## Bước 4

Sau Phase A:

```text
Proceed with Phase B only. Implement the unified experiment runner while reusing existing model/attack code. Do not create VMs yet. Run a very short development sanity test only. Then stop.
```

## Bước 5

Sau runner/monitor/statistics ổn:

```text
Proceed with Phase E only. Detect my laptop OS, architecture, available RAM/disk, and virtualization support. Recommend the most suitable hypervisor and exact VM plan. Do not install or change host settings yet. Give me the commands/GUI steps one at a time and wait for my confirmation after each host-level step.
```

## Bước 6

Khi VM tạo xong:

```text
Proceed with Phase F only. Set up the guest environment, clone/check out the exact repository commit, install the controlled Python dependencies, verify resource limits, verify metadata, and run clean sanity benchmarks only. Do not start attack experiments until clean-run variance is reported.
```

## Bước 7

Chỉ khi clean benchmark ổn:

```text
Proceed with E1 (NMS microbenchmark) only. Generate raw logs and analysis artifacts. Do not run E2-E8 yet. Report whether the measured NMS scaling supports or contradicts the current paper's complexity narrative.
```

Sau đó chạy E2, E3... theo từng experiment family, không giao toàn bộ một lần.
