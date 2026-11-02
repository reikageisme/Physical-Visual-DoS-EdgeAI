@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Project-local .venv is missing. Run setup_env.bat first.
  exit /b 1
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1

set "YOLO_CONFIG_DIR=%CD%\.config"
set "TORCH_HOME=%CD%\.cache\torch"
set "MPLCONFIGDIR=%CD%\.cache\matplotlib"
set "XDG_CACHE_HOME=%CD%\.cache"
set "HF_HOME=%CD%\dataset\.cache\huggingface"
set "HF_HUB_CACHE=%CD%\dataset\.cache\huggingface\hub"
set "HF_DATASETS_CACHE=%CD%\dataset\.cache\huggingface\datasets"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
set "TMP=%CD%\.cache\tmp"
set "TEMP=%CD%\.cache\tmp"

".venv\Scripts\python.exe" audit_derived_dataset.py --data dataset/adv_tshirt_patch_detection.yaml --output results/adv_tshirt_patch_detection_audit.json
if errorlevel 1 exit /b 1

for %%S in (n s l) do (
  for %%M in (yolov8 yolov10 yolo11 yolo12) do (
    if not exist "runs\train\advpatch-%%M%%S\weights\best.pt" (
      ".venv\Scripts\python.exe" train_baselines.py --config configs/adv_tshirt_patch_experiment.yaml --data dataset/adv_tshirt_patch_detection.yaml --scale %%S --models %%M
      if errorlevel 1 exit /b 1
    )
  )
)

if not exist "runs\train\advpatch-yolov13n\weights\best.pt" (
  ".venv\Scripts\python.exe" train_ablation.py --config configs/adv_tshirt_patch_experiment.yaml --data dataset/adv_tshirt_patch_detection.yaml --scale n --variant full --name advpatch-yolov13n --output results/adv_tshirt_patch_ablation.csv
  if errorlevel 1 exit /b 1
)

for %%S in (s l) do (
  if not exist "runs\train\advpatch-yolov13%%S\weights\best.pt" (
    ".venv\Scripts\python.exe" train_yolov13.py --config configs/adv_tshirt_patch_experiment.yaml --data dataset/adv_tshirt_patch_detection.yaml --scale %%S --name advpatch-yolov13%%S
    if errorlevel 1 exit /b 1
  )
)

for %%V in (no-hyperace no-fullpad no-dsc3k2) do (
  findstr /I /C:"YOLOv13-n-%%V" "results\adv_tshirt_patch_ablation.csv" >nul
  if errorlevel 1 (
    ".venv\Scripts\python.exe" train_ablation.py --config configs/adv_tshirt_patch_experiment.yaml --data dataset/adv_tshirt_patch_detection.yaml --scale n --variant %%V --name advpatch-yolov13n-%%V --output results/adv_tshirt_patch_ablation.csv
    if errorlevel 1 exit /b 1
  )
)

".venv\Scripts\python.exe" benchmark_eval.py --config configs/adv_tshirt_patch_benchmark.yaml --data dataset/adv_tshirt_patch_detection.yaml --splits val test --devices both --output results/adv_tshirt_patch_benchmark.csv
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" export_results.py --results results/adv_tshirt_patch_benchmark.csv --ablation results/adv_tshirt_patch_ablation.csv --benchmark-config configs/adv_tshirt_patch_benchmark.yaml --out-dir results/adv_tshirt_patch_results
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" build_adv_tshirt_patch_report.py
if errorlevel 1 exit /b 1

echo AdvT-shirt patch-localization benchmark completed. Results are under results/adv_tshirt_patch_results.
endlocal
