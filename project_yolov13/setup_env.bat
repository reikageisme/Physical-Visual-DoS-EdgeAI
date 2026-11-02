@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"

if not exist ".venv\Scripts\python.exe" (
  echo Creating project-local Python environment...
  python -m venv .venv
  if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1

for %%D in (".cache\pip" ".cache\tmp" ".cache\torch" ".cache\matplotlib" "dataset\.cache\huggingface\hub" "dataset\.cache\huggingface\datasets" ".config" "dataset\images\train" "dataset\images\val" "dataset\images\test" "dataset\labels\train" "dataset\labels\val" "dataset\labels\test" "runs" "weights") do if not exist "%%~D" mkdir "%%~D"
if not exist "results" mkdir "results"
set "PIP_CACHE_DIR=%PROJECT_ROOT%\.cache\pip"
set "TMP=%PROJECT_ROOT%\.cache\tmp"
set "TEMP=%PROJECT_ROOT%\.cache\tmp"
set "TORCH_HOME=%PROJECT_ROOT%\.cache\torch"
set "YOLO_CONFIG_DIR=%PROJECT_ROOT%\.config"
set "MPLCONFIGDIR=%PROJECT_ROOT%\.cache\matplotlib"
set "HF_HOME=%PROJECT_ROOT%\dataset\.cache\huggingface"
set "HF_HUB_CACHE=%PROJECT_ROOT%\dataset\.cache\huggingface\hub"
set "HF_DATASETS_CACHE=%PROJECT_ROOT%\dataset\.cache\huggingface\datasets"
set "XDG_CACHE_HOME=%PROJECT_ROOT%\.cache"

if not defined YOLOV13_TORCH_INDEX_URL set "YOLOV13_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"

echo Installing PyTorch from %YOLOV13_TORCH_INDEX_URL%
python -m pip install --upgrade pip "setuptools<82" wheel --cache-dir "%PIP_CACHE_DIR%"
if errorlevel 1 exit /b 1
python -m pip install torch torchvision --index-url "%YOLOV13_TORCH_INDEX_URL%" --cache-dir "%PIP_CACHE_DIR%"
if errorlevel 1 (
  echo PyTorch install failed. Set YOLOV13_TORCH_INDEX_URL to the CUDA wheel index for this GPU/driver, or https://download.pytorch.org/whl/cpu for CPU-only.
  exit /b 1
)

python -m pip install -r requirements.txt --cache-dir "%PIP_CACHE_DIR%"
if errorlevel 1 exit /b 1

if not exist "yolov13_src\.git" (
  where git >nul 2>nul
  if errorlevel 1 (
    echo Git is required to clone the official iMoonLab source into this project.
    exit /b 1
  )
  git clone https://github.com/iMoonLab/yolov13.git yolov13_src
  if errorlevel 1 exit /b 1
)

python -m pip install --no-deps -e "%PROJECT_ROOT%\yolov13_src" --cache-dir "%PIP_CACHE_DIR%"
if errorlevel 1 exit /b 1

python configure_yolo_paths.py
if errorlevel 1 exit /b 1

git -C "%PROJECT_ROOT%\yolov13_src" rev-parse HEAD > "%PROJECT_ROOT%\results\source_revision.txt"
python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CUDA available:', torch.cuda.is_available())"
if errorlevel 1 exit /b 1
python -c "import ultralytics.nn.modules"
if errorlevel 1 exit /b 1
python -m pip freeze > "%PROJECT_ROOT%\results\environment-lock.txt"
if errorlevel 1 exit /b 1

echo.
echo Setup complete. FlashAttention is optional: upstream falls back to PyTorch scaled_dot_product_attention when flash_attn is unavailable.
echo On Windows, do not install the Linux-only wheel shown in the upstream README.
echo Pip, temporary, and Torch caches are under %PROJECT_ROOT%\.cache; dataset downloads and Hugging Face caches are under %PROJECT_ROOT%\dataset\.cache.
endlocal
