# dekupe installer (Windows). Copies the skill to ~/.claude/skills/dekupe, creates its venv,
# downloads the pinned ONNX models and runs the static audit.
#   powershell -ExecutionPolicy Bypass -File install.ps1            # NVIDIA GPU (CUDA)
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Runtime directml   # AMD/Intel GPU
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Runtime cpu
param([ValidateSet("cuda", "directml", "cpu")][string]$Runtime = "cuda")
$ErrorActionPreference = "Stop"

foreach ($tool in "python", "ffmpeg", "ffprobe") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not found on PATH" }
}

$dest = Join-Path $HOME ".claude\skills\dekupe"
$src = Join-Path $PSScriptRoot "skill\dekupe"
New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item -Recurse -Force (Join-Path $src "*") $dest

$py = Join-Path $dest ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { python -m venv (Join-Path $dest ".venv") }
& $py -m pip install --quiet --upgrade pip
$ort = @{ cuda = "onnxruntime-gpu[cuda,cudnn]"; directml = "onnxruntime-directml"; cpu = "onnxruntime" }[$Runtime]
& $py -m pip install --quiet numpy onnx $ort
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

& $py -I (Join-Path $dest "scripts\fetch_model.py") --all
if ($LASTEXITCODE -ne 0) { throw "model download/verification failed" }
& $py -I (Join-Path $dest "scripts\audit_onnx.py") (Get-ChildItem (Join-Path $dest "models\*.onnx")).FullName
if ($LASTEXITCODE -ne 0) { throw "ONNX audit failed" }

Write-Host "`ndekupe installed to $dest"
Write-Host "Try: & '$py' -I '$dest\scripts\matte.py' input.mp4 --green green.mp4"
