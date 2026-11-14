# scripts/run_app.ps1: start the app locally. Works fully OFFLINE once installed.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 [-Model artifacts\model_package] [-Lang hi] [-NoHeatmap] [-Share]
param(
  [string]$Model = "artifacts/model_package",
  [string]$Lang = "en",
  [switch]$NoHeatmap,
  [switch]$Share
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)                 # repo root
$py = ".venv\Scripts\python.exe"
$ok = $false
if (Test-Path $py) {                                              # are the app packages installed?
  $ErrorActionPreference = "Continue"                             # (PowerShell 5.1: don't abort on the check's stderr)
  & $py -c "import gradio, onnxruntime" 2>&1 | Out-Null
  $ok = ($LASTEXITCODE -eq 0)
  $ErrorActionPreference = "Stop"
}
if (-not $ok) {
  Write-Host "First run: installing app packages (needs internet once)..." -ForegroundColor Yellow
  uv sync --inexact --group app --group app-heatmap              # --inexact keeps teammates' other packages
}
if (-not (Test-Path (Join-Path $Model "model_meta.json"))) {
  Write-Host "No model package in '$Model'. Unzip the package there first (see STEP_BY_STEP.md)." -ForegroundColor Red
}
$env:OCT_MODEL_DIR = $Model
$env:PYTHONUTF8 = "1"                                             # Hindi + emoji safe in any console
$env:GRADIO_ANALYTICS_ENABLED = "False"
$env:GRADIO_TEMP_DIR = Join-Path $PWD "artifacts\tmp_uploads"
if (Test-Path $env:GRADIO_TEMP_DIR) { Remove-Item -Recurse -Force $env:GRADIO_TEMP_DIR }   # no uploads kept between sessions
if ($NoHeatmap) { $env:OCT_ENABLE_HEATMAP = "0" } else { Remove-Item Env:OCT_ENABLE_HEATMAP -ErrorAction SilentlyContinue }
$cli = @("-m", "octscreen.app", "--set", "ui.default_language=$Lang")
if ($Share) { $cli += "--share" }
& $py @cli
