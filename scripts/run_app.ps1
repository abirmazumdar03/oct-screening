
# scripts/run_app.ps1: start the app locally. Works fully OFFLINE once installed.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 [-Model artifacts\model_package] [-Lang hi] [-NoHeatmap] [-Share]

param(
  [string]$Model = "artifacts/model_package",
  [string]$Lang = "en",
  [switch]$NoHeatmap,
  [switch]$Share
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"
$ok = $false

if (Test-Path $py) {
  $ErrorActionPreference = "Continue"
  & $py -c "import gradio, onnxruntime" 2>&1 | Out-Null
  $ok = ($LASTEXITCODE -eq 0)
  $ErrorActionPreference = "Stop"
}

if (-not $ok) {
  Write-Host "First run: installing app packages (needs internet once)..." -ForegroundColor Yellow
  uv sync --inexact --group app --group app-heatmap
}

$env:OCT_MODEL_DIR = $Model
$env:GRADIO_ANALYTICS_ENABLED = "False"
$env:GRADIO_TEMP_DIR = Join-Path $PWD "artifacts\tmp_uploads"

if (Test-Path $env:GRADIO_TEMP_DIR) {
  Remove-Item -Recurse -Force $env:GRADIO_TEMP_DIR
}

if ($NoHeatmap) {
  $env:OCT_ENABLE_HEATMAP = "0"
}

$cli = @("-m", "octscreen.app", "--set", "ui.default_language=$Lang")

if ($Share) {
  $cli += "--share"
}

& $py @cli
