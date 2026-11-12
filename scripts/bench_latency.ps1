# scripts/bench_latency.ps1
# Usage (from the repo folder):  powershell -ExecutionPolicy Bypass -File scripts\bench_latency.ps1 -Machine "aditya-ryzen5"
param(
  [string]$Machine = $env:COMPUTERNAME,
  [string]$Images  = "artifacts/packs/sample_scans",
  [string]$OnnxDir = "artifacts/onnx"
)
$ErrorActionPreference = "Stop"
Write-Host "Before starting: plug in the charger, set Windows power mode to 'Best performance', close other apps." -ForegroundColor Yellow
uv sync --group optimize
uv run python -m octscreen.optimize.benchmark latency --config configs/optimize.yaml --images $Images --machine $Machine --set "export.out_dir=$OnnxDir"
Write-Host "Done. Send artifacts\benchmark\latency_$Machine.json to Abir." -ForegroundColor Green