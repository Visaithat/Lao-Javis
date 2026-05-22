# Launch Javis using the D: drive venv (created because C: was full).
$ErrorActionPreference = "Stop"
$python = "D:\Javis-env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "venv not found at $python - run setup first." -ForegroundColor Red
    exit 1
}
$env:PYTHONIOENCODING = "utf-8"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
& $python main.py
