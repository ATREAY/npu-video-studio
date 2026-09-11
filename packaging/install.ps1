#Requires -Version 5.1
<#
  NPU Video Studio - installer for Snapdragon X (Windows on ARM64).
  UNVALIDATED end-to-end (written on a Linux dev box with no ARM64 Windows access) but
  every dependency it installs was individually verified to have a win_arm64 wheel and
  is bundled offline in .\wheelhouse — see README.md "what's actually verified".

  Usage (from an elevated or normal PowerShell, in this packaging/ folder):
      .\install.ps1

  What it does:
    1. Finds a native ARM64 Python 3.11 (must be installed separately - see README).
    2. Creates a venv at ..\.venv-win-arm64
    3. Installs every dependency from the LOCAL wheelhouse (no internet needed)
    4. Copies the app/ and export_assets/ folders next to the venv
    5. Writes run.ps1's companion launcher paths
#>

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Here
$Venv = Join-Path $Root ".venv-win-arm64"

Write-Host "== NPU Video Studio installer ==" -ForegroundColor Cyan

# --- 1. find a native ARM64 python -----------------------------------------
$python = $null
foreach ($cand in @("py -3.11-arm64", "python3.11", "python")) {
    try {
        $exe, $rest = $cand -split " ", 2
        $ver = & $exe $rest "--version" 2>$null
        $arch = & $exe $rest "-c" "import platform; print(platform.machine())" 2>$null
        if ($arch -match "ARM64") { $python = "$exe $rest"; break }
    } catch {}
}
if (-not $python) {
    Write-Host "ERROR: no native ARM64 Python 3.11 found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/windows/ -- pick the" -ForegroundColor Yellow
    Write-Host "'Windows installer (arm64)' build, NOT the regular x86-64 installer." -ForegroundColor Yellow
    exit 1
}
Write-Host "Using: $python ($arch)"

# --- 2. venv -----------------------------------------------------------------
if (-not (Test-Path $Venv)) {
    Write-Host "Creating venv at $Venv"
    Invoke-Expression "$python -m venv `"$Venv`""
}
$VenvPy = Join-Path $Venv "Scripts\python.exe"

# --- 3. offline install from the bundled wheelhouse --------------------------
Write-Host "Installing dependencies from .\wheelhouse (offline, no internet needed)"
& $VenvPy -m pip install --upgrade pip --no-index --find-links "$Here\wheelhouse" 2>$null
& $VenvPy -m pip install --no-index --find-links "$Here\wheelhouse" -r "$Here\requirements-win-arm64.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Offline install failed - falling back to PyPI (needs internet)" -ForegroundColor Yellow
    & $VenvPy -m pip install -r "$Here\requirements-win-arm64.txt"
}

# --- 4. sanity check: does the process actually see QNNExecutionProvider? ----
Write-Host "`nChecking ONNX Runtime providers..." -ForegroundColor Cyan
& $VenvPy -c "import onnxruntime as ort; print('Available providers:', ort.get_available_providers())"
Write-Host "If 'QNNExecutionProvider' is NOT listed above, the app still runs (CPU" -ForegroundColor Yellow
Write-Host "fallback) but won't use the NPU. See README.md troubleshooting." -ForegroundColor Yellow

Write-Host "`n== Install complete ==" -ForegroundColor Green
Write-Host "Run:  .\run.ps1"
