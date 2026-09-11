#Requires -Version 5.1
<#
  Launch NPU Video Studio. Run .\install.ps1 first, once.

  Examples:
    .\run.ps1                              # webcam -> preview window (for OBS Window Capture)
    .\run.ps1 -Source clip.mp4 -Out out.mp4 -Provider cpu -Debug
    .\run.ps1 -BgMode replace -BgImage .\office.jpg
#>
param(
    [string]$Source = "0",
    [string]$Out = "window",
    [ValidateSet("auto", "npu", "cpu")][string]$Provider = "auto",
    [ValidateSet("blur", "replace", "none")][string]$BgMode = "blur",
    [string]$BgImage = "",
    [string]$SrAsset = "quicksrnetmedium-onnx-float-2x360",
    [string]$Size = "1280x720",
    [switch]$Debug
)

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Here
$VenvPy = Join-Path $Root ".venv-win-arm64\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Host "Not installed yet - run .\install.ps1 first." -ForegroundColor Red
    exit 1
}

$argsList = @("-m", "app.main", "--source", $Source, "--out", $Out,
             "--provider", $Provider, "--bg", $BgMode,
             "--sr-asset", $SrAsset, "--size", $Size)
if ($BgImage) { $argsList += @("--bg-image", $BgImage) }
if ($Debug) { $argsList += "--debug" }

if ($Out -eq "window") {
    Write-Host "Preview window opening. In OBS Studio: Sources -> + -> Window Capture ->" -ForegroundColor Cyan
    Write-Host "select 'NPU Video Studio' -> then Start Virtual Camera." -ForegroundColor Cyan
}

Push-Location $Root
& $VenvPy @argsList
Pop-Location
