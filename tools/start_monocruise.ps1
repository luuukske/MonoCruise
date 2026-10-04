param([switch]$ReviewClips)
$ErrorActionPreference = 'Stop'
try {
    . (Join-Path $PSScriptRoot 'windows_setup.ps1')
    Set-Location $setupRoot
    $python = Ensure-Python
    & $python -c 'import importlib.util,sys; sys.exit(not all(importlib.util.find_spec(n) is not None for n in ["PySide6","pygame","psutil","requests","packaging","PIL","hid","keyboard","truck_telemetry"]))' 
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing MonoCruise dependencies. First run may take several minutes...'
        Run-Checked $python @('-m', 'pip', 'install', '-r', (Join-Path $setupRoot 'requirements.txt'))
    }
    if ($ReviewClips) {
        Run-Checked $python @('-m', 'tools.aeb_review')
    } else {
        Run-Checked $python @((Join-Path $setupRoot 'monocruise.py'))
    }
} catch {
    Write-Host "MonoCruise could not start: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
