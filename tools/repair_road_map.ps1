$ErrorActionPreference = 'Stop'
Write-Host 'The old SQLite export has unreliable node data. Re-exporting with official TruckLib...'
& (Join-Path $PSScriptRoot 'prepare_trucklib_map.ps1')
exit $LASTEXITCODE
