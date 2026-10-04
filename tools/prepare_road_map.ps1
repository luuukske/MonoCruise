$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'prepare_trucklib_map.ps1')
exit $LASTEXITCODE
