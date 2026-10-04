$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
try {
    . (Join-Path $PSScriptRoot 'windows_setup.ps1')
    Add-Type -AssemblyName System.Windows.Forms
    $picker = New-Object System.Windows.Forms.FolderBrowserDialog
    $picker.Description = 'Select ETS2 installation folder containing base.scs'
    if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
    $game = $picker.SelectedPath
    if (-not (Test-Path (Join-Path $game 'base.scs'))) { throw 'No base.scs in selected folder.' }
    $dotnet = Ensure-Dotnet
    $source = Ensure-TruckLibSource
    $project = Join-Path $root 'integrations\trucklib_reader\Reader.csproj'
    Run-Checked $dotnet @('build', $project, '-c', 'Release', ("-p:TruckLibSource=" + $source))
    $version = '1.61'
    $gameExe = Join-Path $game 'bin\win_x64\eurotrucks2.exe'
    if (Test-Path $gameExe) {
        $detected = (Get-Item $gameExe).VersionInfo.ProductVersion
        if ($detected -match '^(1\.\d+(?:\.\d+)*)') { $version = $Matches[1] }
    }
    Write-Host "Using ETS2 version: $version"
    if ($version -notmatch '^1\.\d+(\.\d+)*$') { throw 'Invalid ETS2 version.' }
    $target = Join-Path $root 'maps\roads.json'
    if (Test-Path $target) { Copy-Item $target ($target + '.bak') -Force }
    $reader = Join-Path $root 'integrations\trucklib_reader\bin\Release\net10.0\Reader.dll'
    Run-Checked $dotnet @($reader, $game, $target, $version)
    Write-Host 'Ordinary road map imported. Click Reload ETS2 road map in settings.'
    Write-Host 'Experimental advisory context. No automatic changes to braking.'
} catch {
    Write-Host "TruckLib export failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
