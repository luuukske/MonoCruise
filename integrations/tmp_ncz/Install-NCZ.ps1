$ErrorActionPreference = 'Stop'
try {
    $dll = Join-Path $PSScriptRoot 'MonoCruiseNCZ.dll'
    if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) {
        throw 'MonoCruiseNCZ.dll is missing. Extract the entire ZIP first.'
    }
    if (Get-Process -Name eurotrucks2,amtrucks -ErrorAction SilentlyContinue) {
        throw 'Close ETS2 and ATS before installing.'
    }
    $libraries = @()
    $steam = (Get-ItemProperty 'HKCU:\Software\Valve\Steam' -ErrorAction SilentlyContinue).SteamPath
    if ($steam) { $libraries += $steam }
    if (${env:ProgramFiles(x86)}) { $libraries += (Join-Path ${env:ProgramFiles(x86)} 'Steam') }
    foreach ($base in @($libraries)) {
        $vdf = Join-Path $base 'steamapps\libraryfolders.vdf'
        if (Test-Path -LiteralPath $vdf) {
            $content = Get-Content -LiteralPath $vdf -Raw
            foreach ($match in [regex]::Matches($content, '"path"\s+"([^"]+)"')) {
                $libraries += $match.Groups[1].Value.Replace('\\','\')
            }
        }
    }
    $games = @()
    foreach ($base in ($libraries | Select-Object -Unique)) {
        foreach ($name in @('Euro Truck Simulator 2','American Truck Simulator')) {
            $game = Join-Path $base "steamapps\common\$name"
            $bin = Join-Path $game 'bin\win_x64'
            if ((Test-Path (Join-Path $bin 'eurotrucks2.exe')) -or (Test-Path (Join-Path $bin 'amtrucks.exe'))) {
                $games += $game
            }
        }
    }
    if (-not $games) {
        Add-Type -AssemblyName System.Windows.Forms
        $picker = New-Object System.Windows.Forms.FolderBrowserDialog
        $picker.Description = 'Select the ETS2 or ATS game folder containing bin.'
        if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
        $game = $picker.SelectedPath
        $bin = Join-Path $game 'bin\win_x64'
        if (-not ((Test-Path (Join-Path $bin 'eurotrucks2.exe')) -or (Test-Path (Join-Path $bin 'amtrucks.exe')))) {
            throw 'This is not an ETS2 or ATS game folder. Select the folder containing bin.'
        }
        $games = @($game)
    }
    foreach ($game in ($games | Select-Object -Unique)) {
        $plugins = Join-Path $game 'bin\win_x64\plugins'
        New-Item -ItemType Directory -Path $plugins -Force | Out-Null
        $target = Join-Path $plugins 'MonoCruiseNCZ.dll'
        if (Test-Path -LiteralPath $target) {
            Copy-Item -LiteralPath $target -Destination ($target + '.bak') -Force
        }
        Copy-Item -LiteralPath $dll -Destination $target -Force
        if ((Get-FileHash -LiteralPath $dll).Hash -ne (Get-FileHash -LiteralPath $target).Hash) {
            throw 'Copied DLL did not match the source.'
        }
        Write-Host "Installed: $target" -ForegroundColor Green
    }
    Write-Host 'Done. Run the updated MonoCruise and launch TruckersMP.'
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host 'If access is denied, right-click Install-NCZ.cmd and choose Run as administrator.'
    exit 1
}
