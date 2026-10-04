$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$setupRoot = Split-Path $PSScriptRoot -Parent
$setupCache = Join-Path $setupRoot '.map-tools'

function Run-Checked([string]$program, [string[]]$arguments) {
    & $program @arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$program failed. See output above." }
}

function Ensure-Python {
    $venvPython = Join-Path $setupRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) { return $venvPython }
    $localPython = Join-Path $setupCache 'python\python.exe'
    foreach ($candidate in @((Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
                            (Join-Path $env:ProgramFiles 'Python312\python.exe'))) {
        if (Test-Path $candidate) { $localPython = $candidate; break }
    }
    if (-not (Test-Path $localPython)) {
        New-Item -ItemType Directory -Force $setupCache | Out-Null
        $installer = Join-Path $setupCache 'python-installer.exe'
        Write-Host 'Downloading Python from python.org...'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $installer
        $signature = Get-AuthenticodeSignature $installer
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
            throw 'Python installer signature could not be verified.'
        }
        $target = Join-Path $setupCache 'python'
        Write-Host 'Installing Python for this user...'
        $arguments = @('/quiet', 'InstallAllUsers=0', 'Include_launcher=0',
            'Include_test=0', 'Include_pip=1', 'PrependPath=0', ('TargetDir="' + $target + '"'))
        $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
        if ($process.ExitCode -notin @(0, 3010)) { throw "Python installation failed: $($process.ExitCode)" }
    }
    if (-not (Test-Path $localPython)) { throw 'Python installation did not produce python.exe.' }
    Run-Checked $localPython @('-m', 'venv', (Join-Path $setupRoot '.venv'))
    return $venvPython
}

function Ensure-Dotnet {
    param([string]$channel = '10.0')
    $sdkPattern = '^' + [regex]::Escape($channel.Split('.')[0]) + '\.'
    $command = Get-Command dotnet -ErrorAction SilentlyContinue
    if ($command) {
        $sdks = & $command.Source --list-sdks
        if ($LASTEXITCODE -eq 0 -and ($sdks -match $sdkPattern)) { return $command.Source }
    }
    $directory = Join-Path $setupCache 'dotnet'
    $exe = Join-Path $directory 'dotnet.exe'
    if (Test-Path $exe) {
        $sdks = & $exe --list-sdks
        if ($LASTEXITCODE -eq 0 -and ($sdks -match $sdkPattern)) { return $exe }
    }
    New-Item -ItemType Directory -Force $setupCache | Out-Null
    $script = Join-Path $setupCache 'dotnet-install.ps1'
    Write-Host "Downloading .NET $channel SDK using the official Microsoft installer..."
    Invoke-WebRequest -UseBasicParsing -Uri 'https://dot.net/v1/dotnet-install.ps1' -OutFile $script
    Run-Checked 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        $script, '-Channel', $channel, '-InstallDir', $directory, '-NoPath')
    if (-not (Test-Path $exe)) { throw '.NET SDK installation failed.' }
    $sdks = & $exe --list-sdks
    if ($LASTEXITCODE -ne 0 -or -not ($sdks -match $sdkPattern)) { throw '.NET SDK verification failed.' }
    $env:DOTNET_ROOT = $directory
    return $exe
}

function Ensure-TruckLibSource {
    $commit = 'c42dbe4f2d6bfb7b87c9ff1199c3cf5b7d1e27b0'
    $directory = Join-Path $setupCache ('TruckLib-' + $commit)
    $project = Join-Path $directory 'TruckLib\TruckLib.csproj'
    if (-not (Test-Path $project)) {
        New-Item -ItemType Directory -Force $setupCache | Out-Null
        $archive = Join-Path $setupCache 'official-trucklib.zip'
        Write-Host 'Downloading pinned official TruckLib map reader...'
        Invoke-WebRequest -UseBasicParsing -Uri ("https://github.com/sk-zk/TruckLib/archive/$commit.zip") -OutFile $archive
        Expand-Archive -Path $archive -DestinationPath $setupCache -Force
    }
    if (-not (Test-Path $project)) { throw 'Official TruckLib download incomplete.' }
    [xml]$manifest = Get-Content -Raw $project
    foreach ($node in @($manifest.SelectNodes('//PackageReference[@Include="Microsoft.SourceLink.GitHub"]'))) {
        [void]$node.ParentNode.RemoveChild($node)
    }
    $manifest.Save($project)
    return $directory
}
