@echo off
setlocal
cd /d "%~dp0"
if not exist "monocruise.py" (
    echo Extract the entire ZIP before starting MonoCruise.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_monocruise.ps1"
if errorlevel 1 pause
endlocal
