@echo off
setlocal
cd /d "%~dp0"
if not exist "tools\aeb_review.py" (
    echo Extract the entire ZIP before opening the clip viewer.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_monocruise.ps1" -ReviewClips
if errorlevel 1 pause
endlocal
