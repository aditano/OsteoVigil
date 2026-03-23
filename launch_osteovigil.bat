@echo off
REM OsteoVigil Windows launcher — double-click to bootstrap and start the app.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "BOOTSTRAP_PYTHON="

where py >nul 2>&1
if %errorlevel%==0 (
    py -3.11 -c "import sys" >nul 2>&1 && set "BOOTSTRAP_PYTHON=py -3.11"
    if not defined BOOTSTRAP_PYTHON (
        py -3.12 -c "import sys" >nul 2>&1 && set "BOOTSTRAP_PYTHON=py -3.12"
    )
)

if not defined BOOTSTRAP_PYTHON (
    where python >nul 2>&1 && set "BOOTSTRAP_PYTHON=python"
)

if not defined BOOTSTRAP_PYTHON (
    where python3 >nul 2>&1 && set "BOOTSTRAP_PYTHON=python3"
)

if not defined BOOTSTRAP_PYTHON (
    echo No compatible Python interpreter was found.
    echo Please install Python 3.11 or 3.12, then rerun this launcher.
    pause
    exit /b 1
)

echo Starting OsteoVigil...
%BOOTSTRAP_PYTHON% bootstrap.py --entrypoint desktop
pause
