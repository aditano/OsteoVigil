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

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment with %BOOTSTRAP_PYTHON%...
    %BOOTSTRAP_PYTHON% -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
set "PYTHON=.venv\Scripts\python.exe"
set "REQ_STAMP=.venv\.osteovigil_requirements_installed"
set "NEEDS_INSTALL=0"

if not exist "%REQ_STAMP%" (
    set "NEEDS_INSTALL=1"
) else (
    powershell -NoProfile -Command "if ((Get-Item 'requirements.txt').LastWriteTimeUtc -gt (Get-Item '.venv\.osteovigil_requirements_installed').LastWriteTimeUtc) { exit 10 }"
    if !errorlevel! equ 10 set "NEEDS_INSTALL=1"
)

if "%NEEDS_INSTALL%"=="1" (
    echo Installing OsteoVigil dependencies...
    %PYTHON% -m pip install --upgrade pip
    if errorlevel 1 (
        echo Failed to upgrade pip.
        pause
        exit /b 1
    )
    %PYTHON% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
    type nul > "%REQ_STAMP%"
)

echo Starting OsteoVigil...
%PYTHON% desktop_app.py
pause
