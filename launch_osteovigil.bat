@echo off
REM OsteoVigil Windows launcher — double-click to start the app.
cd /d "%~dp0"

REM Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Try python, fall back to python3
where python >nul 2>&1 && set PYTHON=python || set PYTHON=python3

REM Install PyQt6 if missing
%PYTHON% -c "import PyQt6" >nul 2>&1 || (
    echo Installing PyQt6...
    %PYTHON% -m pip install "PyQt6>=6.6.0"
)

echo Starting OsteoVigil...
%PYTHON% desktop_app.py
pause
