@echo off
setlocal
cd /d "%~dp0"

set "PARENT_VENV=%~dp0..\.venv\Scripts\python.exe"
set "LOCAL_VENV=%~dp0.venv\Scripts\python.exe"

if exist "%PARENT_VENV%" (
    "%PARENT_VENV%" gui_app.py
) else if exist "%LOCAL_VENV%" (
    "%LOCAL_VENV%" gui_app.py
) else (
    python gui_app.py
)

if errorlevel 1 (
    echo.
    echo The GUI closed with an error.
    echo Check the Python environment, dependencies, and Tkinter installation.
    pause
)
endlocal
