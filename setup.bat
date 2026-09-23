@echo off
rem One-time setup on Windows: virtual environment, packages, models, environment check.
rem   setup.bat                   quick setup
rem   setup.bat --fetch-satnogs   also download SatNOGS training data (10-20 min)
setlocal
cd /d "%~dp0"
set "PYLAUNCH=python"
where py >nul 2>nul && set "PYLAUNCH=py -3"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>nul || (
    echo The existing .venv points at a Python that is no longer installed - rebuilding it.
    rmdir /s /q .venv
  )
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PYLAUNCH% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip || goto :fail
"%VPY%" -m pip install -r sdr-doppler-prototype\requirements.txt -r iq-recorder\requirements.txt || goto :fail
"%VPY%" setup_station.py %*
goto :end
:fail
echo.
echo Setup failed - see the messages above. Is Python 3.10+ installed (python.org, tick "Add to PATH")?
:end
pause
endlocal
