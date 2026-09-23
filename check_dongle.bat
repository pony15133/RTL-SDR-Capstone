@echo off
rem 1-minute RTL-SDR hardware test. Extra options: --freq 95.0e6  --gain 40  --simulate
cd /d "%~dp0"
".venv\Scripts\python.exe" check_dongle.py %*
pause
