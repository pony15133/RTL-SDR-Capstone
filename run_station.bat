@echo off
rem Starts the live dashboard (new window + browser) and the automatic capture loop.
rem Extra options go to auto_capture.py, e.g.  run_station.bat --simulate   or   --hours 12
setlocal
cd /d "%~dp0"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" ( echo Run setup.bat first. & pause & exit /b 1 )
if not exist capture_config.json copy capture_config.example.json capture_config.json >nul
start "Station dashboard" "%VPY%" dashboard.py --config capture_config.json
timeout /t 2 >nul
start "" http://localhost:8050
"%VPY%" auto_capture.py --config capture_config.json --forever %*
pause
endlocal
