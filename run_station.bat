@echo off
rem Starts the station web page (new window + browser) and the automatic capture loop.
rem If the capture loop ever crashes it is restarted automatically (after 1 minute).
rem Close this window or press Ctrl+C twice to stop. Logs: logs\station.log
rem Extra options go to auto_capture.py, e.g.  run_station.bat --simulate   or   --hours 12
setlocal
cd /d "%~dp0"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" ( echo Run setup.bat first. & pause & exit /b 1 )
if not exist capture_config.json copy capture_config.example.json capture_config.json >nul
start "Station web page" "%VPY%" dashboard.py --config capture_config.json
timeout /t 2 >nul
start "" http://localhost:8050
set RESTARTS=0
:loop
"%VPY%" auto_capture.py --config capture_config.json --forever %*
if %ERRORLEVEL% EQU 0 goto done
set /a RESTARTS+=1
echo.
echo Capture loop stopped with an error (restart %RESTARTS%). Restarting in 60 s - press Ctrl+C to stop.
timeout /t 60
goto loop
:done
pause
endlocal
