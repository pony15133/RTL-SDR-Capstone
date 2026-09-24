@echo off
rem Overnight training: downloads a large SatNOGS dataset (waiting through the
rem SatNOGS hourly API limit by itself), retrains both models and tests them on
rem the RSP-03 pass. Keeps the computer awake until it finishes, then lets it sleep.
rem Everything is also written to logs\overnight_<date>.txt - send that file in the morning.
rem Keep the laptop plugged in and the lid open. Ctrl+C stops it; run it again to resume.
setlocal
cd /d "%~dp0"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" ( echo Run setup.bat first. & pause & exit /b 1 )
if not exist logs mkdir logs
for /f %%i in ('"%VPY%" -c "import datetime;print(datetime.datetime.now().strftime('%%Y%%m%%d_%%H%%M'))"') do set "STAMP=%%i"
set "LOG=logs\overnight_%STAMP%.txt"
echo Log: %LOG%
set PYTHONUNBUFFERED=1
powershell -NoProfile -Command "& '.\%VPY%' setup_station.py --fetch-satnogs --big-data --keep-awake --skip-doctor %* 2>&1 | Tee-Object -FilePath '%LOG%'"
echo.
echo Finished. Log saved to %LOG%
pause
endlocal
