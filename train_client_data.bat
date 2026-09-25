@echo off
rem Train both models on the client's labelled pass recordings.
rem Usage: train_client_data.bat                      (looks for ..\..\client_pass_recordings)
rem        train_client_data.bat --root E:\CAPSTONE\CAP2\client_pass_recordings
rem        train_client_data.bat --promote            (use the new models)
setlocal
cd /d "%~dp0"
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" ( echo Run setup.bat first. & pause & exit /b 1 )
if not exist logs mkdir logs
for /f "delims=" %%i in ('call "%VPY%" -c "import datetime;print(datetime.datetime.now().strftime('%%Y%%m%%d_%%H%%M'))"') do set "STAMP=%%i"
if not defined STAMP set "STAMP=latest"
set "LOG=logs\client_data_%STAMP%.txt"
echo Log: %LOG%
set PYTHONUNBUFFERED=1
powershell -NoProfile -Command "& '.\%VPY%' train_client_data.py %* 2>&1 | Tee-Object -FilePath '%LOG%'"
echo.
echo Finished. Log saved to %LOG%
pause
endlocal
