@echo off
rem Windows launcher for the desktop GUI (macOS / Linux: launch_gui.sh).
rem Uses the repository .venv if present (created by setup.bat), otherwise Python on PATH.
setlocal
cd /d "%~dp0"
if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" gui_app.py %*
  goto :end
)
where py >/dev/null 2>/dev/null && ( py -3 gui_app.py %* & goto :end )
where python >/dev/null 2>/dev/null && ( python gui_app.py %* & goto :end )
echo Python 3 not found. Run setup.bat in the repository root first, or install Python from python.org.
pause
:end
endlocal
