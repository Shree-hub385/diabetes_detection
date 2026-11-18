@echo off
title Diabetes Detection System
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 ( py -3 run.py %* & goto done )
where python >nul 2>nul
if %errorlevel%==0 ( python run.py %* & goto done )
echo Python 3.9+ was not found. Install it from https://www.python.org/downloads/
echo (tick "Add Python to PATH" during setup), then double-click run.bat again.
:done
if errorlevel 1 pause
pause
