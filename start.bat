@echo off
rem Double-click to start the sign trainer; the browser opens by itself.
rem Close this window (or press Ctrl+C) to stop the app.
chcp 65001 >nul
title Kad se male ruke slože
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found in .venv - see README.md for setup.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m signtrainer
if errorlevel 1 pause
