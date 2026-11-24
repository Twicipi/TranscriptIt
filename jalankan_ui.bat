@echo off
title Whisper Transkriptor Studio GUI
cd /d "%~dp0"

echo ===================================================
echo     Memulai Whisper Transkriptor Studio GUI...
echo ===================================================

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" app_gui.py
) else (
    python app_gui.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Aplikasi berhenti dengan kode error %ERRORLEVEL%.
    pause
)
