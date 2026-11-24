@echo off
title Whisper Transkriptor Studio GUI
cd /d "%~dp0"

echo ===================================================
echo     Memulai Whisper Transkriptor Studio GUI...
echo ===================================================

python app_gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Aplikasi berhenti dengan kode error %ERRORLEVEL%.
    pause
)
