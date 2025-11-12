@echo off
chcp 65001 > nul
title AI Image Analyzer Pro

echo ==========================================
echo    AI Image Analyzer Pro - Starting...
echo ==========================================
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [ERROR] Virtual environment not found
    echo Please run setup.bat first
    pause
    exit
)

echo.
echo [OK] Launching GUI...
echo.

REM Launch GUI
python modern_gui.py

REM Error check
if errorlevel 1 (
    echo.
    echo [ERROR] An error occurred
    pause
)
