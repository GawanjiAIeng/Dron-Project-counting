@echo off
title Object Detection App
cd /d "%~dp0app"
cls
echo ========================================
echo    Object Detection App
echo ========================================
echo.

:: Find Python
set CMD=py
where %CMD% >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    set CMD=python
    where %CMD% >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Python not found!
        pause
        exit /b 1
    )
)

echo Starting... (loading model, please wait)
%CMD% app.py
pause
