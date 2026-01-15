@echo off
title Destiny Engine Ver 2.2
echo Checking environment and dependencies...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] Python is not installed or not in PATH.
    pause
    exit /b
)

:: Install missing libraries automatically
echo Updating libraries (numpy, scipy, matplotlib)...
pip install numpy scipy matplotlib pandas -q

:: Execute the script
echo.
echo Launching the Destiny Integrator...
python destiny_integrator.py

pause