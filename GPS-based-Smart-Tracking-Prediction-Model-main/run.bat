@echo off
REM Quick start script for Delivery Prediction Service
REM This script sets up and runs the Flask API

echo.
echo ========================================
echo    Delivery Prediction Service
echo    Quick Start
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [2/4] Verifying project setup...
python setup_project.py

echo [3/4] Creating sample data (if needed)...
python sample_data.py

echo.
echo [4/4] Starting Flask API server...
echo.
echo ========================================
echo Server starting on: http://localhost:5000
echo ========================================
echo.

python app.py
pause
