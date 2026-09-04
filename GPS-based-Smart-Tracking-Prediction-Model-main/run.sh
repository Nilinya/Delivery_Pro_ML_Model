#!/bin/bash
# Quick start script for Delivery Prediction Service
# This script sets up and runs the Flask API

clear

echo ""
echo "========================================"
echo "    Delivery Prediction Service"
echo "    Quick Start"
echo "========================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "[1/4] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo "[2/4] Verifying project setup..."
python3 setup_project.py

echo "[3/4] Creating sample data (if needed)..."
python3 sample_data.py

echo ""
echo "[4/4] Starting Flask API server..."
echo ""
echo "========================================"
echo "Server starting on: http://localhost:5000"
echo "========================================"
echo ""

python3 app.py
