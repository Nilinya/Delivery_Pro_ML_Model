#!/usr/bin/env python3
"""
Project initialization and setup script.
Run: python setup_project.py
"""

import os
import sys
import shutil
from pathlib import Path


def setup_directories():
    """Ensure all required directories exist."""
    dirs = ['model', 'data', 'templates', 'static', 'tests', '__pycache__']
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_name == '__pycache__':
            continue
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_name}/")


def check_model_files():
    """Check if model artifacts exist."""
    model_files = [
        'model/delivery_model.joblib',
        'model/delivery_scaler.joblib',
        'model/delivery_encoders.joblib',
        'model/feature_columns.joblib'
    ]
    
    print("\n📦 Checking model files...")
    missing = []
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (MISSING)")
            missing.append(file_path)
    
    if missing:
        print("\n⚠️  Missing model files. Run 'python train_model.py' to train the model.")
        return False
    
    return True


def check_data_files():
    """Check if data files exist."""
    data_files = ['data/train.csv', 'data/test.csv']
    
    print("\n📊 Checking data files...")
    missing = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"⚠️  {file_path} (optional)")
            missing.append(file_path)
    
    return len(missing) < 2


def main():
    """Run setup."""
    print("🚀 Delivery Prediction Service - Project Setup\n")
    
    # Setup directories
    setup_directories()
    
    # Check files
    models_ok = check_model_files()
    data_ok = check_data_files()
    
    print("\n" + "="*60)
    print("📋 Setup Summary:")
    print("="*60)
    print(f"Model artifacts: {'✅ OK' if models_ok else '❌ Missing'}")
    print(f"Data files: {'✅ OK' if data_ok else '⚠️  Check data/ folder'}")
    
    print("\n🎯 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Ensure data files in data/ folder (train.csv, test.csv)")
    print("3. Train model: python train_model.py")
    print("4. Run API: python app.py")
    print("5. Visit: http://localhost:5000")
    
    print("\n📚 Available Commands:")
    print("- python train_model.py    # Train/re-train the model")
    print("- python app.py             # Start Flask API server")
    print("- pytest tests/             # Run unit tests")
    
    print("\n✅ Setup complete! Ready to go.\n")


if __name__ == '__main__':
    main()
