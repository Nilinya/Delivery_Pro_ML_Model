"""
Application configuration management.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration."""
    
    # Flask
    DEBUG = False
    TESTING = False
    JSON_SORT_KEYS = False
    
    # Server
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Model
    MODEL_DIR = os.getenv('MODEL_DIR', 'model')
    MODEL_PATH = os.path.join(MODEL_DIR, 'delivery_model.joblib')
    SCALER_PATH = os.path.join(MODEL_DIR, 'delivery_scaler.joblib')
    ENCODERS_PATH = os.path.join(MODEL_DIR, 'delivery_encoders.joblib')
    FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_columns.joblib')
    
    # Data
    DATA_DIR = 'data'
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


def get_config():
    """Get configuration based on environment."""
    env = os.getenv('FLASK_ENV', 'production').lower()
    
    if env == 'development':
        return DevelopmentConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return ProductionConfig()


config = get_config()
