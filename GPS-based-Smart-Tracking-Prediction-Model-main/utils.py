"""
Utility functions for the delivery prediction service.
"""

import logging
import os
from functools import wraps


def setup_logger(name, level=logging.INFO):
    """Setup logger for a module."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


def require_model(f):
    """Decorator to ensure model is loaded before function execution."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(f, '_model_loaded'):
            raise RuntimeError("Model not loaded. Please check model artifacts.")
        return f(*args, **kwargs)
    return decorated_function


def validate_input_data(data, required_fields=None):
    """
    Validate input data contains required fields.
    
    Args:
        data: Input dictionary or DataFrame
        required_fields: List of required field names
    
    Returns:
        bool: True if valid, False otherwise
    """
    if required_fields is None:
        return True
    
    if isinstance(data, dict):
        return all(field in data for field in required_fields)
    
    return all(field in data.columns for field in required_fields)


def ensure_directory(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
    return path


def get_prediction_bounds():
    """Return reasonable bounds for delivery time predictions."""
    return {'min': 5, 'max': 120}  # 5 to 120 minutes
