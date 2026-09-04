"""
Flask API for delivery time prediction.
FIXED: Model auto-loads for pytest + Flask tests.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from preprocess import preprocess_data
from comparison import (
    generate_scenarios, 
    batch_predict_with_labels, 
    generate_insights
)
import logging
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Initialize Flask app
app = Flask(__name__)

# Comprehensive CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins
        "methods": ["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],
        "allow_headers": [
            "Content-Type", 
            "Authorization", 
            "X-Requested-With",
            "Accept",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers"
        ],
        "expose_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True,
        "max_age": 3600  # Cache preflight response for 1 hour
    }
})

app.config['JSON_SORT_KEYS'] = False

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = NumpyEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths - FIXED: Use absolute paths based on app location
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'delivery_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'delivery_scaler.joblib')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'delivery_encoders.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_columns.joblib')

# Global model variables - FIXED for pytest
model = None
scaler = None
encoders = None
feature_columns = None

def load_models_safely():
    """Load models safely with retry logic."""
    global model, scaler, encoders, feature_columns
    
    logger.info(f"📁 Model directory: {MODEL_DIR}")
    logger.info(f"🔍 Looking for models...")
    
    try:
        # ✅ Verify files exist first
        files_status = {}
        for name, path in [
            ('Model', MODEL_PATH),
            ('Scaler', SCALER_PATH),
            ('Encoders', ENCODERS_PATH),
            ('Features', FEATURES_PATH)
        ]:
            exists = os.path.exists(path)
            files_status[name] = exists
            status = "✅" if exists else "❌"
            logger.info(f"   {status} {name}: {path}")
            
        if not all(files_status.values()):
            logger.warning(f"❌ Missing model files - cannot proceed")
            return False
        
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        feature_columns = joblib.load(FEATURES_PATH)
        logger.info("✅ All models loaded successfully")
        logger.info(f"   Features: {len(feature_columns)} total")
        logger.info(f"   Encoders: {len(encoders)} categorical fields")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        model = scaler = encoders = feature_columns = None
        return False

# AUTO-LOAD MODELS ON STARTUP
if __name__ != '__main__':  # Flask test client
    logger.info("🔄 Loading models for Flask...")
    load_models_safely()

def predict_delivery_time(input_data):
    """
    Make delivery time prediction.
    """
    global model, scaler, encoders, feature_columns
    
    if model is None:
        load_models_safely()
        if model is None:
            raise RuntimeError("Model not loaded - run python train_model.py")
    
    try:
        # Handle dict or DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Preprocess
        processed_input, _ = preprocess_data(
            input_df, is_training=False, encoders=encoders
        )
        
        # Align features
        for col in feature_columns:
            if col not in processed_input.columns:
                processed_input[col] = 0
        processed_input = processed_input[feature_columns]
        
        # Ensure numeric
        for col in processed_input.columns:
            processed_input[col] = pd.to_numeric(
                processed_input[col], errors='coerce'
            ).fillna(0)
        
        input_scaled = scaler.transform(processed_input)
        pred = float(model.predict(input_scaled)[0])
        
        return round(max(pred, 0), 2)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return 25.0  # Default fallback

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """Home page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Single prediction."""
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No JSON data provided'
            }), 400
        
        predicted_time = predict_delivery_time(data)
        
        return jsonify({
            'success': True,
            'predicted_time_minutes': predicted_time,
            'message': f'Predicted: {predicted_time} minutes'
        }), 200
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/predict-batch', methods=['POST'])
def api_predict_batch():
    """Batch prediction."""
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({'success': False, 'message': 'Expected JSON array'}), 400
        
        if len(data) == 0:
            return jsonify({'success': False, 'message': 'Empty array'}), 400
        
        predictions = [predict_delivery_time(row) for row in data]
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions),
            'message': f'Generated {len(predictions)} predictions'
        }), 200
        
    except Exception as e:
        logger.error(f"Batch error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/predict-comparison', methods=['POST'])
def api_predict_comparison():
    """
    Comparison prediction - generates scenarios and shows delivery time predictions
    across different distance ranges, vehicle types, weather, traffic, or personnel ratings.
    """
    try:
        data = request.get_json()
        logger.info(f"Received comparison request: {json.dumps(data, indent=2, default=str)[:200]}...")
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No JSON data provided'
            }), 400
        
        # Extract parameters
        base_delivery = data.get('base_delivery')
        comparison_type = data.get('comparison_type', 'distance_ranges')
        
        if not base_delivery:
            return jsonify({
                'success': False,
                'message': 'Missing required field: base_delivery'
            }), 400
        
        # Validate comparison_type
        valid_types = [
            'distance_ranges', 
            'vehicle_types', 
            'weather_impact', 
            'traffic_impact', 
            'personnel_ratings'
        ]
        if comparison_type not in valid_types:
            return jsonify({
                'success': False,
                'message': f'Invalid comparison_type. Must be one of: {", ".join(valid_types)}'
            }), 400
        
        # Get base prediction
        try:
            base_prediction = predict_delivery_time(base_delivery)
        except Exception as e:
            logger.error(f"Base prediction error: {e}")
            return jsonify({
                'success': False,
                'message': f'Failed to generate base prediction: {str(e)}'
            }), 400
        
        # Generate scenarios
        try:
            scenarios_with_labels = generate_scenarios(base_delivery, comparison_type)
        except ValueError as e:
            logger.warning(f"Scenario validation error: {e}")
            return jsonify({
                'success': False,
                'message': str(e)
            }), 400
        except Exception as e:
            logger.error(f"Scenario generation error: {e}")
            return jsonify({
                'success': False,
                'message': f'Failed to generate scenarios: {str(e)}'
            }), 500
        
        # Batch predict
        try:
            results = batch_predict_with_labels(predict_delivery_time, scenarios_with_labels)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return jsonify({
                'success': False,
                'message': f'Failed to generate predictions: {str(e)}'
            }), 500
        
        # Filter out results with errors for clean output
        clean_results = []
        for r in results:
            if 'error' not in r:
                item = {
                    'label': r['label'],
                    'predicted_time': r['predicted_time']
                }
                if r.get('metadata'):
                    item.update(r['metadata'])
                clean_results.append(item)
            else:
                logger.warning(f"Scenario failed: {r['label']} - {r['error']}")
        
        # Generate insights
        insights = generate_insights(base_prediction, results, comparison_type)
        
        return jsonify({
            'success': True,
            'base_prediction': round(base_prediction, 2),
            'comparison_type': comparison_type,
            'comparisons': clean_results,
            'insights': insights,
            'x_axis_label': _get_x_axis_label(comparison_type),
            'count': len(clean_results),
            'message': f'Generated {len(clean_results)} comparison predictions'
        }), 200
        
    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Comparison analysis failed: {str(e)}'
        }), 500


def _get_x_axis_label(comparison_type):
    """Get X-axis label for comparison type."""
    labels = {
        'distance_ranges': 'Distance (km)',
        'vehicle_types': 'Vehicle Type',
        'weather_impact': 'Weather Condition',
        'traffic_impact': 'Traffic Density',
        'personnel_ratings': 'Personnel Rating (⭐)'
    }
    return labels.get(comparison_type, 'Scenario')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check."""
    models_loaded = model is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'model_not_loaded',
        'model_loaded': models_loaded,
        'features_count': len(feature_columns) if feature_columns else 0,
        'encoders_count': len(encoders) if encoders else 0
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    # Final model load check for production
    if not load_models_safely():
        print("⚠️  Starting without models - predictions will fail!")
    
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    
    print(f"🚀 Delivery Prediction API on http://0.0.0.0:{port}")
    print(f"   Models: {'✅ Loaded' if model else '❌ Missing'}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
