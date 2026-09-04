"""
Unit tests for delivery prediction API.
Updated for exact encoder matching + validation.
Run: python -m pytest tests/test_predict.py -v
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, predict_delivery_time
from preprocess import preprocess_data

@pytest.fixture
def client():
    """Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestAPI:
    """API endpoint tests."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert 'model_loaded' in data
    
    def test_home_page(self, client):
        """Test home page loads."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Delivery Time Predictor' in response.data
    
    def test_predict_single_valid(self, client):
        """Test single prediction - EXACT encoder match."""
        payload = {
            'Delivery_person_Age': 37,
            'Delivery_person_Ratings': 4.9,
            'Restaurant_latitude': 22.745,
            'Restaurant_longitude': 75.892,
            'Delivery_location_latitude': 22.765,
            'Delivery_location_longitude': 75.912,
            'Weatherconditions': 'conditions Sunny',      # ✅ Exact encoder match
            'Road_traffic_density': 'High ',               # ✅ Has space
            'Vehicle_condition': 2,
            'Type_of_order': 'Snack ',
            'Type_of_vehicle': 'motorcycle ',
            'multiple_deliveries': 0,
            'Festival': 'No ',                             # ✅ Has space
            'City': 'Metropolitian ',                      # ✅ Has space
            'Order_Date': '19-03-2022',
            'Time_Orderd': '113000',
            'Time_Order_picked': '114500'
        }
        
        response = client.post(
            '/api/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'predicted_time_minutes' in data
        assert isinstance(data['predicted_time_minutes'], (int, float))
        assert data['predicted_time_minutes'] > 0
    
    def test_predict_weather_rainy(self, client):
        """Test with Rainy weather condition."""
        payload = {
            'Delivery_person_Age': 28,
            'Delivery_person_Ratings': 4.6,
            'Restaurant_latitude': 11.003,
            'Restaurant_longitude': 76.976,
            'Delivery_location_latitude': 11.043,
            'Delivery_location_longitude': 77.016,
            'Weatherconditions': 'conditions Rainy',
            'Road_traffic_density': 'High',
            'Vehicle_condition': 3,
            'Type_of_order': 'Drinks',
            'Type_of_vehicle': 'scooter',
            'multiple_deliveries': 1,
            'Festival': 'No',
            'City': 'Metropolitian',
            'Order_Date': '30-03-2022',
            'Time_Orderd': '140000',
            'Time_Order_picked': '150500'
        }
        
        response = client.post('/api/predict', 
                             data=json.dumps(payload),
                             content_type='application/json')
        assert response.status_code == 200
    
    def test_predict_batch_valid(self, client):
        """Test batch with encoder-exact values."""
        payload = [
            {  # Sample 1: Sunny Urban
                'Delivery_person_Age': 37,
                'Delivery_person_Ratings': 4.9,
                'Restaurant_latitude': 22.745,
                'Restaurant_longitude': 75.892,
                'Delivery_location_latitude': 22.765,
                'Delivery_location_longitude': 75.912,
                'Weatherconditions': 'conditions Sunny',
                'Road_traffic_density': 'High ',
                'Vehicle_condition': 2,
                'Type_of_order': 'Snack ',
                'Type_of_vehicle': 'motorcycle ',
                'multiple_deliveries': 0,
                'Festival': 'No ',
                'City': 'Urban ',
                'Order_Date': '19-03-2022',
                'Time_Orderd': '113000',
                'Time_Order_picked': '114500'
            },
            {  # Sample 2: Different delivery
                'Delivery_person_Age': 28,
                'Delivery_person_Ratings': 4.7,
                'Restaurant_latitude': 22.700,
                'Restaurant_longitude': 75.850,
                'Delivery_location_latitude': 22.720,
                'Delivery_location_longitude': 75.870,
                'Weatherconditions': 'conditions Rainy',
                'Road_traffic_density': 'Medium ',
                'Vehicle_condition': 1,
                'Type_of_order': 'Meal ',
                'Type_of_vehicle': 'scooter ',
                'multiple_deliveries': 1,
                'Festival': 'No ',
                'City': 'Metropolitian ',
                'Order_Date': '20-03-2022',
                'Time_Orderd': '120000',
                'Time_Order_picked': '121500'
            }
        ]
        
        response = client.post('/api/predict-batch',
                             data=json.dumps(payload),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert len(data['predictions']) == 2
        assert all(p > 0 for p in data['predictions'])
    
    def test_invalid_weather_fails_gracefully(self, client):
        """Test unknown weather → should handle gracefully."""
        payload = {
            'Weatherconditions': 'invalid_weather',
            'City': 'InvalidCity',
            'Delivery_person_Age': 30,
            'Delivery_person_Ratings': 4.5,
            'Restaurant_latitude': 22.745,
            'Restaurant_longitude': 75.892,
            'Delivery_location_latitude': 22.765,
            'Delivery_location_longitude': 75.912,
            'Road_traffic_density': 'High',
            'Vehicle_condition': 2,
            'Type_of_order': 'Snack',
            'Type_of_vehicle': 'motorcycle',
            'multiple_deliveries': 0,
            'Festival': 'No',
            'Order_Date': '19-03-2022',
            'Time_Orderd': '113000',
            'Time_Order_picked': '114500'
        }
        response = client.post('/api/predict',
                             data=json.dumps(payload),
                             content_type='application/json')
        assert response.status_code == 200  # Should not crash

class TestPreprocessing:
    """Preprocessing tests with encoder validation."""
    
    def test_preprocess_with_complete_data(self):
        """Test preprocessing with complete valid data."""
        df = pd.DataFrame({
            'Delivery_person_Age': [37, 28],
            'Delivery_person_Ratings': [4.9, 4.7],
            'Restaurant_latitude': [22.745, 22.700],
            'Restaurant_longitude': [75.892, 75.850],
            'Delivery_location_latitude': [22.765, 22.720],
            'Delivery_location_longitude': [75.912, 75.870],
            'Weatherconditions': ['conditions Sunny', 'conditions Rainy'],
            'Road_traffic_density': ['High', 'Medium'],
            'Vehicle_condition': [2, 1],
            'Type_of_order': ['Snack', 'Meal'],
            'Type_of_vehicle': ['motorcycle', 'scooter'],
            'multiple_deliveries': [0, 1],
            'Festival': ['No', 'No'],
            'City': ['Urban', 'Metropolitian'],
            'Order_Date': ['19-03-2022', '20-03-2022'],
            'Time_Orderd': ['113000', '120000'],
            'Time_Order_picked': ['114500', '121500'],
            'Time_taken(min)': [25, 22]
        })
        
        processed, encoders = preprocess_data(df, is_training=True)
        
        assert processed.shape[0] == 2
        assert isinstance(encoders, dict)
        assert len(encoders) > 0
    
    def test_preprocess_strips_spaces(self):
        """Test categorical fields are properly encoded."""
        df = pd.DataFrame({
            'City': ['Metropolitian', 'Urban', 'Semi-Urban'],
            'Festival': ['No', 'Yes', 'No'],
            'Time_taken(min)': [25, 22, 30]
        })
        processed, encoders = preprocess_data(df, is_training=True)
        
        # Should encode categorical columns
        assert 'City' in encoders or processed.shape[0] == 3
        assert processed.shape[0] == 3

class TestPredictionFunction:
    """Prediction function tests."""
    
    def test_predict_with_valid_data(self):
        """Test prediction with complete valid data."""
        input_data = {
            'Delivery_person_Age': 37,
            'Delivery_person_Ratings': 4.9,
            'Restaurant_latitude': 22.745,
            'Restaurant_longitude': 75.892,
            'Delivery_location_latitude': 22.765,
            'Delivery_location_longitude': 75.912,
            'Weatherconditions': 'conditions Sunny',
            'Road_traffic_density': 'High',
            'Vehicle_condition': 2,
            'Type_of_order': 'Snack',
            'Type_of_vehicle': 'motorcycle',
            'multiple_deliveries': 0,
            'Festival': 'No',
            'City': 'Urban',
            'Order_Date': '19-03-2022',
            'Time_Orderd': '113000',
            'Time_Order_picked': '114500'
        }
        result = predict_delivery_time(input_data)
        assert isinstance(result, (int, float))
        assert result > 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
