"""
Quick start and testing utilities for the delivery prediction service.
"""

import pandas as pd
import numpy as np
import os


def generate_sample_prediction_data(n_samples=5):
    """
    Generate sample data for testing without requiring actual CSV files.
    
    Returns:
        pd.DataFrame: Sample prediction data
    """
    np.random.seed(42)
    
    return pd.DataFrame({
        'Delivery_person_Age': np.random.randint(18, 60, n_samples),
        'Delivery_person_Ratings': np.round(np.random.uniform(3.5, 5.0, n_samples), 1),
        'Restaurant_latitude': np.round(np.random.uniform(22.7, 22.8, n_samples), 3),
        'Restaurant_longitude': np.round(np.random.uniform(75.8, 75.9, n_samples), 3),
        'Delivery_location_latitude': np.round(np.random.uniform(22.7, 22.8, n_samples), 3),
        'Delivery_location_longitude': np.round(np.random.uniform(75.8, 75.9, n_samples), 3),
        'Weatherconditions': ['conditions Sunny'] * n_samples,
        'Road_traffic_density': np.random.choice(['Low', 'Medium', 'High', 'Jam'], n_samples),
        'Vehicle_condition': np.random.choice([1, 2, 3], n_samples),
        'Type_of_order': np.random.choice(['Snack', 'Meal', 'Drinks', 'Desserts'], n_samples),
        'Type_of_vehicle': np.random.choice(['motorcycle', 'scooter', 'bicycle'], n_samples),
        'multiple_deliveries': np.random.randint(0, 3, n_samples),
        'Festival': np.random.choice(['Yes', 'No'], n_samples),
        'City': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        'Order_Date': ['19-03-2022'] * n_samples,
        'Time_Orderd': ['113000'] * n_samples,
        'Time_Order_picked': ['114500'] * n_samples
    })


def create_sample_data_files(data_dir='data'):
    """
    Create sample train and test CSV files if they don't exist.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(train_path):
        train_data = generate_sample_prediction_data(100)
        train_data.insert(0, 'ID', range(1, len(train_data) + 1))
        train_data.insert(1, 'Delivery_person_ID', range(101, 101 + len(train_data)))
        train_data['Time_taken(min)'] = np.random.randint(15, 60, len(train_data))
        train_data.to_csv(train_path, index=False)
        print(f"✅ Created {train_path}")
    
    if not os.path.exists(test_path):
        test_data = generate_sample_prediction_data(20)
        test_data.insert(0, 'ID', range(1001, 1001 + len(test_data)))
        test_data.insert(1, 'Delivery_person_ID', range(1101, 1101 + len(test_data)))
        test_data.to_csv(test_path, index=False)
        print(f"✅ Created {test_path}")


if __name__ == '__main__':
    create_sample_data_files()
    print("✅ Sample data files created successfully!")
