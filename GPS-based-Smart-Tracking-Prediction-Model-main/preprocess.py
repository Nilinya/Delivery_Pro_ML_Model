import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic


def preprocess_data(df, is_training=True, encoders=None):
    """
    Preprocess delivery data for model training or prediction.
    
    Args:
        df: Input DataFrame
        is_training: If True, returns full processed data with encoders.
                    If False, uses provided encoders for prediction.
        encoders: Dict of LabelEncoder objects (required when is_training=False)
    
    Returns:
        Tuple of (processed_df, encoders_dict)
    """
    df = df.copy()
    
    # Drop unnecessary columns
    df = df.drop(['ID', 'Delivery_person_ID'], axis=1, errors='ignore')
    
    # Replace NaN strings with actual NaN
    df = df.replace('NaN ', np.nan)
    df = df.replace('NaN', np.nan)
    
    # Handle Time_taken(min) - only for training
    if 'Time_taken(min)' in df.columns:
        df['Time_taken(min)'] = df['Time_taken(min)'].astype(str)
        df['Time_taken(min)'] = df['Time_taken(min)'].str.extract(r'(\d+\.?\d*)').astype(float)
        df = df.dropna(subset=['Time_taken(min)'])
    
    # Handle Weather conditions
    if 'Weatherconditions' in df.columns:
        df['Weather_conditions'] = df['Weatherconditions'].apply(
            lambda x: str(x).split()[-1] if pd.notna(x) and 'conditions' in str(x) else str(x) if pd.notna(x) else 'Unknown'
        )
    else:
        df['Weather_conditions'] = 'Unknown'
    
    # Convert numeric columns
    numeric_cols = [
        'Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries',
        'Restaurant_latitude', 'Restaurant_longitude',
        'Delivery_location_latitude', 'Delivery_location_longitude',
        'Vehicle_condition'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Date handling
    if 'Order_Date' in df.columns:
        try:
            df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
        except:
            df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
        
        df['day_of_week'] = df['Order_Date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['Order_Date'].dt.month
        df['day'] = df['Order_Date'].dt.day
        df['quarter'] = df['Order_Date'].dt.quarter
        
        df['day_of_week'] = df['day_of_week'].fillna(0).astype(int)
        df['is_weekend'] = df['is_weekend'].fillna(0).astype(int)
        df['month'] = df['month'].fillna(1).astype(int)
        df['day'] = df['day'].fillna(1).astype(int)
        df['quarter'] = df['quarter'].fillna(1).astype(int)
        
        df = df.drop('Order_Date', axis=1)
    else:
        df['day_of_week'] = 0
        df['is_weekend'] = 0
        df['month'] = 1
        df['day'] = 1
        df['quarter'] = 1
    
    # Distance calculation
    required_coords = [
        'Restaurant_latitude', 'Restaurant_longitude',
        'Delivery_location_latitude', 'Delivery_location_longitude'
    ]
    coords_present = all(col in df.columns for col in required_coords)
    
    if coords_present:
        def calculate_distance(row):
            try:
                if (pd.notna(row['Restaurant_latitude']) and pd.notna(row['Restaurant_longitude']) and
                    pd.notna(row['Delivery_location_latitude']) and pd.notna(row['Delivery_location_longitude'])):
                    return geodesic(
                        (row['Restaurant_latitude'], row['Restaurant_longitude']),
                        (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                    ).km
                else:
                    return np.nan
            except:
                return np.nan
        
        df['distance'] = df.apply(calculate_distance, axis=1)
    else:
        df['distance'] = np.nan
    
    # Prep time calculation
    if 'Time_Orderd' in df.columns and 'Time_Order_picked' in df.columns:
        def parse_time_to_minutes(time_val):
            try:
                if pd.isna(time_val):
                    return np.nan
                time_str = str(time_val).strip()
                if len(time_str) == 6 and time_str.isdigit():
                    hours = int(time_str[0:2])
                    minutes = int(time_str[2:4])
                    seconds = int(time_str[4:6])
                    return hours * 60 + minutes + seconds / 60
                else:
                    time_parts = re.findall(r'\d+', time_str)
                    if len(time_parts) >= 2:
                        hours = int(time_parts[0])
                        minutes = int(time_parts[1])
                        seconds = int(time_parts[2]) if len(time_parts) > 2 else 0
                        return hours * 60 + minutes + seconds / 60
                    else:
                        return np.nan
            except:
                return np.nan
        
        df['order_time_minutes'] = df['Time_Orderd'].apply(parse_time_to_minutes)
        df['picked_time_minutes'] = df['Time_Order_picked'].apply(parse_time_to_minutes)
        
        df['order_prep_time'] = df.apply(
            lambda row: (row['picked_time_minutes'] - row['order_time_minutes'])
            if pd.notna(row['picked_time_minutes']) and pd.notna(row['order_time_minutes'])
            and row['picked_time_minutes'] >= row['order_time_minutes']
            else (row['picked_time_minutes'] + 24*60 - row['order_time_minutes'])
            if pd.notna(row['picked_time_minutes']) and pd.notna(row['order_time_minutes'])
            else np.nan,
            axis=1
        )
        
        df = df.drop(['Time_Orderd', 'Time_Order_picked', 'order_time_minutes', 'picked_time_minutes'],
                     axis=1, errors='ignore')
    else:
        df['order_prep_time'] = 15.0
    
    # Fill NaN values with defaults
    fill_values = {
        'Delivery_person_Age': 30,
        'Delivery_person_Ratings': 4.5,
        'multiple_deliveries': 0,
        'distance': df['distance'].median() if 'distance' in df.columns and not df['distance'].isna().all() else 5.0,
        'order_prep_time': df['order_prep_time'].median() if 'order_prep_time' in df.columns and not df['order_prep_time'].isna().all() else 15.0,
        'Weather_conditions': 'Unknown',
        'Road_traffic_density': 'Medium',
        'Type_of_order': 'Snack',
        'Type_of_vehicle': 'motorcycle',
        'Festival': 'No',
        'City': 'Urban',
        'Vehicle_condition': 1,
        'Restaurant_latitude': df['Restaurant_latitude'].median() if 'Restaurant_latitude' in df.columns else 0,
        'Restaurant_longitude': df['Restaurant_longitude'].median() if 'Restaurant_longitude' in df.columns else 0,
        'Delivery_location_latitude': df['Delivery_location_latitude'].median() if 'Delivery_location_latitude' in df.columns else 0,
        'Delivery_location_longitude': df['Delivery_location_longitude'].median() if 'Delivery_location_longitude' in df.columns else 0,
        'Weatherconditions': 'Unknown'
    }
    
    for col, fill_val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)
    
    # Categorical encoding
    potential_cat_cols = [
        'Weather_conditions', 'Road_traffic_density', 'Type_of_order',
        'Type_of_vehicle', 'Festival', 'City', 'Weatherconditions'
    ]
    cat_cols = [col for col in potential_cat_cols if col in df.columns]
    
    new_encoders = {}
    for col in cat_cols:
        if col in df.columns:
            if is_training:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                new_encoders[col] = le
            else:
                # Use provided encoders for prediction
                df[col] = df[col].astype(str)
                if encoders and col in encoders:
                    le = encoders[col]
                    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    # Ensure all columns are numeric
    for col in df.columns:
        if col != 'Time_taken(min)' and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df, (new_encoders if is_training else encoders)
