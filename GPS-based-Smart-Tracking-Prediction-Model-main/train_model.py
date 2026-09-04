"""
Training script for delivery time prediction model.
Trains XGBoost model and saves artifacts to model/ directory.
COMPLETE VALIDATION + SPACE CLEANING
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from preprocess import preprocess_data

warnings.filterwarnings('ignore')

# Ensure model directory exists
os.makedirs('model', exist_ok=True)
data_path = 'data'

print("🚀 Delivery Time Prediction - Training Pipeline")

# =============================================================================
# LOAD & PROCESS TRAINING DATA
# =============================================================================
print("\n📊 Loading training data...")
csv_path = os.path.join(data_path, 'train.csv')
if not os.path.exists(csv_path):
    csv_path = 'train.csv'

df_train = pd.read_csv(csv_path)
df_train_processed, encoders = preprocess_data(df_train, is_training=True)
print(f"✅ Train shape: {df_train_processed.shape}")

# =============================================================================
# PREPARE FEATURES & TARGETS (X must be defined first)
# =============================================================================
X = df_train_processed.drop('Time_taken(min)', axis=1)
y = df_train_processed['Time_taken(min)']

# =============================================================================
# 🧪 COMPLETE VALIDATION: ALL CATEGORICAL FIELDS + SPACE CLEANING
# =============================================================================
print("\n" + "="*80)
print("🔍 COMPLETE VALIDATION: Field Mappings & Categorical Encoders")
print("="*80)

# 1. RAW DATA ANALYSIS
print("\n📍 1. RAW DATA CATEGORY COUNTS:")
raw_cats = {}
for col in ['City', 'Weatherconditions', 'Roadtrafficdensity', 'Typeoforder', 'Typeofvehicle', 'Festival']:
    if col in df_train.columns:
        raw_cats[col] = df_train[col].value_counts().to_dict()
        print(f"\n{col}:")
        for k, v in list(raw_cats[col].items())[:5]:
            print(f"  {repr(k)}: {v}")

# 2. ALL ENCODERS - FULL DETAILS
print("\n📋 2. ALL CATEGORICAL ENCODERS (SAVED FOR PREDICTION):")
print(f"Total encoders: {len(encoders)}")
for col, le in encoders.items():
    print(f"\n🔑 {col}: {len(le.classes_)} classes")
    print(f"    Classes: {[repr(c) for c in le.classes_]}")
    # Show cleaned versions
    clean_classes = [c.strip() for c in le.classes_]
    unique_clean = list(set(clean_classes))
    print(f"    Cleaned: {len(unique_clean)} unique → {unique_clean}")

# 3. PREDICTION PIPELINE TEST
print("\n🧪 3. PREDICTION PIPELINE VALIDATION:")
sample_test = df_train.head(5).copy()
sample_proc, _ = preprocess_data(sample_test, is_training=False, encoders=encoders)

# Feature alignment
missing = set(X.columns) - set(sample_proc.columns)
extra = set(sample_proc.columns) - set(X.columns) - {'Time_taken(min)'}
print(f"✅ Features: {len(X.columns)} total | Match: {len(missing)==0 and len(extra)==0}")
if missing or extra:
    print(f"   ⚠️  Missing: {missing}")
    print(f"   ⚠️  Extra: {extra}")
else:
    print("   ✅ PERFECT ALIGNMENT")

# 4. DATA QUALITY CHECKS
print("\n📊 4. FEATURE RANGES & STATS:")
key_features = ['City', 'Weather_conditions', 'Road_traffic_density', 'distance', 'order_prep_time']
ranges = X[key_features].describe().loc[['min', 'max', 'mean']].round(2)
print(ranges)

print("\n✅ VALIDATION COMPLETE - READY FOR TRAINING!")
print("="*80)

# =============================================================================
# SPLIT & SCALE DATA
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# TRAIN MODEL
# =============================================================================
print("\n🔧 Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=100,  # Increased for better performance
    max_depth=6,
    random_state=42,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = np.mean(np.abs(y_pred - y_test))

print(f"\n📈 MODEL PERFORMANCE:")
print(f"✅ R² Score: {r2:.3f}")
print(f"✅ RMSE: {rmse:.2f} minutes")
print(f"✅ MAE: {mae:.2f} minutes")
print(f"✅ Prediction range: {y_pred.min():.1f} - {y_pred.max():.1f} min")

# Feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🏆 TOP 5 FEATURES:")
print(importance_df.head().to_string(index=False))

# =============================================================================
# SAVE MODEL ARTIFACTS
# =============================================================================
print("\n💾 Saving model artifacts...")
joblib.dump(model, os.path.join('model', 'delivery_model.joblib'))
joblib.dump(scaler, os.path.join('model', 'delivery_scaler.joblib'))
joblib.dump(encoders, os.path.join('model', 'delivery_encoders.joblib'))
joblib.dump(X.columns.tolist(), os.path.join('model', 'feature_columns.joblib'))
joblib.dump(importance_df, os.path.join('model', 'feature_importance.joblib'))

print("✅ Model artifacts saved to model/ directory")

# =============================================================================
# BATCH PREDICTION ON TEST SET
# =============================================================================
print("\n🔮 Generating test set predictions...")
try:
    test_path = os.path.join(data_path, 'test.csv')
    if not os.path.exists(test_path):
        test_path = 'test.csv'
    
    df_test = pd.read_csv(test_path)
    test_ids = df_test['ID'].copy()
    
    df_test_processed, _ = preprocess_data(df_test, is_training=False, encoders=encoders)
    
    # Align features
    for col in X.columns:
        if col not in df_test_processed.columns:
            df_test_processed[col] = 0
    df_test_processed = df_test_processed[X.columns]
    
    # Scale and predict
    X_test_scaled = scaler.transform(df_test_processed)
    test_predictions = model.predict(X_test_scaled)
    test_predictions = [max(pred, 0) for pred in test_predictions]
    
    # Save submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'Time_taken(min)': test_predictions
    })
    submission.to_csv(os.path.join(data_path, 'submission.csv'), index=False)
    
    print(f"✅ Generated {len(test_predictions)} predictions")
    print(f"📊 Test range: {min(test_predictions):.1f} - {max(test_predictions):.1f} minutes")
    print(f"✅ Submission saved to data/submission.csv")
    
except FileNotFoundError:
    print("⚠️  test.csv not found - skipping submission generation")

print("\n🎉 Training pipeline complete!")
print("\n📁 Saved files:")
print("   model/delivery_model.joblib")
print("   model/delivery_scaler.joblib") 
print("   model/delivery_encoders.joblib")
print("   model/feature_columns.joblib")
print("   model/feature_importance.joblib")
