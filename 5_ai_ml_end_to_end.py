import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("=====================================================")
print("      STARTING END-TO-END AI INFRASTRUCTURE          ")
print("=====================================================\n")

# --- LAYER 1: DATA LAYER (Ingestion) ---
# Simulating raw, messy data arriving from a server's API.
# Format: [CPU_Temp, RAM_Usage, Error_Logs, Status]
print("[LAYER 1] DATA LAYER: Ingesting raw sensor data...")
raw_data = {
    'CPU_Temp': [45, 50, 95, None, 48, 105], # Notice the missing data (None)
    'RAM_Usage': [30, 40, 90, 85, 35, 99],
    'Error_Logs': [0, 1, 50, 2, 0, 150],
    'Status': ['Healthy', 'Healthy', 'Crash', 'Crash', 'Healthy', 'Crash']
}
df = pd.DataFrame(raw_data)
print(f"Ingested {len(df)} rows. Notice missing values!\n")

# --- LAYER 2: PROCESSING LAYER (Cleaning) ---
# Production systems must handle bad data without crashing.
print("[LAYER 2] PROCESSING LAYER: Cleaning and transforming...")
mean_cpu = df['CPU_Temp'].mean()
median_cpu = df['CPU_Temp'].median()

print("CPU_Temp Mean:", mean_cpu)
print("CPU_Temp Median:", median_cpu)

# We fill missing CPU temperatures with the median temp to prevent crashes
df['CPU_Temp'] = df['CPU_Temp'].fillna(df['CPU_Temp'].median())
print("Null values handled. Data is now stable.\n")

# --- LAYER 3: FEATURE LAYER (Engineering) ---
# Creating new signals (features) that help the model learn better.
print("[LAYER 3] FEATURE LAYER: Engineering new features...")
# We create a "Critical Risk" feature: True if Temp > 90 AND RAM > 80
df['Critical_Risk'] = np.where((df['CPU_Temp'] > 90) & (df['RAM_Usage'] > 80), 1, 0)
print("Added 'Critical_Risk' feature based on system logic.\n")

# --- LAYER 4: TRAINING LAYER (Model Building) ---
print("[LAYER 4] TRAINING LAYER: Training the Random Forest model...")
features = df[['CPU_Temp', 'RAM_Usage', 'Error_Logs', 'Critical_Risk']]
labels = df['Status']

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(features, labels)
print("Model trained successfully on historical data.\n")

# --- LAYER 5: DEPLOYMENT LAYER (Serving) ---
# Wrapping the model in an inference function (simulating an API endpoint).
print("[LAYER 5] DEPLOYMENT LAYER: Model ready for live requests.")
def predict_server_health(temp, ram, errors):
    # The API must replicate the exact Feature Engineering from Layer 3
    critical_risk = 1 if (temp > 90 and ram > 80) else 0
    live_features = [[temp, ram, errors, critical_risk]]
    
    prediction = model.predict(live_features)[0]
    return prediction

# Test the deployed model
live_temp, live_ram, live_errors = 47, 38, 0
result = predict_server_health(live_temp, live_ram, live_errors)
print(f"API CALL -> Input: Temp={live_temp}, RAM={live_ram} | Prediction: {result}\n")

# --- LAYER 6: MONITORING LAYER (Performance & Drift) ---
# Tracking what happens when the real world changes unexpectedly.
print("[LAYER 6] MONITORING LAYER: Tracking data drift...")
# Suddenly, a new software update causes average CPU temps to hit 150 (Data Drift)
drifted_temp = 150
print(f"Monitoring Alert! Received extreme outlier Temp: {drifted_temp}")

if drifted_temp > df['CPU_Temp'].max() * 1.2:
    print("WARNING: Severe Data Drift Detected! The incoming data no longer looks like the training data. The model's predictions may be unreliable. Triggering retraining protocol.")
else:
    print("System operating within normal parameters.")
    
print("\n=====================================================")
print("                  PIPELINE COMPLETE                  ")
print("=====================================================")