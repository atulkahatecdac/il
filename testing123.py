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
print(df)


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
print(df)

# --- LAYER 3: FEATURE LAYER (Engineering) ---
# Creating new signals (features) that help the model learn better.
print("[LAYER 3] FEATURE LAYER: Engineering new features...")
# We create a "Critical Risk" feature: True if Temp > 90 AND RAM > 80
df['Critical_Risk'] = np.where((df['CPU_Temp'] > 90) & (df['RAM_Usage'] > 80), 1, 0)
print("Added 'Critical_Risk' feature based on system logic.\n")
print(df)

# --- LAYER 4: TRAINING LAYER (Model Building) ---
print("[LAYER 4] TRAINING LAYER: Training the Random Forest model...")
features = df[['CPU_Temp', 'RAM_Usage', 'Error_Logs', 'Critical_Risk']]
labels = df['Status']

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(features, labels)
print("Model trained successfully on historical data.\n")
