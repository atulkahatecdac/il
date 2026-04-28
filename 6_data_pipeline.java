import json
import pandas as pd

print("=====================================================")
print("          THE 5-STAGE DATA PIPELINE LAB              ")
print("=====================================================\n")

# --- STAGE 1: DATA SOURCES (Logs, IoT, APIs) ---
print("[STAGE 1] DATA SOURCES: Generating raw IoT sensor logs...")
raw_logs = [
    {"sensor_id": "T-101", "temp_c": 45.2, "timestamp": "2026-04-19T08:00:00"},
    {"sensor_id": "T-102", "temp_c": None, "timestamp": "2026-04-19T08:05:00"}, # Missing data
    {"sensor_id": "T-103", "temp_c": 999.9, "timestamp": "2026-04-19T08:10:00"}, # Hardware malfunction
    {"sensor_id": "T-101", "temp_c": 46.1, "timestamp": "2026-04-19T08:15:00"}
]
print(f"Generated {len(raw_logs)} raw log entries.\n")

# --- STAGE 2: INGESTION (Kafka, Airflow) ---
print("[STAGE 2] INGESTION: Moving data into the system...")
staging_area = [log for log in raw_logs]
print("Data successfully ingested into memory.\n")

# --- STAGE 3: STORAGE (S3, Data Warehouse) ---
print("[STAGE 3] STORAGE: Persisting raw data...")
file_path = "raw_datalake_dump.json"
with open(file_path, "w") as f:
    json.dump(staging_area, f)
print(f"Raw data safely persisted to '{file_path}'.\n")

# --- STAGE 4: PROCESSING (Spark, Pandas) ---
print("[STAGE 4] PROCESSING: Cleaning and transforming...")
df = pd.read_json(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['temp_c'] = df['temp_c'].fillna(df['temp_c'].median()) # Fix the missing data
print("Data processed: Timestamps formatted and Nulls imputed.\n")

# --- STAGE 5: VALIDATION (Great Expectations) ---
print("[STAGE 5] VALIDATION: Running automated quality checks...")
validation_passed = True

if (df['temp_c'] > 150).any():
    validation_passed = False
    print("QUALITY CHECK FAILED: Temperature values out of bounds (Suspected Sensor Malfunction).")

if validation_passed:
    print("Quality Check Passed! Data is certified for AI/ML training.")
else:
    print("Pipeline halted to prevent model poisoning.")