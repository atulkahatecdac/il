import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import json

print("=====================================================")
print("          THE 6-STEP FEATURE ENGINEERING LAB         ")
print("=====================================================\n")

# --- THE RAW DATA ---
# A messy dump of user profiles and their raw transaction logs.
users_df = pd.DataFrame({
    'user_id': [101, 102, 103, 104],
    'age': [25, np.nan, 32, 150],        # Note: missing age, impossible age
    'city': ['Pune', 'Mumbai', 'Pune', 'Delhi'],
    'annual_income': [50000, 60000, 120000, 9999999] # Note: extreme outlier
})

transactions_df = pd.DataFrame({
    'user_id': [101, 101, 102, 103, 103, 103],
    'purchase_amount': [1200, 800, 1500, 300, 450, 600]
})

print("RAW DATA STATE:")
print(users_df.head(), "\n")

# --- STEP 1: CLEANING (Handling missing/outliers) ---
print("[STEP 1] CLEANING: Fixing reality's mess...")
# Fill missing age with median, cap impossible ages at 99
users_df['age'] = users_df['age'].fillna(users_df['age'].median())
users_df['age'] = np.where(users_df['age'] > 99, 99, users_df['age'])

# Remove the insane billionaire outlier so they don't skew our ML model
users_df = users_df[users_df['annual_income'] < 1000000]
print("-> Cleaned ages and removed extreme income outliers.\n")

# --- STEP 2: TRANSFORMATION (Converting formats) ---
print("[STEP 2] TRANSFORMATION: Scaling the numbers...")
# ML models hate big numbers mixed with small numbers. We scale income down.
scaler = StandardScaler()
users_df['scaled_income'] = scaler.fit_transform(users_df[['annual_income']])
print("-> Income converted into scaled mathematical weights.\n")

# --- STEP 3: ENCODING (Converting categorical) ---
print("[STEP 3] ENCODING: Translating words to numbers...")
# ML cannot read "Pune" or "Mumbai". We convert cities to binary columns (One-Hot).
encoded_cities = pd.get_dummies(users_df['city'], prefix='city', dtype=int)
users_df = pd.concat([users_df, encoded_cities], axis=1)
users_df = users_df.drop('city', axis=1) # Drop the original text column
print(f"-> Created new binary columns: {list(encoded_cities.columns)}\n")

# --- STEP 4: AGGREGATION (Creating summary features) ---
print("[STEP 4] AGGREGATION: Summarizing user behavior...")
# We don't want raw transactions; we want 'Average Purchase Amount' per user.
user_spend = transactions_df.groupby('user_id')['purchase_amount'].mean().reset_index()
user_spend.rename(columns={'purchase_amount': 'avg_spend'}, inplace=True)

# Merge this new summary feature back into our main profile
features_df = pd.merge(users_df, user_spend, on='user_id', how='left')
features_df['avg_spend'] = features_df['avg_spend'].fillna(0) # Handle users with 0 purchases
print("-> Calculated and attached 'avg_spend' per user.\n")

# --- STEP 5: FEATURE SELECTION (Reducing dimensionality) ---
print("[STEP 5] FEATURE SELECTION: Dropping the dead weight...")
# We drop columns that don't help the ML model (like the raw unscaled income and user_id)
final_features = features_df.drop(['user_id', 'annual_income'], axis=1)
print(f"-> Final selected features for the model: {list(final_features.columns)}\n")

# --- STEP 6: FEATURE STORE (Centralized storage) ---
print("[STEP 6] FEATURE STORE: Saving for production use...")
# In production, features are saved to a fast database (like Redis) so 
# live applications can access them instantly without re-calculating everything.
offline_store_path = "processed_features.parquet"
final_features.to_parquet(offline_store_path)
print(f"-> Features successfully written to '{offline_store_path}' (Simulated Offline Store).\n")

print("=====================================================")
print("FINAL ML-READY FEATURE SET:")
print(final_features)
print("=====================================================")