"""
Week 10 - Component 1: Feature Engineering Service (Final Corrected Version)

This script automates the feature engineering pipeline. This final corrected
version is a comprehensive solution that generates the distinct feature sets
required by BOTH the State of Health (SoH) and State of Power (SoP) models.

It resolves the final KeyError by adding a new section to calculate the
short-term, rolling-window statistics (e.g., 'current_std_10s') that the
pre-trained SoP model expects.
"""
import pandas as pd
import numpy as np
import os
import requests
import shutil

print("--- Feature Engineering Service Started ---")

# --- Configuration ---
INCOMING_DIR = 'incoming_trip_data'
ARCHIVE_DIR = 'archive'
API_URL = 'http://127.0.0.1:5000/predict'

# Ensure directories exist
if not os.path.exists(INCOMING_DIR):
    os.makedirs(INCOMING_DIR)
if not os.path.exists(ARCHIVE_DIR):
    os.makedirs(ARCHIVE_DIR)


def engineer_features(df, vehicle_id):
    """
    Transforms raw trip data into a single, comprehensive feature vector for prediction.
    This function now produces all features required by both the SoH and SoP models,
    using the exact names that the models expect.
    """
    print(f"Engineering features for {vehicle_id}...")

    # --- 1. SoH Model Feature Calculations (Trip-Level Summaries) ---
    # These features summarize the entire trip and are used to predict the long-term
    # State of Health (SoH). They are based on the logic from Week 3.

    print("Calculating trip-level summary features for SoH model...")
    features = {
        'vehicle_id': vehicle_id,
        # Common features also used by SoP model's logic
        'avg_current': df['current_A'].mean(),
        'avg_voltage': df['voltage_V'].mean(),
        'current_std': df['current_A'].std(),
        'voltage_std': df['voltage_V'].std(),
        'mean_max_temp': df['temperature_C'].mean(),
        'dod': df['soc'].iloc[0] - df['soc'].iloc[-1],

        # Exact feature names expected by the SoH model
        'voltage_V_mean': df['voltage_V'].mean(),
        'current_A_mean': df['current_A'].mean(),
        'temperature_C_mean': df['temperature_C'].mean(),
        'temperature_C_max': df['temperature_C'].max(),
        'discharge_time_s': df['time_s'].iloc[-1] - df['time_s'].iloc[0],
        'delta_T_C': df['temperature_C'].iloc[-1] - df['temperature_C'].iloc[0],
    }

    # Calculate the more complex 'voltage_drop_time_s' feature for SoH
    try:
        v_max, v_min = df['voltage_V'].max(), df['voltage_V'].min()
        high_thresh = v_min + (v_max - v_min) * 0.9
        low_thresh = v_min + (v_max - v_min) * 0.2
        t_high = df[df['voltage_V'] >= high_thresh]['time_s'].min()
        t_low = df[df['voltage_V'] <= low_thresh]['time_s'].max()
        if pd.notna(t_high) and pd.notna(t_low) and t_low > t_high:
            features['voltage_drop_time_s'] = t_low - t_high
        else:
            features['voltage_drop_time_s'] = features['discharge_time_s']
    except Exception:
        features['voltage_drop_time_s'] = features['discharge_time_s']

    # --- 2. SoP Model Feature Calculations (Rolling Window Statistics) ---
    # These features are crucial for predicting short-term State of Power (SoP).
    # They capture the recent dynamics of the battery, which is highly predictive of
    # its immediate power capability. We use a 10-second rolling window as developed
    # during the SoP model training week.

    print("Calculating rolling-window features for SoP model...")
    window_size = 10  # 10-second window

    # Calculate rolling statistics for each time step
    df['current_std_10s'] = df['current_A'].rolling(window=window_size, min_periods=1).std()
    df['mean_temp_10s'] = df['temperature_C'].rolling(window=window_size, min_periods=1).mean()

    # Calculate rolling voltage slope (dV/dt)
    # This feature is a powerful indicator of immediate voltage stability under load.
    voltage_rolling = df['voltage_V'].rolling(window=window_size, min_periods=1)
    time_rolling = df['time_s'].rolling(window=window_size, min_periods=1)

    # Use a lambda function to calculate slope (rise/run) for each window
    # We add a small epsilon to the denominator to prevent division by zero.
    voltage_slope = (voltage_rolling.max() - voltage_rolling.min()) / (time_rolling.max() - time_rolling.min() + 1e-6)
    df['voltage_slope_10s'] = voltage_slope

    # Since our models expect a single value per trip, we summarize these
    # time-series features, for example by taking their mean. This captures the
    # average short-term volatility and behavior during the trip.
    features['current_std_10s'] = df['current_std_10s'].mean()
    features['mean_temp_10s'] = df['mean_temp_10s'].mean()
    features['voltage_slope_10s'] = df['voltage_slope_10s'].mean()

    # Add the remaining simple features the SoP model expects
    features['test_time_s'] = features['discharge_time_s']
    features['current_A'] = features['current_A_mean']
    features['voltage_V'] = features['voltage_V_mean']
    features['temperature_C'] = features['temperature_C_mean']
    features['test_temperature_C'] = features['temperature_C_mean']

    # Fill any potential NaN values that might result from calculations
    for key, value in features.items():
        if pd.isna(value):
            features[key] = 0.0

    print(f"Features engineered: {features}")
    return features


def process_incoming_files():
    """Main function to find, process, and archive new trip files."""
    trip_files = [f for f in os.listdir(INCOMING_DIR) if f.endswith('.csv')]

    if not trip_files:
        print("No new trip files to process.")
        return

    for filename in trip_files:
        file_path = os.path.join(INCOMING_DIR, filename)
        print(f"\nFound new file: {filename}")

        try:
            vehicle_id = os.path.splitext(filename)[0].replace('_trip_data', '')
            trip_df = pd.read_csv(file_path)
            feature_payload = engineer_features(trip_df, vehicle_id)

            print(f"Sending feature payload to API: {API_URL}")
            response = requests.post(API_URL, json=feature_payload)

            if response.status_code == 200:
                print("Successfully received prediction from API.")
                print(f"API Response: {response.json()}")
                shutil.move(file_path, os.path.join(ARCHIVE_DIR, filename))
                print(f"Archived processed file: {filename}")
            else:
                print(f"--- API Error ---")
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")

        except Exception as e:
            print(f"--- An Error Occurred processing {filename} ---")
            import traceback
            traceback.print_exc()
            print(f"Error details: {e}")


if __name__ == "__main__":
    process_incoming_files()
    print("\n--- Feature Engineering Service Finished ---")