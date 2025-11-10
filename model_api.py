"""
Week 10 - Component 2: Model Inference API using Flask (Corrected)

This corrected version fixes the TypeError by explicitly casting the NumPy
prediction results to standard Python floats before saving them to the database.
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import sqlite3

# --- 1. Initialization ---
app = Flask(__name__)
DB_NAME = 'fleet_health.db'

# --- 2. Load Models ---
try:
    print("Loading pre-trained models...")
    SOH_MODEL = joblib.load('models/optimized_soh_xgb_model.joblib')
    SOP_MODEL = joblib.load('models/sop_model_final.joblib')
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"--- CRITICAL ERROR: Model file not found ---")
    print(f"Error details: {e}")
    SOH_MODEL = None
    SOP_MODEL = None


# --- 3. Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            predicted_soh REAL,
            predicted_sop REAL,
            health_score REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()


# --- 4. API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if SOH_MODEL is None or SOP_MODEL is None:
        return jsonify({'error': 'Models are not loaded. Server is not ready.'}), 500

    try:
        data = request.get_json()
        print(f"Received data for prediction: {data}")

        features_df = pd.DataFrame([data])

        # --- SoH Prediction ---
        soh_feature_order = SOH_MODEL.feature_names_in_
        predicted_soh = SOH_MODEL.predict(features_df[soh_feature_order])[0]

        # --- SoP Prediction ---
        sop_feature_order = SOP_MODEL.feature_names_in_
        predicted_sop = SOP_MODEL.predict(features_df[sop_feature_order])[0]

        # --- Health Score Calculation ---
        health_score = (1 - predicted_soh) + (1 - (predicted_sop / 50000))

        if health_score > 0.8:
            status = "Priority Maintenance"
        elif health_score > 0.5:
            status = "Monitor"
        else:
            status = "Healthy"

        # --- THE FIX: Convert NumPy types to standard Python floats ---
        db_vehicle_id = data['vehicle_id']
        db_predicted_soh = float(predicted_soh)
        db_predicted_sop = float(predicted_sop)
        db_health_score = float(health_score)
        db_status = status

        # --- Save to Database ---
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO health_records (vehicle_id, predicted_soh, predicted_sop, health_score, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (db_vehicle_id, db_predicted_soh, db_predicted_sop, db_health_score, db_status))
        conn.commit()
        conn.close()

        # --- Return Response ---
        response = {
            'vehicle_id': db_vehicle_id,
            'predicted_soh': db_predicted_soh,
            'predicted_sop': db_predicted_sop,
            'health_score': db_health_score,
            'status': db_status
        }
        return jsonify(response)

    except Exception as e:
        print(f"--- An Error Occurred during prediction ---")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    init_db()
    app.run(debug=True)