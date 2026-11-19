# Industrial Predictive Maintenance

Predicts whether a machine will fail within the next 24 hours using sensor
data (temperature, vibration, pressure, rpm, current).

## Setup
    pip install -r requirements.txt

## Run
    python predictive_maintenance.py

## What it does
1. Generates synthetic sensor data for 30 machines (replace with your own CSV)
2. Adds rolling mean/std features per machine
3. Trains a Random Forest classifier (split by machine)
4. Prints accuracy, confusion matrix, ROC-AUC; saves feature_importance.png
5. Saves maintenance_model.joblib and demos predictions (HEALTHY/WARNING/CRITICAL)

## Using your own data
Replace `generate_data()` with `pd.read_csv("your_file.csv")`. Required columns:
machine_id, hour, temperature, vibration, pressure, rpm, current, failure_soon.

## Folder
- predictive_maintenance.py : main project code
- output/ : sample outputs from a previous run (data, model, chart)
