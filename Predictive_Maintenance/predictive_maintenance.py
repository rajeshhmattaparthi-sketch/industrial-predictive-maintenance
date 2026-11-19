"""
Industrial Predictive Maintenance - Simple Base Project
-------------------------------------------------------
Goal: Predict whether a machine will FAIL soon, using sensor readings
(temperature, vibration, pressure, rpm, current, runtime hours).

Steps:
 1. Generate synthetic sensor data (replace with your real CSV later)
 2. Feature engineering (rolling averages per machine)
 3. Train a Random Forest classifier
 4. Evaluate (accuracy, confusion matrix, ROC-AUC, feature importance)
 5. Save the model and predict on new readings

Install:  pip install numpy pandas scikit-learn matplotlib joblib
Run:      python predictive_maintenance.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

SENSORS = ["temperature", "vibration", "pressure", "rpm", "current"]


# ---------------------------------------------------------------
# 1. DATA GENERATION (simulated machines that degrade over time)
# ---------------------------------------------------------------
def generate_data(n_machines=30, hours_per_machine=500, horizon=24):
    """
    Each machine runs until it fails. Sensor values drift up as the
    machine approaches failure. Label = 1 if failure occurs within
    the next `horizon` hours, else 0.
    """
    frames = []
    for m in range(n_machines):
        life = int(rng.integers(300, hours_per_machine))  # hours until failure
        t = np.arange(life)
        wear = (t / life) ** 3  # 0 -> 1, grows fast near end of life

        df = pd.DataFrame({
            "machine_id": m,
            "hour": t,
            "temperature": 70 + 25 * wear + rng.normal(0, 2, life),
            "vibration":   0.3 + 1.2 * wear + rng.normal(0, 0.05, life),
            "pressure":    100 - 15 * wear + rng.normal(0, 1.5, life),
            "rpm":         1500 - 120 * wear + rng.normal(0, 15, life),
            "current":     10 + 4 * wear + rng.normal(0, 0.3, life),
        })
        df["hours_to_failure"] = life - t
        df["failure_soon"] = (df["hours_to_failure"] <= horizon).astype(int)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------
def add_features(df, window=12):
    """Rolling mean/std per machine capture trends, not just snapshots."""
    df = df.sort_values(["machine_id", "hour"]).copy()
    for col in SENSORS:
        grp = df.groupby("machine_id")[col]
        df[f"{col}_mean"] = grp.transform(
            lambda s: s.rolling(window, min_periods=1).mean())
        df[f"{col}_std"] = grp.transform(
            lambda s: s.rolling(window, min_periods=1).std()).fillna(0)
    return df


def feature_columns():
    cols = list(SENSORS)
    cols += [f"{c}_mean" for c in SENSORS] + [f"{c}_std" for c in SENSORS]
    return cols


# ---------------------------------------------------------------
# 3. TRAIN + 4. EVALUATE
# ---------------------------------------------------------------
def train_and_evaluate(df):
    features = feature_columns()

    # Split by MACHINE (not by row) so the model is tested on unseen machines
    machine_ids = df["machine_id"].unique()
    train_ids, test_ids = train_test_split(
        machine_ids, test_size=0.25, random_state=RANDOM_STATE)
    train = df[df["machine_id"].isin(train_ids)]
    test = df[df["machine_id"].isin(test_ids)]

    X_train, y_train = train[features], train["failure_soon"]
    X_test, y_test = test[features], test["failure_soon"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",   # failures are rare
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    # Feature importance plot
    imp = pd.Series(model.feature_importances_, index=features)
    imp = imp.sort_values().tail(10)
    plt.figure(figsize=(8, 5))
    imp.plot(kind="barh", color="steelblue")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print("Saved: feature_importance.png")

    return model


# ---------------------------------------------------------------
# 5. SAVE MODEL + PREDICT ON NEW DATA
# ---------------------------------------------------------------
def predict_machine_health(model, recent_readings: pd.DataFrame):
    """
    recent_readings: last ~12+ rows of sensor data for ONE machine
    (columns = SENSORS). Returns failure probability and a status.
    """
    d = recent_readings.copy()
    d["machine_id"] = 0
    d["hour"] = np.arange(len(d))
    d = add_features(d)
    latest = d.iloc[[-1]][feature_columns()]
    prob = model.predict_proba(latest)[0, 1]

    if prob >= 0.7:
        status = "CRITICAL - schedule maintenance immediately"
    elif prob >= 0.4:
        status = "WARNING - plan maintenance soon"
    else:
        status = "HEALTHY"
    return prob, status


def main():
    print("Generating data...")
    df = generate_data()
    df = add_features(df)
    df.to_csv("sensor_data.csv", index=False)
    print(f"Dataset: {df.shape[0]} rows, "
          f"failure rate = {df['failure_soon'].mean():.2%}")

    model = train_and_evaluate(df)
    joblib.dump(model, "maintenance_model.joblib")
    print("Saved: maintenance_model.joblib")

    # ---- Demo: a healthy machine vs. a degrading one ----
    n = 20
    healthy = pd.DataFrame({
        "temperature": 70 + rng.normal(0, 2, n),
        "vibration":   0.3 + rng.normal(0, 0.05, n),
        "pressure":    100 + rng.normal(0, 1.5, n),
        "rpm":         1500 + rng.normal(0, 15, n),
        "current":     10 + rng.normal(0, 0.3, n),
    })
    failing = pd.DataFrame({
        "temperature": 93 + rng.normal(0, 2, n),
        "vibration":   1.45 + rng.normal(0, 0.05, n),
        "pressure":    86 + rng.normal(0, 1.5, n),
        "rpm":         1385 + rng.normal(0, 15, n),
        "current":     14 + rng.normal(0, 0.3, n),
    })

    for name, data in [("Machine A", healthy), ("Machine B", failing)]:
        p, status = predict_machine_health(model, data)
        print(f"\n{name}: failure probability = {p:.2%} -> {status}")


if __name__ == "__main__":
    main()
