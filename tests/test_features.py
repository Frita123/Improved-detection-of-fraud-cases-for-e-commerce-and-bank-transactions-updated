# test_features.py
import os
import pandas as pd
import joblib
import numpy as np

# ── BASE DIRECTORIES ──
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── TRAINING FEATURE FILES ──
train_features_map = {
    "Fraud Dataset": "fraud_train_fe.csv",
    "Credit Card Dataset": "creditcard_train_fe.csv"
}

# ── TARGET COLUMN MAPPING ──
target_column_map = {
    "Fraud Dataset": "class",
    "Credit Card Dataset": "Class"
}

# ── MODEL FILES ──
model_file_map = {
    "Fraud Dataset": "fraud_rf.pkl",
    "Credit Card Dataset": "creditcard_rf.pkl"
}

# ── FUNCTION TO LOAD MODEL ──
def load_model(name):
    model_file = model_file_map[name]
    path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    print(f"{name} model loaded: {model_file}")
    return model

# ── TEST FUNCTION ──
def test_datasets():
    for name in train_features_map:
        print(f"\n=== TESTING: {name} ===")

        # Load preprocessed features (same as training)
        fe_path = os.path.join(DATA_DIR, train_features_map[name])
        df = pd.read_csv(fe_path)
        target_col = target_column_map[name]

        # Drop target column
        X = df.drop(columns=[target_col])
        print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features (matches model expectation)")

        # Load model
        model = load_model(name)

        # Test single prediction (first row)
        sample = X.iloc[:1]
        pred = model.predict(sample)
        prob = model.predict_proba(sample)
        print(f"Sample prediction: {pred[0]}, Probabilities: {prob[0]}")

# ── MAIN ──
if __name__ == "__main__":
    test_datasets()
