# explainability.py

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "models/fraud_rf_model.pkl"      # Path to your trained model
DATA_PATH = "data/processed/test_features.csv"  # Features used for SHAP analysis
OUTPUT_DIR = "explainability_plots"          # Where to save plots

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Load Model and Data
# -------------------------------
print("[INFO] Loading model and data...")
model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(DATA_PATH)

# -------------------------------
# SHAP Explainer
# -------------------------------
print("[INFO] Creating SHAP explainer...")
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_test)

# -------------------------------
# Summary Plot
# -------------------------------
print("[INFO] Generating summary plot...")
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "summary_plot.png"))
plt.close()

# -------------------------------
# Feature Importance Plot
# -------------------------------
print("[INFO] Generating feature importance plot...")
# Use mean absolute SHAP values
mean_abs_shap = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values(by='mean_abs_shap', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(mean_abs_shap['feature'], mean_abs_shap['mean_abs_shap'])
plt.gca().invert_yaxis()
plt.title("Feature Importance (mean |SHAP value|)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.close()

print("[INFO] SHAP explainability plots saved to:", OUTPUT_DIR)
