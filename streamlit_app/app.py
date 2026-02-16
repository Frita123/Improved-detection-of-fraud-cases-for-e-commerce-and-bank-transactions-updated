# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import numpy as np

# ── BASE DIRECTORIES ──
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SHAP_DIR = os.path.join(BASE_DIR, "shap_outputs")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Fraud Detection Interactive Dashboard")

# ── SIDEBAR ──
dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["Fraud Dataset", "Credit Card Dataset"]
)

# ── LOAD DATA ──
@st.cache_data
def load_data(dataset_option):
    file_map = {
        "Fraud Dataset": "fraud_cleaned.csv",
        "Credit Card Dataset": "creditcard_cleaned.csv"
    }
    path = os.path.join(DATA_DIR, file_map[dataset_option])
    return pd.read_csv(path)

df = load_data(dataset_option)
st.success(f"{dataset_option} loaded! Shape: {df.shape}")

# ── SHOW RAW DATA ──
if st.checkbox("Show raw data"):
    st.dataframe(df.head(10))

# ───────────────────── EDA (UNCHANGED) ─────────────────────
st.header("📊 Exploratory Data Analysis (EDA)")

numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if "class" in numeric_features:
    numeric_features.remove("class")
if "Class" in numeric_features:
    numeric_features.remove("Class")

feature_option = st.selectbox("Select numeric feature for distribution", numeric_features)

fig, ax = plt.subplots()
sns.histplot(df[feature_option], kde=True, ax=ax)
ax.set_title(f"{feature_option} Distribution")
st.pyplot(fig)

if st.checkbox("Show correlation heatmap"):
    fig2, ax2 = plt.subplots(figsize=(8,6))
    corr = df[numeric_features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

if "class" in df.columns:
    st.subheader("Class Distribution")
    fig3, ax3 = plt.subplots()
    sns.countplot(x="class", data=df, ax=ax3)
    st.pyplot(fig3)
elif "Class" in df.columns:
    st.subheader("Class Distribution")
    fig3, ax3 = plt.subplots()
    sns.countplot(x="Class", data=df, ax=ax3)
    st.pyplot(fig3)

# ───────────────── MODEL + SHAP ─────────────────
st.header("🤖 Model Performance & Explainability")

model_file_map = {
    "Fraud Dataset": "fraud_rf.pkl",
    "Credit Card Dataset": "creditcard_rf.pkl"
}

model_path = os.path.join(MODEL_DIR, model_file_map[dataset_option])
model = joblib.load(model_path)
st.success("Random Forest model loaded")

# Metrics (hardcoded from previous evaluation)
metrics = {
    "Fraud Dataset": {"F1": 0.686, "AUC_PR": 0.626},
    "Credit Card Dataset": {"F1": 0.834, "AUC_PR": 0.807}
}

st.subheader("📈 Model Metrics")
st.table(pd.DataFrame(metrics[dataset_option], index=["Score"]))

# ── SHAP IMAGE (PRECOMPUTED) ──
st.subheader("🧠 SHAP Beeswarm")

shap_file_map = {
    "Fraud Dataset": "fraud_beeswarm.png",
    "Credit Card Dataset": "credit_beeswarm.png"
}

shap_path = os.path.join(SHAP_DIR, shap_file_map[dataset_option])
if os.path.exists(shap_path):
    st.image(shap_path, use_container_width=True)
else:
    st.warning("SHAP image not found")

# ── FEATURE IMPORTANCE ──
st.subheader("🔹 Global Feature Importance (Random Forest)")

# Get correct feature names
if dataset_option == "Fraud Dataset":
    fe_train_path = os.path.join(DATA_DIR, "fraud_train_fe.csv")
    fe_train_df = pd.read_csv(fe_train_path)
    model_features = fe_train_df.drop(columns=["class"]).columns
elif dataset_option == "Credit Card Dataset":
    fe_train_path = os.path.join(DATA_DIR, "creditcard_train_fe.csv")
    fe_train_df = pd.read_csv(fe_train_path)
    model_features = fe_train_df.drop(columns=["Class"]).columns
else:
    # fallback
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_
    else:
        model_features = df.drop(columns=["class"], errors="ignore").columns

importances = model.feature_importances_

# Align lengths to avoid mismatch
min_len = min(len(model_features), len(importances))
model_features = model_features[:min_len]
importances = importances[:min_len]

feat_df = pd.DataFrame({
    "Feature": model_features,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(10)

st.dataframe(feat_df)

fig_imp, ax_imp = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax_imp)
st.pyplot(fig_imp)

# ── SINGLE PREDICTION ──
st.subheader("🔮 Predict Single Transaction")

row = st.number_input("Row index", 0, len(df)-1, 0)

if st.button("Predict"):
    sample = df.iloc[[row]].copy()

    # Drop target columns if present
    sample = sample.drop(columns=["class", "Class"], errors="ignore")

    # Drop non-numeric columns (like datetime or strings)
    non_numeric_cols = sample.select_dtypes(include=["object"]).columns.tolist()
    sample = sample.drop(columns=non_numeric_cols, errors='ignore')

    # Add missing features required by the model
    for c in model_features:
        if c not in sample.columns:
            sample[c] = 0

    # Reorder columns to match model features
    sample = sample[model_features]

    # Predict
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]
    label = "Fraud 🚨" if pred == 1 else "Normal ✅"

    st.success(f"Prediction: {label}")
    st.metric("Fraud Probability", f"{prob:.3f}")

st.markdown("---")
st.write("⚡ Dashboard by: Firehiwet Zerihun")
