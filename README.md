# Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions-updated# 🛡️ Improved Detection of Fraud Cases for E-Commerce and Bank Transactions

This project focuses on building robust machine learning models to detect fraudulent transactions using two datasets:

- E-commerce fraud dataset  
- Credit card transaction dataset  

The pipeline includes feature engineering, model training, evaluation, explainability (SHAP), and preparation for deployment via a Streamlit dashboard.

---

## 📌 Project Objectives

- Detect fraudulent transactions with high accuracy
- Compare performance across two datasets
- Provide model explainability using SHAP
- Visualize results via ROC curves and feature importance
- Prepare for real-time inference using a Streamlit dashboard

---

## ✅ Completed Tasks

### Task 1 — Data Preparation & Feature Engineering

✔ Cleaned raw datasets  
✔ Handled missing values  
✔ Encoded categorical variables  
✔ Scaled numerical features  
✔ Generated engineered features  
✔ Split into train/test sets  

Processed files are stored in:

data/processed/


---

### Task 2 — Model Training

Random Forest classifiers were trained separately for:

- Fraud Dataset  
- Credit Card Dataset  

Saved artifacts:

- Trained models (`.pkl`)
- Feature scalers

These are excluded from Git via `.gitignore`.

---

### Task 3 — Model Evaluation

Metrics computed:

- ROC Curve
- AUC Score
- Feature importance (Random Forest)

Visualizations:

✔ ROC curves for both datasets  
✔ Top 15 RF feature importance plots  

---

### Task 4 — Model Explainability (SHAP)

Implemented SHAP for both models:

✔ Beeswarm plots  
✔ SHAP bar charts  
✔ Mean absolute SHAP values  
✔ Combined feature comparison tables  

Key outputs:

- SHAP summary plots  
- Feature importance rankings  
- CSV comparison tables  

All SHAP artifacts are saved locally and ignored by Git.

---

## 📊 Explainability Outputs

Generated:

- SHAP beeswarm plots  
- SHAP bar charts  
- Combined comparison table:

notebooks/shap_outputs/


---

## 📁 Project Structure

Improved-detection-of-fraud-cases/

│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── shap.ipynb
│ └── shap_outputs/ (ignored by git)
│
├── models/ (ignored by git)
│
├── src/
│ ├── data_loader.py
│ ├── feature_engineering.py
│ └── train.py
│
├── .gitignore
├── requirements.txt
└── README.md


---

## ⚙️ How to Run

### 1. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
2. Install dependencies
pip install -r requirements.txt
3. Run SHAP explainability
Open:

notebooks/shap.ipynb
Run all cells.

📈 Models Used
Random Forest Classifier (Fraud)

Random Forest Classifier (Credit Card)

🔐 Git Management
Ignored files:

.pkl

SHAP plots

SHAP CSV outputs

Virtual environment

Handled via .gitignore.

🚀 Next Steps
🔹 Task 5 — Streamlit Dashboard (Upcoming)
Build an interactive dashboard with:

✅ File upload / manual input
✅ Fraud probability prediction
✅ SHAP explanation per transaction
✅ Feature importance visualization
✅ ROC curve display

Planned features:

Sidebar controls

Dataset selector

Real-time prediction

Explainability panel

🔹 Task 6 — Testing
Implement:

Unit tests for preprocessing

Model prediction tests

Input validation

Edge-case testing

Using:

pytest

🎯 Future Improvements
Hyperparameter tuning

Model comparison (XGBoost / LightGBM)

Real-time API deployment

Dockerization

Cloud hosting

👩‍💻 Author
Firehiwet Zerihun


