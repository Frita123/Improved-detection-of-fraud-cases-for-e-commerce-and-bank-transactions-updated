
# 💳 Improved Detection of Fraud Cases for E-Commerce and Bank Transactions

This project implements a complete **machine learning pipeline** for detecting fraudulent transactions using two datasets:

- E-commerce Fraud Dataset  
- Credit Card Transactions Dataset  

It covers:

✅ Data preprocessing & feature engineering  
✅ Exploratory Data Analysis (EDA)  
✅ Model training (Logistic Regression & Random Forest)  
✅ Model explainability (SHAP)  
✅ Interactive Streamlit dashboard  
✅ Automated testing  

---

## 📁 Project Structure

```

Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions/

│
├── data/
│   └── processed/
│       ├── fraud_cleaned.csv
│       └── creditcard_cleaned.csv
│
├── models/
│   ├── fraud_rf.pkl
│   ├── fraud_scaler.pkl
│   ├── fraud_features.pkl
│   ├── creditcard_rf.pkl
│   ├── creditcard_scaler.pkl
│   └── creditcard_features.pkl
│
├── shap_outputs/
│   ├── fraud_beeswarm.png
│   └── credit_beeswarm.png
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── Modeling.ipynb
│   └── SHAP_Explainability.ipynb
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│   └── test_features.py
│
├── requirements.txt
└── README.md

```

---

## 🎯 Project Objectives

- Detect fraudulent transactions accurately
- Engineer meaningful behavioral features
- Compare Logistic Regression vs Random Forest
- Explain predictions using SHAP
- Build an interactive dashboard for analysis
- Validate pipelines with automated tests

---

## 📊 Exploratory Data Analysis (EDA)

Performed on both datasets:

- Feature distributions
- Correlation heatmaps
- Class imbalance visualization
- Transaction behavior patterns

EDA is implemented both in notebooks and the Streamlit dashboard.

---

## 🛠 Feature Engineering

Key engineered features include:

### Fraud Dataset

- `time_since_signup`
- `hour_of_day`
- `day_of_week`
- `user_transaction_count`
- `time_diff_hours`
- `avg_time_between_tx`
- IP-based country mapping

Categorical variables were one-hot encoded.

### Credit Card Dataset

- Standard PCA features (V1–V28)
- Transaction amount normalization

---

## 🤖 Modeling

Two models were trained:

### Logistic Regression  
Baseline classifier

### Random Forest  
Final production model

Metrics evaluated:

- F1 Score  
- Precision / Recall  
- AUC-PR  
- Confusion Matrix  

Final results:

| Dataset | F1 | AUC-PR |
|--------|----|--------|
| Fraud | 0.686 | 0.626 |
| Credit Card | 0.834 | 0.807 |

Models, scalers, and feature lists are saved in `/models`.

---

## 🧠 Model Explainability (SHAP)

SHAP was used to explain Random Forest predictions:

- Global feature importance
- Beeswarm plots

Saved outputs:

```

shap_outputs/
├── fraud_beeswarm.png
└── credit_beeswarm.png

````

These are displayed directly inside the Streamlit app.

---

## 📈 Streamlit Dashboard

Features:

✅ Dataset selector  
✅ Interactive EDA  
✅ Model metrics  
✅ SHAP visualizations  
✅ Feature importance bar charts  
✅ Single transaction prediction  

### Run the app:

```bash
cd streamlit_app
streamlit run app.py
````

---

## 🧪 Automated Testing

Tests ensure:

* Datasets load correctly
* Target column exists
* Models load properly
* Predictions run without errors
* Feature alignment matches training

Run tests:

```bash
cd tests
python test_features.py
```

---

## ⚙ Installation

Create virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Key Technologies

* Python
* Pandas / NumPy
* Scikit-Learn
* SHAP
* Streamlit
* Matplotlib / Seaborn
* Joblib

---

## 👩‍💻 Author

**Firehiwet Zerihun**

Data Analyst & Machine Learning Practitioner

---

## ✅ Status

✔ Feature engineering complete
✔ Modeling complete
✔ Explainability complete
✔ Dashboard complete
✔ Testing complete

---

## 📌 Notes

* Feature lists are saved to prevent prediction mismatch
* SHAP plots are precomputed for performance
* Random Forest chosen as final model
* Project follows production-style ML workflow

---


