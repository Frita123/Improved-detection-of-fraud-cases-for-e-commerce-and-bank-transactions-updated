# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_fraud_data(fraud_df):
    """
    Perform basic cleaning on Fraud dataset:
    - Handle missing values
    - Remove duplicates
    - Correct data types
    """
    fraud_df = fraud_df.drop_duplicates()
    
    fraud_df['age'] = fraud_df['age'].fillna(fraud_df['age'].median())
    fraud_df['browser'] = fraud_df['browser'].fillna('Unknown')
    fraud_df['sex'] = fraud_df['sex'].fillna('U')
    
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(str)
    fraud_df['class'] = fraud_df['class'].astype(int)
    
    return fraud_df


def scale_features(X_train, X_test, numerical_features):
    """
    Standard scale numerical features
    """
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    return X_train_scaled, X_test_scaled, scaler


def encode_features(X_train, X_test, categorical_features):
    """
    One-hot encode categorical features
    """
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    X_train_cat = ohe.fit_transform(X_train[categorical_features])
    X_test_cat = ohe.transform(X_test[categorical_features])
    
    return X_train_cat, X_test_cat, ohe
