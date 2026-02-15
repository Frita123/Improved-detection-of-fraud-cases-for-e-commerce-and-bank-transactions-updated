# src/features.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def create_time_features(fraud_df):
    """
    Add time-based features for fraud dataset
    """
    fraud_df['time_since_signup'] = (
        fraud_df['purchase_time'] - fraud_df['signup_time']
    ).dt.total_seconds() / 3600  # in hours
    
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    
    return fraud_df


def handle_class_imbalance(X_train, y_train, method="undersample", random_state=42):
    """
    Handle class imbalance using undersampling (default) or oversampling
    Returns resampled X_train and y_train
    """
    train_data = np.hstack([X_train, y_train.reshape(-1, 1)])
    
    # Split majority and minority
    majority = train_data[train_data[:, -1] == 0]
    minority = train_data[train_data[:, -1] == 1]
    
    if method == "undersample":
        majority_resampled = resample(
            majority,
            replace=False,
            n_samples=len(minority),
            random_state=random_state
        )
        resampled = np.vstack([majority_resampled, minority])
    
    elif method == "oversample":
        minority_resampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=random_state
        )
        resampled = np.vstack([majority, minority_resampled])
    
    else:
        raise ValueError("Method must be 'undersample' or 'oversample'")
    
    np.random.shuffle(resampled)
    
    X_resampled = resampled[:, :-1]
    y_resampled = resampled[:, -1]
    
    return X_resampled, y_resampled
