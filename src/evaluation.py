import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


def load_model(name="fraud_model.pkl"):
    """
    Load trained model from disk.
    """
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    model = joblib.load(path)
    print(f"Loaded model from: {path}")
    return model


def evaluate_model(model, df: pd.DataFrame, target="class"):
    """
    Evaluate model on dataframe.
    """

    X = df.drop(columns=[target])
    y = df[target]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:\n")
    print(cm)

    roc_auc = roc_auc_score(y, y_prob)
    print("\nROC AUC:", roc_auc)

    plot_confusion_matrix(cm)
    plot_roc_curve(y, y_prob)

    return {
        "confusion_matrix": cm,
        "roc_auc": roc_auc
    }


def plot_confusion_matrix(cm):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()


if __name__ == "__main__":
    print("Import this module into notebooks or scripts.")
