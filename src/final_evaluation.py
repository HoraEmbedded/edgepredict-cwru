"""
Final evaluation of the tuned Random Forest model.
Produces figures and metrics for the technical report.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "cwru_model_tuned.pkl")
META_FILE = os.path.join(BASE_DIR, "models", "cwru_model_tuned_meta.json")
FIG_DIR = os.path.join(BASE_DIR, "report", "figures")
REPORT_DIR = os.path.join(BASE_DIR, "report")

FEATURES = ["rms", "crest_factor", "kurtosis", "skewness"]
RANDOM_STATE = 42


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("Confusion matrix - tuned Random Forest")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(model):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(7, 5))
    sns.barplot(
        x=[FEATURES[i] for i in indices],
        y=[importances[i] for i in indices],
        hue=[FEATURES[i] for i in indices],
        palette="viridis",
        legend=False,
    )
    plt.title("Feature importance - tuned Random Forest")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "feature_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")

    print("\nFeature importances:")
    for i in indices:
        print(f"  {FEATURES[i]}: {importances[i]:.4f}")

    return importances


def plot_roc_curves(model, X_test, y_test, classes):
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title("ROC curves - one vs rest")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "roc_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading dataset and tuned model...")
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES].values
    y = df["label"].values

    model = joblib.load(MODEL_FILE)
    with open(META_FILE, "r") as fp:
        meta = json.load(fp)

    classes = list(model.classes_)
    print(f"Classes: {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Predicting on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(REPORT_DIR, "final_classification_report.csv")
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")
    print(classification_report(y_test, y_pred))

    print("\nGenerating figures...")
    plot_confusion_matrix(y_test, y_pred, classes)
    importances = plot_feature_importance(model)
    plot_roc_curves(model, X_test, y_test, classes)

    summary = {
        "test_accuracy": float(acc),
        "feature_importances": {
            FEATURES[i]: float(importances[i]) for i in range(len(FEATURES))
        },
        "n_test_samples": int(len(y_test)),
        "classes": classes,
    }
    summary_path = os.path.join(REPORT_DIR, "final_evaluation_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
