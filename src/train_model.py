"""
Train a Random Forest classifier on the CWRU feature dataset.
Evaluates the model and saves it to models/cwru_model.pkl.
"""

import os
import json
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "cwru_model.pkl")
META_FILE = os.path.join(MODEL_DIR, "cwru_model_meta.json")

FEATURES = ["rms", "crest_factor", "kurtosis", "skewness"]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"Total rows: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    X = df[FEATURES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    if acc < 0.95:
        print("WARNING: accuracy below 95 percent target.")

    print(f"\nSaving model to {MODEL_FILE}")
    joblib.dump(model, MODEL_FILE)

    meta = {
        "features": FEATURES,
        "classes": list(model.classes_),
        "accuracy": float(acc),
        "n_estimators": 100,
    }
    with open(META_FILE, "w") as fp:
        json.dump(meta, fp, indent=2)

    print(f"Metadata saved to {META_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
