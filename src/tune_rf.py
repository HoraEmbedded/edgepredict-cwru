"""
Hyperparameter tuning for the Random Forest classifier using GridSearchCV.
Protocol is frozen:

- 80/20 stratified split, random_state=42
- 3-fold stratified cross-validation on the training set
- Test set opened only once, at the end
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "features.csv")
REPORT_DIR = os.path.join(BASE_DIR, "report")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RESULTS_FILE = os.path.join(REPORT_DIR, "gridsearch_results.csv")
BEST_PARAMS_FILE = os.path.join(REPORT_DIR, "best_params.json")
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "cwru_model_tuned.pkl")
META_FILE = os.path.join(MODEL_DIR, "cwru_model_tuned_meta.json")

FEATURES = ["rms", "crest_factor", "kurtosis", "skewness"]
RANDOM_STATE = 42


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES].values
    y = df["label"].values
    print(f"Total samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    n_combinations = 1
    for v in param_grid.values():
        n_combinations *= len(v)
    print(f"\nGrid size: {n_combinations} combinations")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    print("\nStarting GridSearchCV (this may take a while)...")
    t0 = time.perf_counter()
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)
    t1 = time.perf_counter()
    print(f"\nGridSearch completed in {t1 - t0:.1f} s")

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results = cv_results.sort_values("rank_test_score")
    cv_results.to_csv(RESULTS_FILE, index=False)
    print(f"GridSearch results saved to {RESULTS_FILE}")

    with open(BEST_PARAMS_FILE, "w") as fp:
        json.dump({
            "best_params": grid.best_params_,
            "best_cv_accuracy": float(grid.best_score_),
            "n_combinations": n_combinations,
            "total_time_s": round(t1 - t0, 1),
        }, fp, indent=2)
    print(f"Best params saved to {BEST_PARAMS_FILE}")

    print("\nEvaluating best model on test set...")
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_acc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, BEST_MODEL_FILE)
    print(f"\nTuned model saved to {BEST_MODEL_FILE}")

    meta = {
        "features": FEATURES,
        "classes": list(best_model.classes_),
        "best_params": grid.best_params_,
        "best_cv_accuracy": float(grid.best_score_),
        "test_accuracy": float(test_acc),
    }
    with open(META_FILE, "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"Metadata saved to {META_FILE}")


if __name__ == "__main__":
    main()
