"""
Compare six supervised learning algorithms on the CWRU feature dataset.
Protocol is frozen before any training:

- 80/20 stratified split, random_state=42
- 5-fold stratified cross-validation on the training set
- test set used only once, at the end
- default hyperparameters for all models, no tuning in this step
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "features.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "report")
RESULTS_FILE = os.path.join(RESULTS_DIR, "model_comparison.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models", "comparison")

FEATURES = ["rms", "crest_factor", "kurtosis", "skewness"]
RANDOM_STATE = 42


def build_models():
    """
    Return a dict of name -> model instance.
    All models use default parameters for fair comparison.
    """
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM_RBF": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
    }


def time_inference(model, X, n_repeats=3):
    """
    Measure mean inference time per sample over n_repeats runs.
    Returns milliseconds per sample.
    """
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict(X)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / len(X))
    return float(np.mean(times))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES].values
    y = df["label"].values
    print(f"Total samples: {len(X)}")
    print(f"Classes: {np.unique(y).tolist()}")

    print("\nSplitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("\nScaling features (required for SVM, KNN, MLP)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = build_models()
    results = []

    for name, model in models.items():
        print(f"\n=== {name} ===")

        needs_scaling = name in ("SVM_RBF", "KNN", "MLP")
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test

        print("Cross-validation (5-fold)...")
        t0 = time.perf_counter()
        cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        t1 = time.perf_counter()
        cv_time_s = t1 - t0

        print(f"CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"CV time: {cv_time_s:.2f} s")

        print("Training on full training set...")
        t0 = time.perf_counter()
        model.fit(Xtr, y_train)
        t1 = time.perf_counter()
        train_time_s = t1 - t0
        print(f"Training time: {train_time_s:.2f} s")

        print("Evaluating on test set...")
        y_pred = model.predict(Xte)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {test_acc:.4f}")

        print("Measuring inference time...")
        infer_ms = time_inference(model, Xte)
        print(f"Inference: {infer_ms:.4f} ms per sample")

        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        size_kb = os.path.getsize(model_path) / 1024.0
        print(f"Model size: {size_kb:.1f} KB")

        results.append({
            "model": name,
            "cv_accuracy_mean": round(cv_scores.mean(), 4),
            "cv_accuracy_std": round(cv_scores.std(), 4),
            "cv_time_s": round(cv_time_s, 2),
            "train_time_s": round(train_time_s, 2),
            "test_accuracy": round(test_acc, 4),
            "inference_ms_per_sample": round(infer_ms, 4),
            "model_size_kb": round(size_kb, 1),
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("test_accuracy", ascending=False)
    df_results.to_csv(RESULTS_FILE, index=False)

    print("\n=== Final comparison ===")
    print(df_results.to_string(index=False))
    print(f"\nResults saved to {RESULTS_FILE}")

    with open(os.path.join(RESULTS_DIR, "comparison_scaler.pkl"), "wb") as fp:
        joblib.dump(scaler, fp)
    print("Scaler saved for reuse.")


if __name__ == "__main__":
    main()
