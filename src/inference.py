"""On-device inference engine for EdgePredict-CWRU.
Loads the trained model, simulates a real-time stream of windows,
and measures per-window inference latency.
"""

import os
import time
import json
import joblib
import numpy as np

from ingestion import load_signal, window_signal
from features import extract_features_vectorized


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_FILE = os.path.join(BASE_DIR, "models", "cwru_model.pkl")
META_FILE = os.path.join(BASE_DIR, "models", "cwru_model_meta.json")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def load_model():
    model = joblib.load(MODEL_FILE)
    with open(META_FILE, "r") as fp:
        meta = json.load(fp)
    return model, meta


def run_inference(model, windows):
    """
    Vectorized inference on a batch of windows.
    Returns (predictions, features, total_time_s).
    """
    t0 = time.perf_counter()

    features = extract_features_vectorized(windows)
    predictions = model.predict(features)

    t1 = time.perf_counter()
    return predictions, features, (t1 - t0)


def main():
    print("Loading model...")
    model, meta = load_model()
    print(f"Model classes: {meta['classes']}")
    print(f"Training accuracy: {meta['accuracy']}")

    sample_file = os.path.join(DATA_DIR, "105.mat")
    print(f"\nLoading signal from {sample_file}")
    signal = load_signal(sample_file)
    windows = window_signal(signal)
    n = windows.shape[0]
    print(f"Signal length: {len(signal)}, windows: {n}")

    print("\nRunning vectorized inference...")
    predictions, features, total_s = run_inference(model, windows)

    per_window_ms = (total_s * 1000.0) / n
    print("\nInference summary")
    print(f"Total windows processed: {n}")
    print(f"Total time:              {total_s * 1000.0:.3f} ms")
    print(f"Per-window latency:      {per_window_ms:.3f} ms")

    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPredicted class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")

    if per_window_ms < 10.0:
        print("\nTarget met: per-window latency below 10 ms.")
    else:
        print("\nWARNING: per-window latency above 10 ms.")


if __name__ == "__main__":
    main()
