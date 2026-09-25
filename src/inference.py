"""
On-device inference engine for EdgePredict-CWRU.
Loads the trained model, simulates a real-time stream of windows,
and measures inference latency per window.
"""

import os
import time
import json
import joblib
import numpy as np

from ingestion import load_signal, window_signal
from features import extract_features


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_FILE = os.path.join(BASE_DIR, "models", "cwru_model.pkl")
META_FILE = os.path.join(BASE_DIR, "models", "cwru_model_meta.json")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def load_model():
    model = joblib.load(MODEL_FILE)
    with open(META_FILE, "r") as fp:
        meta = json.load(fp)
    return model, meta


def predict_window(model, meta, window):
    """
    Extract features from a single window and predict the class.
    Returns (label, latency_ms).
    """
    t0 = time.perf_counter()

    f = extract_features(window)
    x = np.array([[f["rms"], f["crest_factor"], f["kurtosis"], f["skewness"]]])
    label = model.predict(x)[0]

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    return label, latency_ms


def main():
    print("Loading model...")
    model, meta = load_model()
    print(f"Model classes: {meta['classes']}")
    print(f"Training accuracy: {meta['accuracy']}")

    sample_file = os.path.join(DATA_DIR, "105.mat")
    print(f"\nLoading signal from {sample_file}")
    signal = load_signal(sample_file)
    windows = window_signal(signal)
    print(f"Signal length: {len(signal)}, windows: {windows.shape[0]}")

    print("\nRunning inference on all windows...")
    latencies = []
    predictions = []

    for w in windows:
        label, latency = predict_window(model, meta, w)
        latencies.append(latency)
        predictions.append(label)

    latencies = np.array(latencies)

    print("\nInference summary")
    print(f"Total windows processed: {len(latencies)}")
    print(f"Mean latency:   {latencies.mean():.3f} ms")
    print(f"Min latency:    {latencies.min():.3f} ms")
    print(f"Max latency:    {latencies.max():.3f} ms")
    print(f"P95 latency:    {np.percentile(latencies, 95):.3f} ms")

    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPredicted class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")


if __name__ == "__main__":
    main()
