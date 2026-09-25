"""
Background thread running inference in a loop and updating shared state.
Uses the tuned Random Forest model.
"""

import os
import time
import json
import joblib
import threading
import numpy as np

from ingestion import load_signal, window_signal
from features import extract_features


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_FILE = os.path.join(BASE_DIR, "models", "cwru_model_tuned.pkl")
META_FILE = os.path.join(BASE_DIR, "models", "cwru_model_tuned_meta.json")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

DEFAULT_FILE = "105.mat"
LOOP_DELAY_S = 0.5


def load_model():
    model = joblib.load(MODEL_FILE)
    with open(META_FILE, "r") as fp:
        meta = json.load(fp)
    return model, meta


def inference_loop(state, stop_event, source_file=DEFAULT_FILE):
    model, meta = load_model()
    path = os.path.join(DATA_DIR, source_file)
    signal = load_signal(path)
    windows = window_signal(signal)
    n = windows.shape[0]

    print(f"[inference] Loaded {source_file}, windows: {n}")
    print(f"[inference] Model classes: {meta['classes']}")
    print(f"[inference] Test accuracy: {meta.get('test_accuracy', 'n/a')}")

    idx = 0
    while not stop_event.is_set():
        window = windows[idx]

        t0 = time.perf_counter()
        f = extract_features(window)
        x = np.array([[f["rms"], f["crest_factor"], f["kurtosis"], f["skewness"]]])
        proba = model.predict_proba(x)[0]
        label = model.classes_[int(np.argmax(proba))]
        confidence = float(np.max(proba))
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        state.update(
            label=str(label),
            features=f,
            confidence=confidence,
            latency_ms=latency_ms,
            window_index=idx,
            source_file=source_file,
        )

        idx = (idx + 1) % n
        time.sleep(LOOP_DELAY_S)


def start_inference_thread(state, source_file=DEFAULT_FILE):
    stop_event = threading.Event()
    thread = threading.Thread(
        target=inference_loop,
        args=(state, stop_event, source_file),
        daemon=True,
    )
    thread.start()
    return thread, stop_event
