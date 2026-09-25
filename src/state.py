"""
Thread-safe shared state for the EdgePredict system.
Stores the latest prediction and a rolling history.
"""

import threading
from collections import deque


class SharedState:
    def __init__(self, history_size=100):
        self._lock = threading.Lock()
        self._latest = {
            "label": None,
            "features": None,
            "confidence": None,
            "latency_ms": None,
            "window_index": None,
            "source_file": None,
        }
        self._history = deque(maxlen=history_size)

    def update(self, label, features, confidence, latency_ms, window_index, source_file):
        with self._lock:
            self._latest = {
                "label": label,
                "features": features,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "window_index": window_index,
                "source_file": source_file,
            }
            self._history.append(self._latest.copy())

    def get_latest(self):
        with self._lock:
            return dict(self._latest)

    def get_history(self):
        with self._lock:
            return list(self._history)
