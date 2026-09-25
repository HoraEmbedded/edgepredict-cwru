"""
Thread-safe shared state for the EdgePredict system.
Stores the latest prediction, a rolling history, and the current source file.
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
        self._source_file = "105.mat"
        self._reload_requested = False

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

    def get_source_file(self):
        with self._lock:
            return self._source_file

    def set_source_file(self, filename):
        with self._lock:
            self._source_file = filename
            self._reload_requested = True

    def consume_reload_request(self):
        with self._lock:
            requested = self._reload_requested
            self._reload_requested = False
            return requested
