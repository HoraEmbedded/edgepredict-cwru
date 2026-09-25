"""
CWRU dataset ingestion module.
Loads .mat files and splits continuous signals into fixed-size windows.
"""

import os
import numpy as np
from scipy.io import loadmat


WINDOW_SIZE = 1024
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_signal(filepath: str) -> np.ndarray:

    mat = loadmat(filepath)
    key = None
    for k in mat.keys():
        if k.endswith("DE_time"):
            key = k
            break
    if key is None:
        raise ValueError(f"No DE_time key found in {filepath}")
    signal = mat[key].flatten()
    return signal


def window_signal(signal: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """
    Split a continuous signal into non-overlapping windows.
    """
    num_windows = len(signal) // window_size
    trimmed = signal[: num_windows * window_size]
    windows = trimmed.reshape(num_windows, window_size)
    return windows


def main():
    print("CWRU ingestion module")
    files = sorted(os.listdir(DATA_DIR))
    for fname in files:
        if not fname.endswith(".mat"):
            continue
        path = os.path.join(DATA_DIR, fname)
        signal = load_signal(path)
        windows = window_signal(signal)
        print(f"{fname}: signal length = {len(signal)}, windows = {windows.shape}")


if __name__ == "__main__":
    main()
