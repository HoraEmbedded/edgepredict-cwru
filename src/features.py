"""
Statistical feature extraction for vibration signals.
Computes RMS, Crest Factor, Kurtosis and Skewness for each window.
"""

import numpy as np


def rms(window: np.ndarray) -> float:
    return float(np.sqrt(np.mean(window ** 2)))


def crest_factor(window: np.ndarray) -> float:
    r = rms(window)
    if r == 0:
        return 0.0
    return float(np.max(np.abs(window)) / r)


def kurtosis(window: np.ndarray) -> float:
    mean = np.mean(window)
    centered = window - mean
    m2 = np.mean(centered ** 2)
    m4 = np.mean(centered ** 4)
    if m2 == 0:
        return 0.0
    return float(m4 / (m2 ** 2))


def skewness(window: np.ndarray) -> float:
    mean = np.mean(window)
    centered = window - mean
    m2 = np.mean(centered ** 2)
    m3 = np.mean(centered ** 3)
    if m2 == 0:
        return 0.0
    return float(m3 / (m2 ** 1.5))


def extract_features(window: np.ndarray) -> dict:
    """
    Return a dict of features for a single window.
    """
    return {
        "rms": rms(window),
        "crest_factor": crest_factor(window),
        "kurtosis": kurtosis(window),
        "skewness": skewness(window),
    }


def extract_features_batch(windows: np.ndarray) -> np.ndarray:
    """
    Apply extract_features to each row of a 2D array.
    """
    features = np.zeros((windows.shape[0], 4), dtype=np.float64)
    for i, w in enumerate(windows):
        f = extract_features(w)
        features[i, 0] = f["rms"]
        features[i, 1] = f["crest_factor"]
        features[i, 2] = f["kurtosis"]
        features[i, 3] = f["skewness"]
    return features
