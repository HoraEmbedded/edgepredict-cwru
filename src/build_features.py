"""
Build the feature dataset from raw CWRU files.
"""

import os
import csv
import numpy as np

from ingestion import load_signal, window_signal
from features import extract_features_batch


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "features.csv")


FILE_TO_CLASS = {
    "97.mat": "Normal",
    "98.mat": "Normal",
    "99.mat": "Normal",
    "105.mat": "InnerRace",
    "106.mat": "InnerRace",
    "107.mat": "InnerRace",
    "118.mat": "Ball",
    "119.mat": "Ball",
    "120.mat": "Ball",
    "130.mat": "OuterRace",
    "131.mat": "OuterRace",
    "132.mat": "OuterRace",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []

    for fname, label in FILE_TO_CLASS.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"Missing file: {fname}, skipping.")
            continue

        signal = load_signal(path)
        windows = window_signal(signal)
        features = extract_features_batch(windows)

        for f in features:
            rows.append([f[0], f[1], f[2], f[3], label])

        print(f"{fname} -> {features.shape[0]} windows, class = {label}")

    with open(OUTPUT_FILE, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["rms", "crest_factor", "kurtosis", "skewness", "label"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
