"""
Build the feature dataset from raw CWRU files.
Produces a CSV file with one row per window and a class label.
Only files present in data/raw are processed.
"""

import os
import csv
import numpy as np

from ingestion import load_signal, window_signal
from features import extract_features_batch


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "features.csv")


# Complete CWRU 12k Drive End dataset mapping
FILE_TO_CLASS = {}

# Normal Baseline: 97-100
for i in range(97, 101):
    FILE_TO_CLASS[f"{i}.mat"] = "Normal"

# Inner Race (IR)
for i in range(105, 109):
    FILE_TO_CLASS[f"{i}.mat"] = "InnerRace"
for i in range(169, 173):
    FILE_TO_CLASS[f"{i}.mat"] = "InnerRace"
for i in range(209, 213):
    FILE_TO_CLASS[f"{i}.mat"] = "InnerRace"
for i in range(3001, 3005):
    FILE_TO_CLASS[f"{i}.mat"] = "InnerRace"

# Ball (B)
for i in range(118, 122):
    FILE_TO_CLASS[f"{i}.mat"] = "Ball"
for i in range(185, 189):
    FILE_TO_CLASS[f"{i}.mat"] = "Ball"
for i in range(222, 226):
    FILE_TO_CLASS[f"{i}.mat"] = "Ball"
for i in range(3005, 3009):
    FILE_TO_CLASS[f"{i}.mat"] = "Ball"

# Outer Race (OR@6)
for i in range(130, 134):
    FILE_TO_CLASS[f"{i}.mat"] = "OuterRace"
for i in range(197, 201):
    FILE_TO_CLASS[f"{i}.mat"] = "OuterRace"
for i in range(234, 238):
    FILE_TO_CLASS[f"{i}.mat"] = "OuterRace"
for i in range(3009, 3013):
    FILE_TO_CLASS[f"{i}.mat"] = "OuterRace"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    total_expected = len(FILE_TO_CLASS)
    processed = 0
    skipped = 0

    for fname, label in FILE_TO_CLASS.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            skipped += 1
            continue

        try:
            signal = load_signal(path)
            windows = window_signal(signal)
            features = extract_features_batch(windows)

            for f in features:
                rows.append([f[0], f[1], f[2], f[3], label])

            processed += 1
            print(f"[{processed}] {fname} -> {features.shape[0]} windows, class = {label}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    with open(OUTPUT_FILE, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["rms", "crest_factor", "kurtosis", "skewness", "label"])
        writer.writerows(rows)

    print(f"\nProcessed files: {processed} out of {total_expected}")
    print(f"Skipped missing files: {skipped}")
    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
