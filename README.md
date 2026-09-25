# EdgePredict-CWRU

Embedded predictive maintenance system for bearing fault diagnosis, running fully on a Raspberry Pi 4 without any Cloud dependency.

## Overview

This project implements an Edge AI pipeline that classifies bearing faults in real time using the CWRU Bearing Dataset.

The system runs entirely on-device:

- Signal ingestion and overlapping windowing
- Statistical feature extraction (RMS, Crest Factor, Kurtosis, Skewness)
- Machine Learning inference with a tuned Random Forest classifier
- Web supervision dashboard served by Flask

## Architecture

Three concurrent modules run on the Raspberry Pi:

- Module A: Signal ingestion and feature extraction
- Module B: On-device inference engine
- Module C: Flask web server and Chart.js dashboard

## Hardware

- Raspberry Pi 4 Model B
- Cortex-A72, four cores
- 3.7 GiB RAM
- 8 GB USB drive used as boot and storage device (no microSD)

## Software stack

- Python 3.13
- NumPy, SciPy
- Scikit-Learn, Joblib
- Flask
- Chart.js
- systemd

## Dataset

CWRU Bearing Dataset, 12k Drive End subset.

- 48 files (4 unavailable from public mirrors)
- 4 classes: Normal, InnerRace, OuterRace, Ball
- Sampling frequency: 12 kHz
- Window size: 1024 points
- Step: 128 points (87.5 percent overlap)
- 54781 windows generated

Class distribution:

- InnerRace: 15100
- Ball: 15095
- Normal: 13231
- OuterRace: 11356

## Machine Learning pipeline

Six algorithms were compared under a frozen protocol:

| Model            | CV accuracy       | Test accuracy | Inference (ms/sample) | Size (KB) |
| ---------------- | ----------------- | ------------- | --------------------- | --------- |
| RandomForest     | 0.9807 +/- 0.0014 | 0.9792        | 0.0133                | 15456.4   |
| GradientBoosting | 0.9783 +/- 0.0016 | 0.9760        | 0.0113                | 513.4     |
| MLP              | 0.9780 +/- 0.0014 | 0.9759        | 0.0033                | 91.2      |
| DecisionTree     | 0.9721 +/- 0.0012 | 0.9699        | 0.0003                | 171.6     |
| KNN              | 0.9608 +/- 0.0012 | 0.9617        | 0.0111                | 3617.4    |
| SVM_RBF          | 0.9512 +/- 0.0034 | 0.9586        | 1.0810                | 829.0     |

Random Forest was selected for its best accuracy, best CV mean, lowest CV variance, and fast inference.

Hyperparameter tuning with GridSearchCV (36 combinations, 3-fold CV):

- n_estimators: 50
- max_depth: 20
- min_samples_split: 2
- min_samples_leaf: 1

Final tuned model test accuracy: 0.9779.

Feature importance:

- RMS: 0.6512
- Kurtosis: 0.1843
- Skewness: 0.0951
- Crest Factor: 0.0694

## Inference performance

Latency per window: 0.0133 ms per sample in batch inference.

This is far below the 10 ms target from the specification.

## Runtime footprint

Measured with htop at steady state:

- CPU usage: 4 to 9 percent
- Memory: 162 MB for the Python process
- Load average: 0.75 / 0.64 / 0.45 on a 4-core system

## Deployment

The application runs as a systemd service:

    sudo systemctl status edgepredictpi.service

Dashboard access:

    http://<raspberry-pi-ip>:5000

## Documentation

- docs/dataset.md
- docs/features.md
- docs/model.md
- docs/model_comparison.md
- docs/tuning.md
- docs/final_evaluation.md
- docs/ia_summary.md
- docs/performance.md
- docs/api.md
- docs/deployment.md
- docs/test_plan.md

## Report

The technical report is available at report/report.md.

## Notes

- The GridSearchCV tuning was performed on the development machine because the Raspberry Pi 4 without active cooling reached thermal throttling at 84.7 degrees Celsius under sustained multi-core load.
- The tuned model was then transferred to the Pi for deployment.
- This reflects a standard MLOps workflow: train and tune off-device, deploy the final artifact on the target.

## Author

Horacia Azonhoumon
GSEA - Embedded Electronics and Automation
ENSA Tanger
