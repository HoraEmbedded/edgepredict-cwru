# EdgePredict-CWRU

Embedded predictive maintenance system for bearing fault diagnosis, running fully on a Raspberry Pi 4 without any Cloud dependency.

## Overview

This project implements an Edge AI pipeline that classifies bearing faults in real time using the CWRU Bearing Dataset.

The system runs entirely on-device:

- Signal ingestion and windowing
- Statistical feature extraction
- Machine Learning inference with a Random Forest classifier
- Web supervision dashboard served by Flask

## Architecture

Three concurrent modules run on the Raspberry Pi:

- Module A: Signal ingestion and feature extraction
- Module B: On-device inference engine
- Module C: Flask web server and dashboard

## Hardware

- Raspberry Pi 4 Model B
- 8 GB USB drive used as boot and storage device

## Software stack

- Python 3
- NumPy, SciPy
- Scikit-Learn, Joblib
- Flask
- Chart.js

## Dataset

CWRU Bearing Dataset, 12 files, 4 classes:

- Normal
- Inner Race fault
- Outer Race fault
- Ball fault

Sampling frequency: 12 kHz
Window size: 1024 points

## Model

Random Forest Classifier, 100 trees.

Features:

- RMS
- Crest Factor
- Kurtosis
- Skewness

Accuracy on test set: 100 percent on the selected subset.

## Performance

Inference latency per window: below 10 ms using vectorized batch inference.

## Deployment

The application runs as a systemd service:

    sudo systemctl status edgepredictpi.service

Dashboard access:

    http://<raspberry-pi-ip>:5000

## Documentation

See the docs folder for:

- dataset description
- feature definitions
- model training results
- API reference
- deployment guide
- test plan

##Demonstration

## Author

HoraEmbedded
GSEA - Embedded Electronics and Automation
