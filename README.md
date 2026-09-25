# EdgePredict-CWRU

Embedded predictive maintenance system running fully on a Raspberry Pi 4 (Edge Computing / TinyML).

The system classifies bearing faults in real time using the CWRU Bearing Dataset.

## Modules

- Module A: Signal ingestion and statistical feature extraction
- Module B: On-device Machine Learning inference
- Module C: Web supervision dashboard (Flask + Chart.js)

## Target

- Hardware: Raspberry Pi 4 Model B
- Language: Python 3
- Libraries: NumPy, SciPy, Scikit-Learn, Joblib, Flask, Chart.js
- Inference latency target: below 10 ms per signal block
