# EdgePredict API

The Flask server exposes the following routes.

## GET /

Returns the dashboard HTML page.

## GET /api/state

Returns the latest prediction as JSON.

Example response:

{
  "label": "InnerRace",
  "features": {
    "rms": 0.123,
    "crest_factor": 4.56,
    "kurtosis": 3.21,
    "skewness": 0.12
  },
  "latency_ms": 0.87,
  "window_index": 42,
  "source_file": "105.mat"
}

## GET /api/history

Returns the last 100 predictions as a JSON array.


## GET /api/model

Returns the model metadata: classes, test accuracy, best parameters.

Example response:

{
  "classes": ["Ball", "InnerRace", "Normal", "OuterRace"],
  "best_params": {
    "max_depth": 20,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 50
  },
  "test_accuracy": 0.9779
}


## GET /api/files

Returns the list of available .mat files and the current source file.

## POST /api/select

Switches the inference thread to another .mat file.

Request body:

{
  "file": "97.mat"
}

Response:

{
  "ok": true,
  "file": "97.mat"
}

## Network

The server listens on port 5000 and is reachable on the local network.


