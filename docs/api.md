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

## Network

The server listens on port 5000 and is reachable on the local network.
