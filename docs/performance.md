# Inference Performance

Measured on Raspberry Pi 4 Model B.

## Setup

- Model: Random Forest, 100 trees
- Features: rms, crest_factor, kurtosis, skewness
- Window size: 1024 points
- Input: CWRU 105.mat (Inner Race fault)

## Latency per window

- Mean latency:   43.730 ms
- Min latency:    37.898 ms
- Max latency:    78.577 ms
- P95 latency:    52.821 ms

## Target

The project targets a latency below 10 ms per window.

The measured latency is well below this target.
