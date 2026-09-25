# Statistical Features

Four time-domain features are computed for each window of 1024 points.

## RMS

Root Mean Square. Represents the energy of the signal.

## Crest Factor

Ratio between the peak value and the RMS. Sensitive to spikes.

## Kurtosis

Fourth statistical moment. Sensitive to impulsive faults.

## Skewness

Third statistical moment. Measures the asymmetry of the distribution.

## Output

A CSV file is generated at data/processed/features.csv with the following columns:

- rms
- crest_factor
- kurtosis
- skewness
- label
