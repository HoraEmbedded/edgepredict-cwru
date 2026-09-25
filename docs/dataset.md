# CWRU Dataset

The CWRU Bearing Dataset is a reference dataset for bearing fault diagnosis.

## Classes used in this project

- Normal
- Inner Race fault
- Outer Race fault
- Ball fault

## Sampling

- Sampling frequency: 12 kHz
- Window size: 1024 points
- Step: 128 points
- Overlap: 87.5 percent

## Files

48 files are used, from the 12k Drive End subset.

File distribution per class:

- Normal: 4 files
- InnerRace: 16 files
- Ball: 16 files
- OuterRace: 12 files

The 4 files 3009.mat to 3012.mat (Outer Race 0.028 inch severity) were not available from the CWRU website and public mirrors at the time of download.

## Feature dataset

Generated file: data/processed/features.csv

Total windows: 54781

Class distribution:

- InnerRace: 15100 windows
- Ball: 15095 windows
- Normal: 13231 windows
- OuterRace: 11356 windows

## Descriptive statistics

RMS: mean 0.397, std 0.519, min 0.056, max 2.866
Crest Factor: mean 4.458, std 1.698, min 2.425, max 11.565
Kurtosis: mean 6.491, std 6.161, min 2.301, max 52.006
Skewness: mean 0.014, std 0.168, min -0.838, max 1.260
