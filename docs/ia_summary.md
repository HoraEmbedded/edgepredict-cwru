# AI Pipeline Summary

## Step 1: Feature extraction

Four time-domain features per window: RMS, Crest Factor, Kurtosis, Skewness.

## Step 2: Exploratory analysis

Class distribution, per-class statistics, feature boxplots, correlation matrix, pairplot.

## Step 3: Model comparison

Six algorithms compared under a frozen protocol.

## Step 4: Hyperparameter tuning

GridSearchCV on the selected Random Forest.

## Step 5: Final evaluation

Confusion matrix, classification report, feature importance, ROC curves.

## Key figures

- report/figures/class_distribution.png
- report/figures/feature_boxplots.png
- report/figures/feature_correlation.png
- report/figures/feature_pairplot.png
- report/figures/confusion_matrix.png
- report/figures/feature_importance.png
- report/figures/roc_curves.png

## Key results

- 6 models compared
- 36 hyperparameter combinations tested
- 10957 test windows
- Final test accuracy: 0.9779
- Feature importance dominated by RMS (0.65) and Kurtosis (0.18)
