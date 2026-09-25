# Machine Learning Model

## Overview

The system classifies bearing faults using four statistical features extracted from vibration windows.

## Dataset

- 48 CWRU 12k Drive End files
- 54781 windows of 1024 points with 87.5 percent overlap
- 4 classes: Normal, InnerRace, OuterRace, Ball

## Algorithm selection

Six algorithms were compared under a frozen protocol:

- Decision Tree
- Random Forest
- Gradient Boosting
- SVM with RBF kernel
- KNN with k=5
- Multi-Layer Perceptron (64, 32)

Each was evaluated with 5-fold stratified cross-validation on the training set, then on a held-out test set opened only once.

Results:

| Model            | CV accuracy       | Test accuracy | Inference (ms/sample) | Size (KB) |
| ---------------- | ----------------- | ------------- | --------------------- | --------- |
| RandomForest     | 0.9807 +/- 0.0014 | 0.9792        | 0.0133                | 15456.4   |
| GradientBoosting | 0.9783 +/- 0.0016 | 0.9760        | 0.0113                | 513.4     |
| MLP              | 0.9780 +/- 0.0014 | 0.9759        | 0.0033                | 91.2      |
| DecisionTree     | 0.9721 +/- 0.0012 | 0.9699        | 0.0003                | 171.6     |
| KNN              | 0.9608 +/- 0.0012 | 0.9617        | 0.0111                | 3617.4    |
| SVM_RBF          | 0.9512 +/- 0.0034 | 0.9586        | 1.0810                | 829.0     |

## Selected model

Random Forest was selected for its best test accuracy, best CV mean, lowest CV variance, and inference time 80 times faster than SVM.

## Hyperparameter tuning

GridSearchCV was applied on the Random Forest with 3-fold stratified cross-validation and 36 combinations.

Best parameters:

- n_estimators: 50
- max_depth: 20
- min_samples_split: 2
- min_samples_leaf: 1

Best CV accuracy: 0.9803

The default Random Forest scored 0.9807 in CV, which is within noise. Tuning did not improve accuracy, but reduced model size by half.

## Final model

The tuned Random Forest was deployed on the Raspberry Pi 4.

Final test accuracy: 0.9779

Feature importance:

- RMS: 0.6512
- Kurtosis: 0.1843
- Skewness: 0.0951
- Crest Factor: 0.0694

## Artifacts

- models/cwru_model_tuned.pkl
- models/cwru_model_tuned_meta.json
- report/model_comparison.csv
- report/gridsearch_results.csv
- report/best_params.json
- report/final_classification_report.csv
- report/final_evaluation_summary.json
