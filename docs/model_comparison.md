# Model Comparison

Six supervised learning algorithms were compared under a frozen protocol.

## Protocol

- 80/20 stratified split, random_state=42
- 5-fold stratified cross-validation on the training set
- Test set used only once, at the end
- Default hyperparameters, no tuning in this step

## Results

| Model            | CV accuracy       | Test accuracy | Inference (ms/sample) | Model size (KB) |
| ---------------- | ----------------- | ------------- | --------------------- | --------------- |
| RandomForest     | 0.9807 +/- 0.0014 | 0.9792        | 0.0133                | 15456.4         |
| GradientBoosting | 0.9783 +/- 0.0016 | 0.9760        | 0.0113                | 513.4           |
| MLP              | 0.9780 +/- 0.0014 | 0.9759        | 0.0033                | 91.2            |
| DecisionTree     | 0.9721 +/- 0.0012 | 0.9699        | 0.0003                | 171.6           |
| KNN              | 0.9608 +/- 0.0012 | 0.9617        | 0.0111                | 3617.4          |
| SVM_RBF          | 0.9512 +/- 0.0034 | 0.9586        | 1.0810                | 829.0           |

## Selected model

Random Forest is selected for the final deployment.

Reasons:

- highest test accuracy
- highest cross-validation mean
- lowest cross-validation variance
- inference 80 times faster than SVM
- acceptable model size for a Raspberry Pi 4

## Rejected models

- SVM: too slow at inference (1.08 ms per sample).
- KNN: large model size and lower accuracy.
- MLP: longer training for lower accuracy than Random Forest.
- DecisionTree: lower accuracy, less robust.
- GradientBoosting: slower training, lower accuracy than Random Forest.
