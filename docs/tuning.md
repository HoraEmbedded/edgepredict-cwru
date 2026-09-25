# Random Forest Hyperparameter Tuning

## Protocol

- 80/20 stratified split, random_state=42
- 3-fold stratified cross-validation on the training set
- Test set opened only once, at the end
- GridSearchCV with accuracy scoring

## Grid

- n_estimators: [50, 100, 200]
- max_depth: [None, 20, 30]
- min_samples_split: [2, 5]
- min_samples_leaf: [1, 2]

Total combinations: 36
Total training runs: 108

## Results

Best parameters:

- n_estimators: 50
- max_depth: 20
- min_samples_split: 2
- min_samples_leaf: 1

Best CV accuracy: 0.9803
CV standard deviation: 0.0002

Default Random Forest CV accuracy (Part B): 0.9807
Default Random Forest test accuracy (Part B): 0.9792

## Interpretation

The GridSearch did not improve the cross-validation accuracy over the default Random Forest parameters.

The gap between the tuned model and the default model is 0.0004, which is within the cross-validation noise.

This result confirms that the Scikit-Learn default hyperparameters for Random Forest were already near-optimal for this dataset.

## Note on tuning location

The GridSearchCV tuning was performed on the development machine, not on the Raspberry Pi.

Reason: the Pi 4 without active cooling reaches thermal limits under sustained multi-core load. A first attempt triggered soft temperature limit throttling at 84.7 degrees Celsius (throttled=0xe0008).

The tuned model was then transferred to the Pi for deployment and inference measurement.

This reflects a standard MLOps workflow: train and tune off-device, deploy the final artifact on the target.
