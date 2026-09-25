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

## Results

Best parameters: <to fill>

Best CV accuracy: <to fill>

Test accuracy with tuned model: <to fill>

## Comparison with default model

Default Random Forest test accuracy: 0.9792

Tuned Random Forest test accuracy: <to fill>


## Note on tuning location

The GridSearchCV tuning was performed on the development machine, not on the Raspberry Pi.

Reason: the Pi 4 without active cooling reaches thermal limits under sustained multi-core load (observed 84.7 degrees Celsius and soft temperature limit throttling during a first attempt).

The tuned model was then transferred to the Pi for deployment and inference measurement.

This reflects a standard MLOps workflow: train and tune off-device, deploy the final artifact on the target.
