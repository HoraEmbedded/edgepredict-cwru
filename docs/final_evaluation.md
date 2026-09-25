# Final Evaluation

The tuned Random Forest model was evaluated on the held-out test set.

## Test accuracy

0.9779 on 10957 test windows.

## Per-class metrics

| Class     | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Ball      | 0.96      | 0.96   | 0.96     | 3019    |
| InnerRace | 0.99      | 0.99   | 0.99     | 3020    |
| Normal    | 1.00      | 1.00   | 1.00     | 2647    |
| OuterRace | 0.96      | 0.96   | 0.96     | 2271    |

Macro average F1: 0.98

## Feature importance

| Feature      | Importance |
| ------------ | ---------- |
| RMS          | 0.6512     |
| Kurtosis     | 0.1843     |
| Skewness     | 0.0951     |
| Crest Factor | 0.0694     |

## Observations

- RMS dominates, which is consistent with the physics of bearing faults: defects increase the overall vibration energy.
- Kurtosis captures the impulsive nature of defects.
- Crest Factor is redundant with RMS on this dataset, which explains its low importance.
- Normal class is almost perfectly separated from the fault classes.
- Ball and OuterRace show the highest mutual confusion, which is physically expected because their vibration signatures overlap.

## Figures

- report/figures/confusion_matrix.png
- report/figures/feature_importance.png
- report/figures/roc_curves.png
