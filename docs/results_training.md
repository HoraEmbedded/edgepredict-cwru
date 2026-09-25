Total rows: 2249
Class distribution:
label
Normal       1182
InnerRace     356
OuterRace     356
Ball          355
Name: count, dtype: int64

Training set: 1799 samples
Test set: 450 samples

Training Random Forest...

Evaluating...
Accuracy: 1.0000

Confusion matrix:
[[ 71   0   0   0]
 [  0  71   0   0]
 [  0   0 237   0]
 [  0   0   0  71]]

Classification report:
              precision    recall  f1-score   support

        Ball       1.00      1.00      1.00        71
   InnerRace       1.00      1.00      1.00        71
      Normal       1.00      1.00      1.00       237
   OuterRace       1.00      1.00      1.00        71

    accuracy                           1.00       450
   macro avg       1.00      1.00      1.00       450
weighted avg       1.00      1.00      1.00       450

