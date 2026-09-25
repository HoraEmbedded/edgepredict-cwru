# Machine Learning Model

## Algorithm

Random Forest Classifier with 100 trees.

## Input features

- rms
- crest_factor
- kurtosis
- skewness

## Output classes

- Normal
- InnerRace
- OuterRace
- Ball

## Training

Data is split 80 percent train, 20 percent test, with stratification.

The model is evaluated using accuracy, confusion matrix and classification report.

## Target

Accuracy above 95 percent.

## Artifacts

- models/cwru_model.pkl: serialized model
- models/cwru_model_meta.json: features, classes, accuracy

These files are not versioned on GitHub.
