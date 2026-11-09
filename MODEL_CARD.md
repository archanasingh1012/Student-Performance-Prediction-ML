# Model Card

## Project
Student Academic Risk Prediction

## Objective
Classify students into High Risk and Low Risk categories using machine learning.

## Prediction Target
risk_level

## Model Development
The project uses a machine-learning pipeline with preprocessing and classification.

Model evaluation includes:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Model Tuning
GridSearchCV with stratified cross-validation and weighted F1 scoring is used for model tuning.

## Explainability
SHAP TreeExplainer is used for supported tree-based models.

## Limitations
- Predictions depend on dataset quality.
- Dataset bias may affect predictions.
- Target-generation methods must be verified.
- Predictions should support educators, not replace human decisions.
