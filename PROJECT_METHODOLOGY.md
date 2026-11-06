# Project Methodology

## 1. Objective
To classify student academic risk using machine learning and understand model predictions using Explainable AI.

## 2. Dataset
Dataset: College_student_behaviourial_data.csv
Target variable: risk_level
Classes: High Risk and Low Risk

## 3. Data Preprocessing
- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- Standardisation of numerical features.
- One-hot encoding of categorical features.
- Exclusion of identifier and potentially leakage-prone columns.

## 4. Exploratory Data Analysis
- Dataset structure analysis.
- Missing value analysis.
- Feature distribution analysis.
- Academic risk distribution visualisation.

## 5. Machine Learning Algorithms
- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost

## 6. Model Evaluation
- Accuracy
- Precision
- Recall
- Weighted F1-score
- Confusion Matrix
- Classification Report

## 7. Hyperparameter Tuning
GridSearchCV and stratified cross-validation are used for model tuning.

## 8. Explainable AI
SHAP is used to understand feature contributions to predictions from supported tree-based models.

## 9. Workflow
1. Dataset loading
2. Exploratory Data Analysis
3. Data preprocessing
4. Model training
5. Model evaluation
6. Hyperparameter tuning
7. SHAP explainability
8. Student risk prediction

## 10. Research Considerations
Verify target generation, avoid data leakage, document dataset sources, and report model limitations.
