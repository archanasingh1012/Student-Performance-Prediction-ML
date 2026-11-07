# Project Workflow: Student Academic Risk Prediction

## 1. Problem Definition
The project aims to classify students into High Risk and Low Risk categories using machine-learning techniques and behavioural and engagement-related features.

## 2. Dataset Collection
- Dataset: College_student_behaviourial_data.csv
- File format: CSV
- Target variable: risk_level
- Classes: High Risk and Low Risk

## 3. Exploratory Data Analysis
The dataset is explored to understand:
- Dataset dimensions and data types
- Missing values
- Class distribution
- Numerical feature distributions
- Relationships between features

## 4. Data Preprocessing
The preprocessing stage includes:
- Separating input features and target variable
- Removing identifier and potentially leakage-related columns
- Handling missing numerical and categorical values
- Standardising numerical features
- Encoding categorical features

## 5. Dataset Splitting
The dataset is divided into training and testing sets using stratified sampling to preserve class proportions.

## 6. Model Development
Four machine-learning classification algorithms are compared:

1. Logistic Regression
2. Random Forest
3. Extra Trees
4. XGBoost

## 7. Model Evaluation
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

## 8. Hyperparameter Tuning
GridSearchCV and stratified cross-validation are used for XGBoost hyperparameter tuning, with weighted F1-score as the scoring metric.

## 9. Explainable AI
SHAP is used to interpret predictions from supported tree-based models and understand the contribution of individual features.

## 10. Final Output
The project produces model evaluation results, visualisations, saved model files, and explanations to support transparent student academic risk analysis.

## 11. Research Considerations
The target variable generation process should be verified to avoid data leakage and misleading prediction claims. Model performance should be reported with class-wise metrics and limitations.