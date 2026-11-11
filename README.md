# Explainable Student Academic Risk Prediction

An M.Tech Data Science and Analytics project that compares machine-learning models for classifying student academic risk using behavioural and engagement features. It also uses SHAP to interpret predictions from tree-based models.
## Project objective

The objective is to analyse student behavioural and engagement data, develop machine-learning classification models, compare their performance, and interpret model predictions using Explainable AI (SHAP).

The project aims to identify important factors associated with academic risk and support transparent, data-driven student analytics.

## 1. Dataset setupc:

Place the dataset file here:

`data/College_student_behaviourial_data.csv`

The project expects the target column `risk_level`.

Before publishing or sharing the dataset, check its licence and usage conditions. The CSV is excluded from Git by `.gitignore` by default. Do not commit identifiable or confidential student information.

## 2. Feature selection and leakage caution

The configuration excludes `student_id`, `GPA`, `avg_course_grade`, `enrollment_status`, and `risk_level` from model predictors. This is a precaution against identifier use and possible target leakage.

**Research limitation:** verify the dataset documentation to understand exactly how `risk_level` was assigned. If it was derived directly from one or more input variables, model performance may measure how well the model reproduces those rules rather than how well it predicts future risk. Do not describe the task as future prediction unless the dataset contains a genuine time-based future outcome.

## 3. Create a virtual environment (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

## 4. Run the project

Run all main analysis and model comparisons:

```powershell
python main.py
```

Run XGBoost hyperparameter tuning:

```powershell
python tune_model.py
```

Generate SHAP explanation (requires a tree-based best model):

```powershell
python explainability.py
```

Generate an example prediction using the saved best model:

```powershell
python predict_student.py
```

## 5. Project structure

```text
student_performance_mtech_project/
├── data/
├── models/
├── results/
│   └── figures/
├── config.py
├── data_preprocessing.py
├── eda.py
├── model_training.py
├── evaluation.py
├── tune_model.py
├── explainability.py
├── predict_student.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Methods

- Preprocessing: median imputation and standardisation for numeric columns; most-frequent imputation and one-hot encoding for categorical columns.
- Split: stratified train/test split.
- Models: Logistic Regression, Random Forest, Extra Trees, and XGBoost.
- Evaluation: accuracy, weighted precision, weighted recall, weighted F1-score, classification reports, and confusion matrices.
- Tuning: stratified cross-validation with GridSearchCV and weighted F1 scoring.
- Explainability: SHAP TreeExplainer for supported tree models.

## 7. Reproducibility and research reporting

Record the dataset source/version, date accessed, feature definitions, label-generation method, Python/package versions, random seed, train/test split, and all evaluation results. Report per-class precision/recall/F1 and the confusion matrix, not accuracy alone. Discuss class imbalance, limitations, and possible data leakage.

## 8. GitHub

Do not upload `.venv`, generated model files, or private student data. The dataset is ignored by default. Upload it only if its licence explicitly permits redistribution and it contains no confidential personal information.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard
for predicting student academic risk.

### Dashboard Features
- Student information input form
- Academic risk prediction
- Prediction probability visualization
- Interactive charts and tables

### How to Run the Dashboard

1. Install the required libraries:

   pip install -r requirements.txt

2. Start the Streamlit application:

   streamlit run app.py

3. Open the local URL displayed in the terminal.

### Machine Learning Model

The dashboard uses the best-performing trained model
saved in the models folder.
