# Reproducibility Guide

## Experimental Configuration

- Train-test split: 80% training and 20% testing
- Random state: 42
- Cross-validation: 5 folds
- Tuning method: GridSearchCV
- Tuning score: Weighted F1

## Dataset Information

- Dataset: College_student_behaviourial_data.csv
- Target variable: risk_level
- Classes: High Risk and Low Risk

## Data Leakage Prevention

The following columns are excluded from model features:

- student_id
- GPA
- avg_course_grade
- enrollment_status
- risk_level

## Reproducibility Checklist

- Record the dataset source and version.
- Record the date of dataset access.
- Record Python and package versions.
- Keep the random seed consistent.
- Save model evaluation results.
- Document any preprocessing changes.
