import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from config import MODEL_DIR, RESULTS_DIR, RANDOM_STATE, CV_FOLDS
from data_preprocessing import load_dataset, prepare_data


def tune_xgboost():
    df = load_dataset()
    X_train, X_test, y_train, y_test, preprocessor, _, _ = prepare_data(df)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="binary:logistic" if len(encoder.classes_) == 2 else "multi:softprob",
            eval_metric="logloss" if len(encoder.classes_) == 2 else "mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    param_grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.03, 0.1],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, param_grid, scoring="f1_weighted",
        cv=cv, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_encoded)

    joblib.dump(search.best_estimator_, MODEL_DIR / "tuned_xgboost.joblib")
    joblib.dump(encoder, MODEL_DIR / "tuned_xgboost_label_encoder.joblib")

    with open(RESULTS_DIR / "tuning_results.txt", "w") as f:
        f.write(f"Best parameters: {search.best_params_}\n")
        f.write(f"Best cross-validation weighted F1: {search.best_score_:.4f}\n")

    print("Best parameters:", search.best_params_)
    print("Best CV weighted F1:", round(search.best_score_, 4))
    print("Saved tuned model.")
    return search.best_estimator_, encoder, X_test, y_test


if __name__ == "__main__":
    tune_xgboost()
