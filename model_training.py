import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from config import MODEL_DIR, RANDOM_STATE
from data_preprocessing import load_dataset, prepare_data


def train_models():
    df = load_dataset()
    X_train, X_test, y_train, y_test, preprocessor, _, _ = prepare_data(df)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    trained = {}
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")
        print(f"Trained and saved: {name}")

    # XGBoost needs numeric class labels.
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic" if len(label_encoder.classes_) == 2 else "multi:softprob",
            eval_metric="logloss" if len(label_encoder.classes_) == 2 else "mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    xgb_pipeline.fit(X_train, y_train_encoded)
    trained["XGBoost"] = xgb_pipeline

    joblib.dump(xgb_pipeline, MODEL_DIR / "XGBoost.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "xgboost_label_encoder.joblib")
    joblib.dump(list(X_train.columns), MODEL_DIR / "feature_columns.joblib")

    return trained, X_test, y_test, label_encoder


if __name__ == "__main__":
    train_models()
