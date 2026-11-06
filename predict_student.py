import joblib
import pandas as pd

from config import MODEL_DIR
from data_preprocessing import load_dataset, DROP_FROM_FEATURES, TARGET


def predict_example():
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run evaluation.py before making predictions.")

    df = load_dataset()
    feature_columns = [c for c in df.columns if c not in DROP_FROM_FEATURES]
    example = df[feature_columns].iloc[[0]].copy()

    model = joblib.load(model_path)
    prediction = model.predict(example)[0]

    model_name = (MODEL_DIR / "best_model_name.txt").read_text().strip()
    if model_name == "XGBoost":
        encoder = joblib.load(MODEL_DIR / "xgboost_label_encoder.joblib")
        prediction = encoder.inverse_transform([int(prediction)])[0]

    print("Example student input:\n", example.to_string(index=False))
    print("\nPredicted risk level:", prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(example)[0]
        if model_name == "XGBoost":
            classes = joblib.load(MODEL_DIR / "xgboost_label_encoder.joblib").classes_
        else:
            classes = model.named_steps["classifier"].classes_
        print("Predicted class probabilities:")
        for label, probability in zip(classes, probabilities):
            print(f"{label}: {probability:.3f}")


if __name__ == "__main__":
    predict_example()
