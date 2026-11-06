import joblib
import shap
import matplotlib.pyplot as plt

from config import MODEL_DIR, FIGURES_DIR
from data_preprocessing import load_dataset, prepare_data


def run_shap():
    df = load_dataset()
    X_train, X_test, y_train, y_test, _, _, _ = prepare_data(df)

    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run evaluation.py first to create best_model.joblib.")

    model = joblib.load(model_path)
    best_name = (MODEL_DIR / "best_model_name.txt").read_text().strip()

    # SHAP TreeExplainer is designed for tree-based models.
    if best_name not in ["RandomForest", "ExtraTrees", "XGBoost"]:
        print(f"Best model is {best_name}. SHAP TreeExplainer is not used for this model.")
        print("For this run, select a tree model or use a model-agnostic explainer.")
        return

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(X_test)

    # SHAP works more reliably with a dense matrix for this dataset size.
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    plt.figure()
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[-1], transformed, feature_names=feature_names, show=False)
    elif getattr(shap_values, "ndim", 0) == 3:
        shap.summary_plot(shap_values[:, :, -1], transformed, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("SHAP summary plot saved.")


if __name__ == "__main__":
    run_shap()
