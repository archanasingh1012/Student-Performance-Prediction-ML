import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from config import RESULTS_DIR, FIGURES_DIR, MODEL_DIR
from model_training import train_models


def evaluate_models():
    trained, X_test, y_test, label_encoder = train_models()
    rows = []
    reports = {}

    for name, model in trained.items():
        raw_predictions = model.predict(X_test)

        if name == "XGBoost":
            predictions = label_encoder.inverse_transform(raw_predictions.astype(int))
        else:
            predictions = raw_predictions

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

        rows.append({
            "Model": name,
            "Accuracy": accuracy,
            "Weighted_Precision": precision,
            "Weighted_Recall": recall,
            "Weighted_F1": f1,
        })
        reports[name] = classification_report(y_test, predictions, output_dict=True, zero_division=0)

        cm = confusion_matrix(y_test, predictions, labels=label_encoder.classes_)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{name}_confusion_matrix.png", dpi=300)
        plt.close()

    results = pd.DataFrame(rows).sort_values("Weighted_F1", ascending=False)
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    with open(RESULTS_DIR / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4)

    best_name = results.iloc[0]["Model"]
    best_model = trained[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    with open(MODEL_DIR / "best_model_name.txt", "w") as f:
        f.write(best_name)

    print("\nMODEL COMPARISON:\n", results.to_string(index=False))
    print(f"\nBest model by weighted F1: {best_name}")
    print(f"Results saved in: {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    evaluate_models()
