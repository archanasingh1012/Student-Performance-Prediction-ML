from eda import run_eda
from evaluation import evaluate_models


def main():
    print("=" * 65)
    print("STUDENT ACADEMIC RISK PREDICTION - MACHINE LEARNING PIPELINE")
    print("=" * 65)

    print("\n[1/2] Running exploratory data analysis...")
    run_eda()

    print("\n[2/2] Training and evaluating machine learning models...")
    evaluate_models()

    print("\nPipeline completed. Check the results and figures folders.")
    print("Next optional steps: python tune_model.py, then python explainability.py")


if __name__ == "__main__":
    main()
